#!/usr/bin/env python3
"""
Drone Swarm Trajectory Tracker — Ground Truth Detection Pipeline (run2.py)
==========================================================================
Uses Ground Truth annotations SOLELY as the Object Detector (raw bounding boxes),
while the entire multi-target tracking, dynamic velocity-adaptive gating,
Kalman kinematic prediction, and learned Deep Q-Network data association
are performed 100% by OUR RL TRACKING MODEL.

Key Architecture:
  1. Detection Source: Ground Truth Bounding Boxes [x1, y1, x2, y2]
     (All dataset track IDs are stripped; detections are shuffled per frame to
      ensure no frame-to-frame ordering correlation).
  2. Multi-Target Tracking Model: OUR MODEL
     - 2D Constant-Velocity Kalman Filter (state: [x, y, vx, vy])
     - Velocity-Adaptive Dynamic Gating: R_dynamic = clip(20 + 2*v + 4*missed, 20, 65) px
     - 8-subregion polar spatial discretization
     - Learned Deep Q-Network (DroneQNetwork) candidate scoring: Q in [0, 1]
     - Global Hungarian Bipartite Assignment: min sum(dist * (1.5 - Q))
  3. Continuous Spatial Trajectory Tracing: Real-time neon flight paths
  4. Real-time Rendering & Telemetry HUD Overlay

Usage:
  # Run on default test video with ground truth detection inputs
  python run2.py

  # Run on specific benchmark test sequence
  python run2.py --video results/uavswarm_test_04_raw.mp4
  python run2.py --video results/uavswarm_test_08_raw.mp4
  python run2.py --video results/uavswarm_test_14_raw.mp4

  # Run with an explicit ground truth annotation file
  python run2.py --video results/uavswarm_test_raw.mp4 --gt results/uavswarm_02_gt.txt

  # Run headless (outputs tracked MP4, CSV, and JSON directly)
  python run2.py --video results/uavswarm_test_04_raw.mp4 --headless
"""

import os
import sys
import math
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

# ===================== CONFIG =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = CURRENT_DIR

DEFAULT_VIDEO = os.path.join(REPO_ROOT, "results", "uavswarm_test_raw.mp4")
DEFAULT_POLICY = os.path.join(REPO_ROOT, "checkpoints", "drone_tracker_policy_best.pth")

# Automatic mapping from test videos to known benchmark GT files
KNOWN_GT_MAP = {
    "uavswarm_test_raw.mp4": os.path.join(REPO_ROOT, "results", "uavswarm_02_gt.txt"),
    "uavswarm_test_04_raw.mp4": os.path.join(REPO_ROOT, "results", "uavswarm_04_gt.txt"),
    "uavswarm_test_08_raw.mp4": os.path.join(REPO_ROOT, "results", "uavswarm_08_gt.txt"),
    "uavswarm_test_14_raw.mp4": os.path.join(REPO_ROOT, "results", "uavswarm_14_gt.txt"),
}

# Velocity-Adaptive Dynamic Gating Parameters (Matching run.py & Shrimp architecture)
BASE_GATE = 20.0       # Base envelope (px)
VEL_COEF = 2.0        # Velocity scaling factor (2.0 * speed)
MISSED_COEF = 4.0     # Uncertainty expansion factor (4.0 * missed)
MIN_GATE = 20.0       # Minimum gate for hovering drones
MAX_GATE = 65.0       # Maximum gate cap (strictly below 60-90px inter-drone spacing)
MAX_MISSED = 8        # Frames before an unobserved track is terminated
TRAIL_LENGTH = 80     # Length of trajectory history trails to render


# ===================== GROUND TRUTH OBJECT DETECTOR =====================
class GroundTruthObjectDetector:
    """
    Extracts purely anonymous bounding boxes and centroids [cx, cy] from ground truth.
    All dataset track IDs are completely discarded.
    Detections are intentionally randomized/shuffled per frame to eliminate any dataset ordering.
    """
    def __init__(self, gt_path: str):
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        self.gt_path = gt_path
        self.frames_detections = {}
        self._load_annotations()

    def _load_annotations(self):
        print(f"📦 Loading ground-truth object detection bounding boxes from: {self.gt_path}")

        # Check if CSV with headers or MOT text
        with open(self.gt_path, 'r') as f:
            first_line = f.readline().strip()

        has_header = any(h in first_line.lower() for h in ['frame', 'track_id', 'bb_left', 'cx', 'cy'])

        if has_header:
            df = pd.read_csv(self.gt_path)
            for _, row in df.iterrows():
                f_idx = int(row['frame'])
                if 'cx' in row and 'cy' in row:
                    cx = float(row['cx'])
                    cy = float(row['cy'])
                    x1 = float(row.get('x1', cx - 10))
                    y1 = float(row.get('y1', cy - 10))
                    x2 = float(row.get('x2', cx + 10))
                    y2 = float(row.get('y2', cy + 10))
                elif 'bb_left' in row and 'bb_top' in row:
                    x1 = float(row['bb_left'])
                    y1 = float(row['bb_top'])
                    w_box = float(row['width'])
                    h_box = float(row['height'])
                    x2 = x1 + w_box
                    y2 = y1 + h_box
                    cx = x1 + w_box / 2.0
                    cy = y1 + h_box / 2.0
                else:
                    continue

                if f_idx not in self.frames_detections:
                    self.frames_detections[f_idx] = []
                self.frames_detections[f_idx].append({
                    'cx': cx,
                    'cy': cy,
                    'box': [x1, y1, x2, y2],
                    'score': 1.0
                })
        else:
            # MOT16 format: frame, id, bb_left, bb_top, width, height, conf, ...
            with open(self.gt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) < 6:
                        continue

                    f_idx = int(float(parts[0]))
                    bb_left = float(parts[2])
                    bb_top = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])

                    if len(parts) >= 7:
                        try:
                            if float(parts[6]) == 0:
                                continue
                        except ValueError:
                            pass

                    x1 = bb_left
                    y1 = bb_top
                    x2 = bb_left + width
                    y2 = bb_top + height
                    cx = x1 + width / 2.0
                    cy = y1 + height / 2.0

                    if f_idx not in self.frames_detections:
                        self.frames_detections[f_idx] = []
                    # Strictly anonymous bounding boxes and centroids (NO track IDs)
                    self.frames_detections[f_idx].append({
                        'cx': cx,
                        'cy': cy,
                        'box': [x1, y1, x2, y2],
                        'score': 1.0
                    })

        total_dets = sum(len(v) for v in self.frames_detections.values())
        print(f"✅ Loaded {total_dets} anonymous drone bounding boxes across {len(self.frames_detections)} frames.")

    def get_detections_for_frame(self, frame_idx: int) -> list:
        """
        Returns anonymous detections for the given frame.
        Shuffles the list to guarantee that tracking does NOT exploit any dataset ordering.
        """
        dets = list(self.frames_detections.get(frame_idx, []))
        random.shuffle(dets)
        return dets


# ===================== OUR MODEL: DRONE Q-NETWORK =====================
class DroneQNetwork(nn.Module):
    """
    5-feature RL Q-Network scoring candidate detections for learned data association.
    Input features: [dx/gate, dy/gate, region_id/7.0, vx/gate, vy/gate]
    """
    def __init__(self, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== OUR MODEL: 2D KALMAN FILTER =====================
class SimpleKF:
    """Constant-velocity 2D Kalman filter for smooth centroid state estimation."""
    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 5.0
        self.Q = np.eye(4, dtype=np.float32) * 1.0

    def predict(self) -> tuple:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float):
        z = np.array([cx, cy], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P


# ===================== OUR MODEL: VELOCITY-ADAPTIVE RL TRACKER =====================
class DroneRLTracker:
    """
    Multi-Target Drone Swarm Tracker:
      1. Dynamic velocity-adaptive gating: R_dyn = clip(20 + 2*v + 4*missed, 20, 65)
      2. 8-subregion spatial discretization
      3. Deep Q-Network (RL) candidate scoring
      4. Hungarian global optimal bipartite association
      5. Autonomous track ID management and trajectory tracing
    """
    def __init__(self, policy_path: str, device: torch.device):
        self.device = device
        self.policy = DroneQNetwork().to(device)
        if os.path.exists(policy_path):
            ckpt = torch.load(policy_path, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.policy.load_state_dict(ckpt['model'])
            else:
                self.policy.load_state_dict(ckpt)
            print(f"✅ Loaded our trained RL tracker policy from {policy_path}")
        else:
            print(f"⚠️  Policy checkpoint not found at {policy_path}, running with initialized weights.")
        self.policy.eval()

        self.tracks = {}          # {track_id: {'kf', 'box', 'missed', 'cx', 'cy', 'vx', 'vy', 'history', 'hits'}}
        self.trajectories = {}    # {track_id: [{'frame', 'cx', 'cy', 'vx', 'vy', 'box'}]}
        self.next_id = 1          # Track IDs are generated solely by our model

    def compute_dynamic_gate(self, vx: float, vy: float, missed: int = 0) -> float:
        """Scales the association search gate dynamically based on current velocity and missed frames."""
        speed = math.hypot(vx, vy)
        gate = BASE_GATE + VEL_COEF * speed + MISSED_COEF * missed
        return float(np.clip(gate, MIN_GATE, MAX_GATE))

    def get_subregion(self, dx: float, dy: float, gate: float) -> int:
        """Encodes relative offset into 8 spatial zones (inner/outer rings x 4 quadrants)."""
        dist = math.hypot(dx, dy)
        if dist > gate:
            return -1
        ring = 0 if dist <= gate / 2.0 else 1
        angle = math.atan2(dy, dx)
        if -math.pi / 4 <= angle < math.pi / 4:
            q = 0
        elif math.pi / 4 <= angle < 3 * math.pi / 4:
            q = 1
        elif angle >= 3 * math.pi / 4 or angle < -3 * math.pi / 4:
            q = 2
        else:
            q = 3
        return ring * 4 + q

    def process_frame(self, detections: list, frame_idx: int) -> list:
        """
        Takes raw, anonymous object detections and performs association entirely
        using our model (Kalman prediction + Dynamic Gating + DroneQNetwork + Hungarian).
        """
        tids = list(self.tracks.keys())
        active = []

        # If no active tracks, initialize new tracks from all incoming detections
        if not tids:
            for det in detections:
                tid = self.next_id
                self.tracks[tid] = {
                    'kf': SimpleKF(det['cx'], det['cy']),
                    'box': det['box'],
                    'cx': det['cx'], 'cy': det['cy'],
                    'vx': 0.0, 'vy': 0.0,
                    'hits': 1, 'missed': 0,
                    'history': [(int(det['cx']), int(det['cy']))],
                }
                self.trajectories[tid] = [{
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                }]
                self.next_id += 1
            return []

        # Build Hungarian assignment cost matrix using our RL model
        cost_matrix = np.full((len(tids), len(detections)), 1e5)
        gates = []

        for i, tid in enumerate(tids):
            data = self.tracks[tid]
            kf = data['kf']
            pred_cx, pred_cy = kf.predict()
            vx = data['vx']
            vy = data['vy']
            dyn_gate = self.compute_dynamic_gate(vx, vy, data['missed'])
            gates.append(dyn_gate)

            cands = []
            for j, det in enumerate(detections):
                dx = det['cx'] - pred_cx
                dy = det['cy'] - pred_cy
                dist = math.hypot(dx, dy)
                if dist <= dyn_gate:
                    reg = self.get_subregion(dx, dy, dyn_gate)
                    cands.append((j, dx, dy, dist, reg))

            if cands:
                feats = torch.FloatTensor([[
                    dx / dyn_gate,
                    dy / dyn_gate,
                    (reg if reg >= 0 else 0) / 7.0,
                    vx / dyn_gate,
                    vy / dyn_gate,
                ] for _, dx, dy, _, reg in cands]).to(self.device)

                with torch.no_grad():
                    q_vals = self.policy(feats).cpu().numpy().flatten()

                for (j, dx, dy, dist, _), q in zip(cands, q_vals):
                    # Combine spatial Euclidean distance with our learned RL association score
                    cost_matrix[i, j] = dist * (1.5 - float(q))

        # Solve optimal bipartite matching globally
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            tid = tids[r]
            dyn_gate = gates[r]
            if cost_matrix[r, c] < 1e4:
                det = detections[c]
                data = self.tracks[tid]
                step_d = math.hypot(det['cx'] - data['cx'], det['cy'] - data['cy'])
                if step_d > dyn_gate + 5.0:
                    continue

                vx = det['cx'] - data['cx']
                vy = det['cy'] - data['cy']
                data['vx'] = 0.7 * data['vx'] + 0.3 * vx
                data['vy'] = 0.7 * data['vy'] + 0.3 * vy
                data['cx'] = det['cx']
                data['cy'] = det['cy']
                data['box'] = det['box']
                data['kf'].update(det['cx'], det['cy'])
                data['history'].append((int(det['cx']), int(det['cy'])))
                if len(data['history']) > TRAIL_LENGTH:
                    data['history'].pop(0)

                data['hits'] += 1
                data['missed'] = 0

                if tid not in self.trajectories:
                    self.trajectories[tid] = []
                self.trajectories[tid].append({
                    'frame': frame_idx,
                    'cx': float(det['cx']),
                    'cy': float(det['cy']),
                    'vx': float(vx),
                    'vy': float(vy),
                    'box': [float(b) for b in det['box']],
                })

                matched_tracks.add(tid)
                matched_dets.add(c)

                if data['hits'] >= 2:
                    active.append({
                        'track_id': tid,
                        'box': det['box'],
                        'cx': det['cx'],
                        'cy': det['cy'],
                        'history': list(data['history']),
                        'gate': dyn_gate,
                        'speed': math.hypot(data['vx'], data['vy'])
                    })

        # Update missed frames and remove stale tracks
        for tid in tids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > MAX_MISSED:
                    del self.tracks[tid]

        # Spawn new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.tracks[tid] = {
                    'kf': SimpleKF(det['cx'], det['cy']),
                    'box': det['box'],
                    'cx': det['cx'], 'cy': det['cy'],
                    'vx': 0.0, 'vy': 0.0,
                    'hits': 1, 'missed': 0,
                    'history': [(int(det['cx']), int(det['cy']))],
                }
                if tid not in self.trajectories:
                    self.trajectories[tid] = []
                self.trajectories[tid].append({
                    'frame': frame_idx,
                    'cx': float(det['cx']),
                    'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                })
                self.next_id += 1

        return active


# ===================== MAIN PIPELINE =====================
def main():
    parser = argparse.ArgumentParser(description="Drone Swarm RL Trajectory Tracker — Ground Truth Detection Pipeline")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="Path to input test video MP4")
    parser.add_argument("--gt", type=str, default="", help="Path to ground truth annotation file (MOT gt.txt or CSV)")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Path to our trained RL policy checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="", help="Path to output tracked video MP4")
    parser.add_argument("--csv", type=str, default="", help="Path to output trajectory CSV")
    parser.add_argument("--json", type=str, default="", help="Path to output trajectory JSON")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without opening GUI display window")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    video_filename = os.path.basename(video_path)
    base_stem = os.path.splitext(video_filename)[0]

    # Resolve Ground Truth file: user explicit argument -> known map -> fallback heuristic
    gt_file = args.gt
    if not gt_file:
        if video_filename in KNOWN_GT_MAP:
            gt_file = KNOWN_GT_MAP[video_filename]
        else:
            candidates = [
                os.path.join(REPO_ROOT, "results", f"{base_stem}_gt.txt"),
                os.path.join(REPO_ROOT, "results", f"{base_stem}.csv"),
                os.path.join(os.path.dirname(video_path), "gt", "gt.txt"),
                os.path.join(os.path.dirname(video_path), "gt.txt"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    gt_file = cand
                    break

    if not gt_file or not os.path.exists(gt_file):
        print(f"❌ Error: No matching ground truth annotations found for video: {video_path}")
        print(f"   Please specify explicitly using --gt path/to/gt.txt")
        sys.exit(1)

    out_video_path = args.output or os.path.join(REPO_ROOT, "results", f"tracked_gt_{base_stem}.mp4")
    out_csv_path = args.csv or os.path.join(REPO_ROOT, "results", f"trajectories_gt_{base_stem}.csv")
    out_json_path = args.json or os.path.join(REPO_ROOT, "results", f"trajectories_gt_{base_stem}.json")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 70)
    print("🛸 DRONE SWARM TRACKER — GT DETECTIONS + OUR RL TRACKING MODEL")
    print("=" * 70)
    print(f"🎥 Input Video    : {video_path}")
    print(f"📦 Object Detector: Ground Truth Bounding Boxes ({gt_file})")
    print(f"🧠 Tracking Model : OUR RL DroneQNetwork Policy ({args.policy})")
    print(f"🎯 Dynamic Gating : R = clip({BASE_GATE:.0f} + {VEL_COEF:.1f}*v + {MISSED_COEF:.1f}*k, {MIN_GATE:.0f}, {MAX_GATE:.0f}) px")
    print(f"💾 Output Video   : {out_video_path}")
    print(f"📊 Trajectory CSV : {out_csv_path}")

    # Load Ground Truth Detector (provides raw anonymous bounding boxes)
    detector = GroundTruthObjectDetector(gt_file)

    # Initialize OUR RL Tracker (performs 100% of data association and flight path tracking)
    tracker = DroneRLTracker(args.policy, device)

    # Open Video Capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    window_title = f"RL Drone Tracker (Our Model Tracking GT Detections) — {base_stem}"
    if not args.headless:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(1280, int(w * 1.5)), min(720, int(h * 1.5)))
        print(f"🖥️  GUI Window active: '{window_title}' (Press 'q'/ESC to quit, SPACE to pause/resume)")

    frame_idx = 0
    t_start = time.time()
    active_counts = []

    print(f"\n🚀 Running our RL model tracking on {total_frames} frames ({w}x{h} @ {fps:.1f} FPS)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. RETRIEVE RAW ANONYMOUS BOUNDING BOXES FOR THIS FRAME (SHUFFLED)
        detections = detector.get_detections_for_frame(frame_idx)

        # 2. RUN OUR REINFORCEMENT LEARNING TRACKER (Kalman + Dynamic Gate + Q-Network + Hungarian)
        active_tracks = tracker.process_frame(detections, frame_idx)
        active_counts.append(len(active_tracks))

        # 3. RENDER TRAJECTORIES & OUR MODEL'S ASSIGNED TRACKS ON FRAME
        for trk in active_tracks:
            tid = trk['track_id']
            box = trk['box']
            hist = trk['history']
            gate_r = int(trk['gate'])

            # Deterministic neon color per model-tracked ID
            color = (
                int((tid * 67 + 50) % 256),
                int((tid * 137 + 100) % 256),
                int((tid * 223 + 150) % 256)
            )

            # Continuous motion trajectory trail generated by our model
            if len(hist) > 1:
                for p in range(1, len(hist)):
                    cv2.line(frame, hist[p - 1], hist[p], color, 2, cv2.LINE_AA)

            # Drone centroid marker
            cx, cy = int(trk['cx']), int(trk['cy'])
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # Our model's Velocity-Adaptive Dynamic Gate ring
            cv2.circle(frame, (cx, cy), gate_r, color, 1, cv2.LINE_AA)

            # Drone bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Model Track ID Badge with dynamic gating radius
            label = f"#{tid} ({gate_r}px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 2, max(th + 2, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Telemetry HUD Overlay
        cv2.rectangle(frame, (10, 10), (430, 105), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (430, 105), (0, 255, 128), 1)
        cv2.putText(frame, f"Active Drones Traced: {len(active_tracks)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} ({fps:.0f} FPS playback)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detector: GT BBoxes | Tracker: Our RL Model", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

        writer.write(frame)

        # 4. LIVE INTERACTIVE DISPLAY
        if not args.headless:
            cv2.imshow(window_title, frame)
            delay = max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF
            if key == 27 or key == ord('q'):
                print(f"\n⏹️  User pressed 'q' / ESC — stopping playback.")
                break
            elif key == ord(' '):
                print(f"⏸️  Paused at frame {frame_idx}. Press SPACE to resume...")
                while True:
                    k2 = cv2.waitKey(100) & 0xFF
                    if k2 == ord(' ') or k2 == 27 or k2 == ord('q'):
                        break

    cap.release()
    writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0

    # Save trajectory dataset generated entirely by our tracker
    clean_rows = []
    for tid, pts in tracker.trajectories.items():
        if len(pts) >= 2:
            for pt in pts:
                clean_rows.append({
                    'track_id': tid,
                    'frame': pt['frame'],
                    'cx': pt['cx'], 'cy': pt['cy'],
                    'vx': pt['vx'], 'vy': pt['vy'],
                    'x1': pt['box'][0], 'y1': pt['box'][1],
                    'x2': pt['box'][2], 'y2': pt['box'][3]
                })

    df_clean = pd.DataFrame(clean_rows)
    df_clean.to_csv(out_csv_path, index=False)
    with open(out_json_path, 'w') as f:
        json.dump(tracker.trajectories, f, indent=2)

    # Print Summary Report
    print("\n" + "=" * 70)
    print("📈 TRACKING PERFORMANCE SUMMARY (OUR MODEL ON GT DETECTIONS)")
    print("=" * 70)
    print(f"⏱️  Processing Time : {elapsed:.2f} seconds ({proc_fps:.1f} FPS)")
    print(f"🎬 Frames Processed: {frame_idx} / {total_frames}")
    print(f"🛸 Persistent Tracks: {df_clean['track_id'].nunique() if not df_clean.empty else 0}")
    print(f"📍 Trajectory Points: {len(df_clean)}")
    print(f"📊 Mean Active Swarm: {np.mean(active_counts):.1f} drones/frame (Peak: {np.max(active_counts) if active_counts else 0})")
    print(f"🎥 Tracked Video   : {out_video_path} ({os.path.getsize(out_video_path)/1024/1024:.2f} MB)")
    print(f"📄 Trajectory Logs : {out_csv_path} and {out_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
