#!/usr/bin/env python3
"""
Drone Swarm Trajectory Tracker — Live Runtime Inference Pipeline
================================================================
Adapted from: Shrimp/run.py
Enhanced for: Multi-Drone Swarm Trajectory Tracing (UAVSwarm Dataset)

Key Features:
  1. Runtime D-FINE Object Detection: Detects drones directly on incoming video frames
     using fine-tuned weights (checkpoints/dfine_drone_weights.pth).
  2. Shrimp Drone RL Agent (DroneQNetwork): Learned Q-value scoring for data association.
  3. Velocity-Adaptive Dynamic Gating: R_dynamic = clip(20 + 2*v + 4*missed, 20, 65) px.
  4. Global Optimal Bipartite Association: Hungarian algorithm (scipy linear_sum_assignment).
  5. Continuous Spatial Trajectory Tracing: Full multi-target flight path curves.
  6. Live Visualization GUI: Real-time OpenCV window with neon trails, gate rings, and HUD.
  7. Video & Data Export: Saves tracked MP4, trajectory CSV, and JSON records.

Usage:
  # Run live tracking with D-FINE on default test video
  python run.py

  # Run on any arbitrary drone video
  python run.py --video results/uavswarm_test_04_raw.mp4
  python run.py --video results/uavswarm_test_08_raw.mp4
  python run.py --video results/uavswarm_test_14_raw.mp4
  python run.py --video /path/to/any_drone_video.mp4

  # Run headless (no GUI window, outputs video directly)
  python run.py --video results/uavswarm_test_04_raw.mp4 --headless
"""

import os
import sys
import math
import time
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment

# ===================== CONFIG =====================
INPUT_W = 640
INPUT_H = 640
SCORE_THRES = 0.35

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = CURRENT_DIR

DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "checkpoints", "dfine_drone_weights.pth")
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "dfine_core", "configs", "dfine", "custom", "dfine_hgnetv2_m_uavswarm.yml")
DEFAULT_POLICY = os.path.join(REPO_ROOT, "checkpoints", "drone_tracker_policy_best.pth")
DEFAULT_VIDEO = os.path.join(REPO_ROOT, "results", "uavswarm_test_raw.mp4")

# Velocity-Adaptive Dynamic Gating Parameters
BASE_GATE = 20.0       # Base envelope (px)
VEL_COEF = 2.0        # Velocity scaling factor (2.0 * speed)
MISSED_COEF = 4.0     # Uncertainty expansion factor (4.0 * missed)
MIN_GATE = 20.0       # Minimum gate for hovering drones
MAX_GATE = 65.0       # Maximum gate cap (strictly below 60-90px inter-drone spacing)
MAX_MISSED = 8        # Frames before an unobserved track is terminated
TRAIL_LENGTH = 80     # Length of trajectory history trails to render


# ===================== RUNTIME D-FINE DETECTOR =====================
class DFineDetector:
    """
    Runtime Object Detector powered by fine-tuned D-FINE (HGNetv2).
    Processes raw video frames on GPU/MPS/CPU at runtime.
    """
    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, config_path: str = DEFAULT_CONFIG,
                 device: torch.device = None, score_thresh: float = SCORE_THRES):
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.score_thresh = score_thresh

        # Ensure dfine_core is accessible on sys.path
        dfine_dir = os.path.join(REPO_ROOT, "dfine_core")
        if dfine_dir not in sys.path:
            sys.path.insert(0, dfine_dir)

        from src.core import YAMLConfig

        print(f"📦 Initializing D-FINE detector on {self.device}...")
        cfg = YAMLConfig(config_path, resume=weights_path)
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state_dict)

        self.model = cfg.model.eval().to(self.device)
        self.postprocessor = cfg.postprocessor.deploy()

        self.transform = T.Compose([
            T.Resize((INPUT_H, INPUT_W)),
            T.ToTensor()
        ])
        print("✅ D-FINE runtime detector ready.")

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Runs object detection on a single BGR video frame at runtime.
        Returns a list of dicts: [{'cx', 'cy', 'box': [x1, y1, x2, y2], 'score': float}]
        """
        h, w = frame_bgr.shape[:2]
        im_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        im_tensor = self.transform(im_pil).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[w, h]]).to(self.device)

        with torch.no_grad():
            output = self.model(im_tensor)
            labels, boxes, scores = self.postprocessor(output, orig_size)

        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        detections = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            if score >= self.score_thresh:
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                detections.append({
                    'cx': cx,
                    'cy': cy,
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score)
                })
        return detections


# ===================== DRONE Q-NETWORK =====================
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


# ===================== 2D KALMAN FILTER =====================
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


# ===================== VELOCITY-ADAPTIVE RL TRACKER =====================
class DroneRLTracker:
    """
    Multi-Target Swarm Tracker combining:
      1. Dynamic velocity-adaptive gating: R_dyn = clip(20 + 2*v + 4*missed, 20, 65)
      2. 8-subregion spatial discretization
      3. Deep Q-Network candidate scoring
      4. Hungarian global optimal bipartite association
      5. Full flight path trajectory tracing
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
            print(f"✅ Loaded RL policy from {policy_path}")
        else:
            print(f"⚠️  Policy checkpoint not found at {policy_path}, running with initialized weights.")
        self.policy.eval()

        self.tracks = {}          # {track_id: {'kf', 'box', 'missed', 'cx', 'cy', 'vx', 'vy', 'history', 'hits'}}
        self.trajectories = {}    # {track_id: [{'frame', 'cx', 'cy', 'vx', 'vy', 'box'}]}
        self.next_id = 1

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
        tids = list(self.tracks.keys())
        active = []

        # If no active tracks, initialize new tracks from all detections
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

        # Build Hungarian assignment cost matrix
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
                    # Combine spatial Euclidean distance with learned RL association score
                    cost_matrix[i, j] = dist * (1.5 - float(q))

        # Solve optimal bipartite matching
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
    parser = argparse.ArgumentParser(description="Drone Swarm RL Trajectory Tracker — Live Runtime Runner")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="Path to input test video MP4")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to fine-tuned D-FINE weights (.pth)")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Path to trained RL policy checkpoint (.pth)")
    parser.add_argument("--conf", type=float, default=SCORE_THRES, help="D-FINE object detection confidence threshold")
    parser.add_argument("--output", type=str, default="", help="Path to output tracked video MP4")
    parser.add_argument("--csv", type=str, default="", help="Path to output trajectory CSV")
    parser.add_argument("--json", type=str, default="", help="Path to output trajectory JSON")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without opening GUI display window")
    parser.add_argument("--benchmark-gt", type=str, default="", help="Optional MOT ground truth file for benchmarking")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    base_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = f"result_with_our_dfine/tracked_{base_stem}.mp4"
    out_csv_path = f"result_with_our_dfine/trajectories_{base_stem}.csv"
    out_json_path = f"result_with_our_dfine/trajectories_{base_stem}.json"

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 65)
    print("🚁 DRONE SWARM RL TRAJECTORY TRACKER")
    print("=" * 65)
    print(f"🎥 Input Video    : {video_path}")
    print(f"📦 D-FINE Model   : {args.weights} (conf >= {args.conf})")
    print(f"🧠 RL Policy      : {args.policy} (Device: {device})")
    print(f"🎯 Dynamic Gating : R = clip({BASE_GATE:.0f} + {VEL_COEF:.1f}*v + {MISSED_COEF:.1f}*k, {MIN_GATE:.0f}, {MAX_GATE:.0f}) px")
    print(f"💾 Output Video   : {out_video_path}")
    print(f"📊 Trajectory CSV : {out_csv_path}")

    # Initialize Tracker
    tracker = DroneRLTracker(args.policy, device)

    # Initialize D-FINE Detector or optional GT benchmark loader
    use_gt_benchmark = bool(args.benchmark_gt and os.path.exists(args.benchmark_gt))
    if use_gt_benchmark:
        print(f"📡 Using benchmark ground truth annotations: {args.benchmark_gt}")
        gt_dets = {}
        with open(args.benchmark_gt, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    f_idx = int(parts[0])
                    x, y, w_box, h_box = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    if f_idx not in gt_dets:
                        gt_dets[f_idx] = []
                    gt_dets[f_idx].append({
                        'cx': x + w_box / 2.0,
                        'cy': y + h_box / 2.0,
                        'box': [x, y, x + w_box, y + h_box],
                        'score': 1.0
                    })
        detector = None
    else:
        print("🔍 Initializing live runtime D-FINE object detector...")
        detector = DFineDetector(weights_path=args.weights, device=device, score_thresh=args.conf)

    # Open Video Capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    window_title = f"RL Drone Swarm Tracker — {base_stem}"
    if not args.headless:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(1280, int(w * 1.5)), min(720, int(h * 1.5)))
        print(f"🖥️  GUI Window active: '{window_title}' (Press 'q'/ESC to quit, SPACE to pause/resume)")

    frame_idx = 0
    t_start = time.time()
    active_counts = []

    print(f"\n🚀 Running real-time tracking on {total_frames} frames ({w}x{h} @ {fps:.1f} FPS)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. RUN OBJECT DETECTION AT RUNTIME (D-FINE)
        if use_gt_benchmark:
            detections = gt_dets.get(frame_idx, [])
        else:
            detections = detector.detect(frame)

        # 2. RUN REINFORCEMENT LEARNING DATA ASSOCIATION & TRACKING
        active_tracks = tracker.process_frame(detections, frame_idx)
        active_counts.append(len(active_tracks))

        # 3. RENDER TRAJECTORIES & ANNOTATIONS ON FRAME
        for trk in active_tracks:
            tid = trk['track_id']
            box = trk['box']
            hist = trk['history']
            gate_r = int(trk['gate'])

            # Deterministic neon color per drone ID
            color = (
                int((tid * 67 + 50) % 256),
                int((tid * 137 + 100) % 256),
                int((tid * 223 + 150) % 256)
            )

            # Continuous motion trajectory trail
            if len(hist) > 1:
                for p in range(1, len(hist)):
                    cv2.line(frame, hist[p - 1], hist[p], color, 2, cv2.LINE_AA)

            # Drone centroid marker
            cx, cy = int(trk['cx']), int(trk['cy'])
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # Velocity-Adaptive Dynamic Gate visualization ring
            cv2.circle(frame, (cx, cy), gate_r, color, 1, cv2.LINE_AA)

            # Drone bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID Badge with current dynamic gate radius
            label = f"#{tid} ({gate_r}px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 2, max(th + 2, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Telemetry HUD Overlay
        cv2.rectangle(frame, (10, 10), (370, 100), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (370, 100), (0, 255, 128), 1)
        cv2.putText(frame, f"Active Drones Traced: {len(active_tracks)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} ({fps:.0f} FPS)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detector: D-FINE | Dynamic Gate: 20-65px", (20, 85),
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

    # Save trajectory dataset
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
    print("\n" + "=" * 65)
    print("📈 TRACKING PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"⏱️  Processing Time : {elapsed:.2f} seconds ({proc_fps:.1f} FPS)")
    print(f"🎬 Frames Processed: {frame_idx} / {total_frames}")
    print(f"🛸 Persistent Tracks: {df_clean['track_id'].nunique() if not df_clean.empty else 0}")
    print(f"📍 Trajectory Points: {len(df_clean)}")
    print(f"📊 Mean Swarm Size : {np.mean(active_counts):.1f} drones/frame (Peak: {np.max(active_counts) if active_counts else 0})")
    print(f"🎥 Tracked Video   : {out_video_path} ({os.path.getsize(out_video_path)/1024/1024:.2f} MB)")
    print(f"📄 Trajectory Logs : {out_csv_path} and {out_json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
