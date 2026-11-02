#!/usr/bin/env python3
"""
Drone Swarm Video Compiler & Trajectory Tracking Evaluation Pipeline
=====================================================================
1. Compiles UAVSwarm image sequence into an MP4 video.
2. Runs fine-tuned D-FINE detector (PyTorch deploy mode) on the video.
3. Associates multi-drone candidate detections using the trained Shrimp Drone RL Agent (Q-network).
4. Renders continuous colored trajectory trails and drone ID bounding boxes.
5. Saves the tracked output video and exports trajectory CSV and JSON.
"""

import os
import sys
import math
import time
import glob
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torchvision.transforms as T

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DFINE_DIR = "/content/D-FINE"
if DFINE_DIR not in sys.path and os.path.exists(DFINE_DIR):
    sys.path.insert(0, DFINE_DIR)

from src.core import YAMLConfig
from train_drone_agent import DroneQNetwork

# ── SIMPLE KALMAN FILTER ──────────────────────────────────────────────────────
class SimpleKF:
    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 5.0
        self.R = np.eye(2, dtype=np.float32) * 20.0

    def predict(self):
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


# ── DRONE RL TRAJECTORY TRACKER (VELOCITY-ADAPTIVE DYNAMIC GATING) ──────────────
class DroneRLTracker:
    def __init__(self, policy_path: str, device: torch.device,
                 base_gate: float = 20.0, vel_coef: float = 2.0, missed_coef: float = 4.0,
                 min_gate: float = 20.0, max_gate: float = 65.0, max_missed: int = 8):
        self.device = device
        self.base_gate = base_gate
        self.vel_coef = vel_coef
        self.missed_coef = missed_coef
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.max_missed = max_missed

        self.policy = DroneQNetwork().to(device)
        ckpt = torch.load(policy_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            self.policy.load_state_dict(ckpt['model'])
        else:
            self.policy.load_state_dict(ckpt)
        self.policy.eval()

        self.tracks = {}          # {track_id: {'kf', 'box', 'missed', 'cx', 'cy', 'vx', 'vy', 'history', 'hits'}}
        self.trajectories = {}    # {track_id: [{'frame', 'cx', 'cy', 'vx', 'vy', 'box'}]}
        self.next_id = 1

    def compute_dynamic_gate(self, vx: float, vy: float, missed: int = 0) -> float:
        """
        Velocity-Adaptive Dynamic Gating:
        Dynamically calculates the tracking gate size based on drone velocity:
        - When hovering/slow (speed ~0.5px): gate contracts to ~20px, locking out neighbor swarm drones.
        - When accelerating/dashing (speed ~15px): gate expands to ~50px, preserving the track.
        """
        speed = math.hypot(vx, vy)
        gate = self.base_gate + self.vel_coef * speed + self.missed_coef * missed
        return float(np.clip(gate, self.min_gate, self.max_gate))

    def _get_subregion(self, dx: float, dy: float, gate: float) -> int:
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
                    reg = self._get_subregion(dx, dy, dyn_gate)
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
                    cost_matrix[i, j] = dist * (1.5 - float(q))

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
                if len(data['history']) > 150:
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
                    })

        # Increment missed on unassigned tracks
        for tid in tids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_missed:
                    del self.tracks[tid]

        # Spawn new tracks for unassigned detections
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


# ── STEP 1: COMPILE IMAGES TO VIDEO ──────────────────────────────────────────
def compile_images_to_video(images_dir: str, output_video_path: str, fps: int = 25) -> tuple:
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    first_img = cv2.imread(image_files[0])
    h, w = first_img.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"🎬 Compiling {len(image_files)} images from {images_dir} into video ({w}x{h} @ {fps}fps)...")
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"  ✓ Video compiled successfully: {output_video_path} ({os.path.getsize(output_video_path)/1024/1024:.2f} MB)")
    return output_video_path, len(image_files), (w, h)


# ── STEP 2: LOAD FINE-TUNED D-FINE DETECTOR ──────────────────────────────────
class DFineDetector(nn.Module):
    def __init__(self, config_path: str, weights_path: str, device: torch.device):
        super().__init__()
        self.device = device
        cfg = YAMLConfig(config_path, resume=weights_path)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(weights_path, map_location="cpu")
        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint

        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(device)
        self.postprocessor = cfg.postprocessor.deploy().to(device)
        self.eval()

        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    @torch.no_grad()
    def detect(self, cv2_frame: np.ndarray, score_threshold: float = 0.35) -> list:
        h, w = cv2_frame.shape[:2]
        frame_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb)

        orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(self.device)
        im_data = self.transforms(im_pil).unsqueeze(0).to(self.device)

        outputs = self.model(im_data)
        labels, boxes, scores = self.postprocessor(outputs, orig_size)

        scr = scores[0].cpu().numpy()
        bx = boxes[0].cpu().numpy()

        valid_mask = scr >= score_threshold
        valid_boxes = bx[valid_mask]
        valid_scores = scr[valid_mask]

        detections = []
        for b, s in zip(valid_boxes, valid_scores):
            x1, y1, x2, y2 = b
            detections.append({
                'cx': float((x1 + x2) / 2.0),
                'cy': float((y1 + y2) / 2.0),
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'score': float(s),
            })
        return detections


# ── STEP 3: RUN END-TO-END TRAJECTORY TRACKING ────────────────────────────────
def run_video_tracking(
    input_video: str,
    output_video: str,
    detector: DFineDetector,
    tracker: DroneRLTracker,
    output_csv: str,
    output_json: str,
    score_threshold: float = 0.35
):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    frame_idx = 0
    t_start = time.time()
    total_det_time = 0.0
    total_trk_time = 0.0

    print(f"\n🚀 Running Drone Trajectory Tracking on {input_video} ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Detection
        t0 = time.time()
        detections = detector.detect(frame, score_threshold=score_threshold)
        total_det_time += time.time() - t0

        # 2. Tracking
        t1 = time.time()
        active_tracks = tracker.process_frame(detections, frame_idx)
        total_trk_time += time.time() - t1

        # 3. Draw Trajectories & Bounding Boxes
        for trk in active_tracks:
            tid = trk['track_id']
            box = trk['box']
            hist = trk['history']

            # Unique deterministic color per track ID
            color = (
                int((tid * 67) % 256),
                int((tid * 137) % 256),
                int((tid * 223) % 256)
            )

            # Draw continuous trajectory motion trail
            if len(hist) > 1:
                for p in range(1, len(hist)):
                    cv2.line(frame, hist[p - 1], hist[p], color, 2, cv2.LINE_AA)

            # Draw drone box + label
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Drone #{tid}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 4. HUD Telemetry Overlay
        cv2.rectangle(frame, (10, 10), (330, 105), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (330, 105), (0, 255, 128), 1)
        cv2.putText(frame, f"Active Drones: {len(active_tracks)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        fps_curr = frame_idx / (time.time() - t_start)
        cv2.putText(frame, f"Trajectory Tracing FPS: {fps_curr:.1f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  Frame {frame_idx}/{total_frames} | Active: {len(active_tracks)} drones | Total tracks: {len(tracker.trajectories)} | FPS: {fps_curr:.1f}")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start

    print(f"\n✅ Video Tracking Complete in {elapsed:.2f}s ({frame_idx/elapsed:.1f} FPS)!")
    print(f"  Output Video: {output_video} ({os.path.getsize(output_video)/1024/1024:.2f} MB)")

    # 5. Export Trajectory Data
    traj_rows = []
    for tid, pts in tracker.trajectories.items():
        for pt in pts:
            traj_rows.append({
                'track_id': tid,
                'frame': pt['frame'],
                'cx': pt['cx'],
                'cy': pt['cy'],
                'vx': pt['vx'],
                'vy': pt['vy'],
                'x1': pt['box'][0],
                'y1': pt['box'][1],
                'x2': pt['box'][2],
                'y2': pt['box'][3],
            })

    if traj_rows:
        df_traj = pd.DataFrame(traj_rows)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_traj.to_csv(output_csv, index=False)
        with open(output_json, 'w') as f:
            json.dump(tracker.trajectories, f, indent=2)

        print(f"📊 Trajectories Exported:")
        print(f"  ✓ {output_csv} ({len(df_traj)} trajectory points, {len(tracker.trajectories)} distinct drones)")
        print(f"  ✓ {output_json}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compile UAVSwarm images to video and trace drone trajectories.")
    parser.add_argument("--images_dir", default="/content/uavswarm/test/UAVSwarm-02/img1", help="Path to sequence img1/ directory")
    parser.add_argument("--dfine_config", default="/content/D-FINE/configs/dfine/custom/dfine_hgnetv2_m_uavswarm.yml", help="D-FINE YAML config")
    parser.add_argument("--dfine_weights", default="/content/dfine_drone_weights.pth", help="Fine-tuned D-FINE weights (.pth)")
    parser.add_argument("--policy_weights", default="/content/drone_tracker_policy_best.pth", help="Trained DroneAgent policy (.pth)")
    parser.add_argument("--output_dir", default="/content/results", help="Directory for output video and trajectory files")
    parser.add_argument("--score_thresh", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--gate_radius", type=float, default=250.0, help="RL tracking gating radius")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(args.output_dir, exist_ok=True)
    raw_video = os.path.join(args.output_dir, "uavswarm_test_raw.mp4")
    tracked_video = os.path.join(args.output_dir, "tracked_drone_trajectories.mp4")
    out_csv = os.path.join(args.output_dir, "drone_trajectories.csv")
    out_json = os.path.join(args.output_dir, "drone_trajectories.json")

    # Step 1: Images -> Video
    compile_images_to_video(args.images_dir, raw_video, fps=25)

    # Step 2: Initialize Detector and RL Tracker
    print(f"Loading Fine-Tuned D-FINE detector from {args.dfine_weights}...")
    detector = DFineDetector(args.dfine_config, args.dfine_weights, device)

    print(f"Loading Trained Drone RL Tracker Policy from {args.policy_weights}...")
    tracker = DroneRLTracker(args.policy_weights, device, gate_radius=args.gate_radius)

    # Step 3: Run Trajectory Tracing
    run_video_tracking(
        input_video=raw_video,
        output_video=tracked_video,
        detector=detector,
        tracker=tracker,
        output_csv=out_csv,
        output_json=out_json,
        score_threshold=args.score_thresh
    )

if __name__ == "__main__":
    main()
