"""
Drone Swarm Tracker — Full Inference Pipeline
=============================================
Adapted from: Shrimp/run.py
Changes for drone case:
  - Larger gate radius (250px vs 150px)
  - Multi-drone counting replaced with swarm density / active count display
  - Removed VectorHysteresisCounter (shrimp-specific count line)
  - Added swarm size histogram overlay
  - Higher score threshold (0.35 vs 0.25) — drones at distance are harder
  - Runs D-FINE ONNX model instead of the shrimp last.onnx
"""

import cv2
import math
import time
import numpy as np
import torch
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
from train_drone_agent import DroneQNetwork

# ── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH   = '../results/dfine_uavswarm.onnx'   # D-FINE exported ONNX
POLICY_PATH  = '../results/drone_tracker_policy_best.pth'
VIDEO_PATH   = 'input_video.mp4'

INPUT_W, INPUT_H = 800, 800      # Must match D-FINE training resolution
SCORE_THRES  = 0.35              # Higher than shrimp (0.25) — fewer false positives
NMS_IOU      = 0.30
BASE_GATE    = 20.0              # Velocity-adaptive base gate
VEL_COEF     = 2.0               # Scale gate with drone velocity
MISSED_COEF  = 4.0               # Expand gate during occlusions/misses
MIN_GATE     = 20.0              # Contracted gate for hovering drones (seals out neighbors)
MAX_GATE     = 65.0              # Max gate cap (strictly below 70-90px swarm spacing)
MAX_MISSED   = 8                 # Frames before track is dropped


# ── SIMPLE KALMAN FILTER (same as Shrimp run.py) ─────────────────────────────

class SimpleKF:
    """2D constant-velocity Kalman filter for drone centroid tracking."""

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


# ── NMS ──────────────────────────────────────────────────────────────────────

def nms_indices(boxes, scores, score_thr, iou_thr):
    """Greedy NMS. Same implementation as Shrimp run.py for consistency."""
    valid = sorted([i for i, s in enumerate(scores) if s >= score_thr],
                   key=lambda i: scores[i], reverse=True)
    keep = []
    removed = [False] * len(valid)

    for i, idx_a in enumerate(valid):
        if removed[i]:
            continue
        keep.append(idx_a)
        box_a = boxes[idx_a]
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        for j in range(i + 1, len(valid)):
            if removed[j]:
                continue
            idx_b = valid[j]
            box_b = boxes[idx_b]
            xx1, yy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            xx2, yy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union = area_a + area_b - w * h + 1e-6
            if (w * h / union) > iou_thr:
                removed[j] = True
    return keep


# ── RL TRACKER ───────────────────────────────────────────────────────────────

class DroneRLTracker:
    """
    Multi-drone RL tracker. Runs one DroneQNetwork instance (shared weights)
    across all active tracks — the network is general, not per-track.

    Data association per track:
      1. Kalman-predict the track's next position
      2. Gather all detections within GATE_RADIUS
      3. Run QNetwork on each candidate → pick highest Q-value
      4. Update Kalman with matched detection
      5. Unmatched tracks → increment missed counter → drop after MAX_MISSED
      6. Unmatched detections → spawn new tracks

    This is architecturally identical to Shrimp run.py's RLTracker,
    just with drone-appropriate hyperparameters.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.policy = DroneQNetwork().to(device)
        self.policy.load_state_dict(
            torch.load(POLICY_PATH, map_location=device))
        self.policy.eval()

        self.tracks = {}    # {track_id: {'kf', 'box', 'missed', 'cx', 'cy', 'vx', 'vy', 'history', 'hits'}}
        self.trajectories = {}  # Complete lifetime trajectory history per track_id: [{'frame', 'cx', 'cy', 'vx', 'vy', 'box'}]
        self.next_id = 1

    def compute_dynamic_gate(self, vx: float, vy: float, missed: int = 0) -> float:
        """
        Velocity-Adaptive Dynamic Gating:
        Dynamically calculates the tracking gate size based on drone velocity:
        - When hovering/slow (speed ~0.5px): gate contracts to ~20px, locking out neighbor swarm drones.
        - When accelerating/dashing (speed ~15px): gate expands to ~50px, preserving the track.
        """
        speed = math.hypot(vx, vy)
        gate = BASE_GATE + VEL_COEF * speed + MISSED_COEF * missed
        return float(np.clip(gate, MIN_GATE, MAX_GATE))

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

    def process_frame(self, detections: list, frame_idx: int = 0) -> list:
        """
        Args:
            detections: list of {'cx', 'cy', 'box': [x1,y1,x2,y2]}
            frame_idx:  current frame sequence number
        Returns:
            list of dicts: {'track_id', 'box', 'cx', 'cy', 'history', 'gate'}
        """
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
                if self.tracks[tid]['missed'] > MAX_MISSED:
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


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Load D-FINE ONNX
    session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])
    inp_names = [i.name for i in session.get_inputs()]
    out_names = [o.name for o in session.get_outputs()]

    tracker = DroneRLTracker(device)
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_idx = 0
    total_tracker_ms = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        H, W = frame.shape[:2]

        # ── Detection ─────────────────────────────────────────────────────
        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, :]
        orig_size = np.array([[float(W), float(H)]], dtype=np.float32)

        outputs = session.run(out_names, {inp_names[0]: img, inp_names[1]: orig_size})
        boxes  = np.squeeze(outputs[1])
        scores = np.squeeze(outputs[2])

        detections = []
        keep = nms_indices(
            boxes if boxes.ndim > 1 else boxes[np.newaxis],
            scores if scores.ndim > 0 else np.array([scores]),
            SCORE_THRES, NMS_IOU
        )
        for k in keep:
            x1, y1, x2, y2 = boxes[k] if boxes.ndim > 1 else boxes
            detections.append({'cx': (x1+x2)/2, 'cy': (y1+y2)/2,
                               'box': [float(x1), float(y1), float(x2), float(y2)]})

        # ── Tracking ──────────────────────────────────────────────────────
        t0 = time.time()
        tracks = tracker.process_frame(detections, frame_idx)
        total_tracker_ms += (time.time() - t0) * 1000

        # ── Visualization & Trajectory Tracing ─────────────────────────────
        for trk in tracks:
            tid = trk['track_id']
            box = trk['box']
            hist = trk['history']
            color = ((tid * 67) % 255, (tid * 113) % 255, (tid * 197) % 255)

            # Draw trajectory motion trail
            if len(hist) > 1:
                for p in range(1, len(hist)):
                    cv2.line(frame, hist[p-1], hist[p], color, 2)

            # Bounding box + Drone Track ID
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Drone #{tid}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD Overlay
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        avg_fps = 1000.0 / (total_tracker_ms / max(frame_idx, 1))
        cv2.putText(frame, f"Tracker FPS: {avg_fps:.1f}", (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

        cv2.imshow('Drone RL Trajectory Tracker', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Trajectory Export ──────────────────────────────────────────────
    import pandas as pd, json, os
    os.makedirs('../results', exist_ok=True)
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
        df_traj.to_csv('../results/drone_trajectories.csv', index=False)
        with open('../results/drone_trajectories.json', 'w') as f:
            json.dump(tracker.trajectories, f, indent=2)
        print(f"✅ Traced {len(tracker.trajectories)} drone trajectories ({len(df_traj)} total points).")
        print("   Saved: ../results/drone_trajectories.csv")
        print("   Saved: ../results/drone_trajectories.json")

    print(f"\nDone. Avg tracker FPS: {1000.0 / (total_tracker_ms / max(frame_idx, 1)):.1f}")


if __name__ == '__main__':
    main()
