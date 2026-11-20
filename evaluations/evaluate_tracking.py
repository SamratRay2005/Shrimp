#!/usr/bin/env python3
"""
Full Tracking Evaluation Suite (MOTA, HOTA, IDF1, Purity & Ablations)
=====================================================================
Evaluates DroneRLTracker (from run2.py) and 3 controlled baselines across all 36
official UAVSwarm test sequences using strictly isolated ground-truth per-frame
detections (track IDs stripped, detections deterministically shuffled per frame).

Outputs saved to evaluations/results/:
  - results/tier_a/per_sequence_metrics.csv          (36 sequences for DroneRLTracker)
  - results/tier_a/per_sequence_all_methods.csv      (36 sequences x 4 trackers)
  - results/tier_a/ablation_summary.csv              (36-seq & 4-seq aggregate metrics)
  - results/tier_a/subset_06_28_30_46_summary.csv
  - results/tier_a/subset_06_28_30_46_per_sequence.csv
  - results/tier_a/detailed_results.json             (complete structured telemetry + metadata)
  - results/predictions/mot_oracle/<seq>.txt         (MOT16 predictions for DroneRLTracker)
  - results/plots/ablation_comparison.png
  - results/plots/per_sequence_scores.png
  - results/plots/failure_and_scaling_analysis.png
"""

import os
import sys
import csv
import json
import math
import time
import random
import configparser
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run2 import (
    DroneRLTracker,
    DroneQNetwork,
    SimpleKF,
    BASE_GATE,
    VEL_COEF,
    MISSED_COEF,
    MIN_GATE,
    MAX_GATE,
    MAX_MISSED,
    TRAIL_LENGTH,
)
from mot_metrics import (
    compute_iou_matrix,
    evaluate_sequence,
    aggregate_sequence_results,
)


EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
TIER_A_DIR = RESULTS_DIR / "tier_a"
GT_ROOT = DATA_DIR / "uavswarm_test_gt"
PRED_MOT_DIR = RESULTS_DIR / "predictions" / "mot_oracle"
PLOTS_DIR = RESULTS_DIR / "plots"
POLICY_PATH = REPO_ROOT / "checkpoints" / "drone_tracker_policy_best.pth"
LOCAL_SUBSET = {"UAVSwarm-02", "UAVSwarm-04", "UAVSwarm-08", "UAVSwarm-14"}
TARGET_SUBSET_06_28_30_46 = {"UAVSwarm-06", "UAVSwarm-28", "UAVSwarm-30", "UAVSwarm-46"}


# ===================== CONTROLLED ABLATION TRACKERS =====================

class DynGateHungarianNoRLTracker(DroneRLTracker):
    """
    Controlled ablation tracker matching DroneRLTracker (SimpleKF + Velocity-Adaptive
    Dynamic Gate + Hungarian + step_d guard + EMA velocity), using pure Euclidean
    distance `cost_matrix[i, j] = dist` without RL Q-score weighting.
    """

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
            pred_cx, pred_cy = data['kf'].predict()
            vx, vy = data['vx'], data['vy']
            dyn_gate = self.compute_dynamic_gate(vx, vy, data['missed'])
            gates.append(dyn_gate)

            for j, det in enumerate(detections):
                dx = det['cx'] - pred_cx
                dy = det['cy'] - pred_cy
                dist = math.hypot(dx, dy)
                if dist <= dyn_gate:
                    cost_matrix[i, j] = dist

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

                self.trajectories.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': float(vx), 'vy': float(vy),
                    'box': [float(b) for b in det['box']],
                })
                matched_tracks.add(tid)
                matched_dets.add(c)
                if data['hits'] >= 2:
                    active.append({'track_id': tid})

        for tid in tids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > MAX_MISSED:
                    del self.tracks[tid]

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
                self.trajectories.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                })
                self.next_id += 1

        return active


class Static250GreedyRLTracker(DroneRLTracker):
    """
    Ablation 2: Pre-Phase-6 Shrimp-style baseline (SimpleKF + fixed R=250px gate
    + sequential greedy argmax Q per track without Hungarian optimization).
    """
    STATIC_GATE = 250.0

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
                }
                self.trajectories[tid] = [{
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                }]
                self.next_id += 1
            return []

        matched_tracks = set()
        matched_dets = set()

        for tid in tids:
            data = self.tracks[tid]
            pred_cx, pred_cy = data['kf'].predict()
            vx = data['vx']
            vy = data['vy']

            cands = []
            for j, det in enumerate(detections):
                if j in matched_dets:
                    continue
                dx = det['cx'] - pred_cx
                dy = det['cy'] - pred_cy
                dist = math.hypot(dx, dy)
                if dist <= self.STATIC_GATE:
                    reg = self.get_subregion(dx, dy, self.STATIC_GATE)
                    cands.append((j, dx, dy, reg))

            if cands:
                feats = torch.FloatTensor([[
                    dx / self.STATIC_GATE,
                    dy / self.STATIC_GATE,
                    (reg if reg >= 0 else 0) / 7.0,
                    vx / self.STATIC_GATE,
                    vy / self.STATIC_GATE,
                ] for _, dx, dy, reg in cands]).to(self.device)
                with torch.no_grad():
                    q_vals = self.policy(feats).cpu().numpy().flatten()

                best_pos = int(np.argmax(q_vals))
                best_j = cands[best_pos][0]
                det = detections[best_j]

                step_vx = det['cx'] - data['cx']
                step_vy = det['cy'] - data['cy']
                data['vx'] = 0.7 * data['vx'] + 0.3 * step_vx
                data['vy'] = 0.7 * data['vy'] + 0.3 * step_vy
                data['cx'] = det['cx']
                data['cy'] = det['cy']
                data['box'] = det['box']
                data['kf'].update(det['cx'], det['cy'])
                data['hits'] += 1
                data['missed'] = 0

                self.trajectories.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': float(step_vx), 'vy': float(step_vy),
                    'box': [float(b) for b in det['box']],
                })
                matched_tracks.add(tid)
                matched_dets.add(best_j)
                if data['hits'] >= 2:
                    active.append({'track_id': tid})

        for tid in tids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > MAX_MISSED:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.tracks[tid] = {
                    'kf': SimpleKF(det['cx'], det['cy']),
                    'box': det['box'],
                    'cx': det['cx'], 'cy': det['cy'],
                    'vx': 0.0, 'vy': 0.0,
                    'hits': 1, 'missed': 0,
                }
                self.trajectories.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                })
                self.next_id += 1

        return active


class KalmanIoUHungarianTracker:
    """
    Ablation 3: Classical SORT-style IoU tracker (SimpleKF shifts last box center
    to predicted centroid, matches via Hungarian on 1 - IoU with IoU >= 0.30).
    """
    IOU_THRESH = 0.30

    def __init__(self):
        self.tracks = {}
        self.trajectories = {}
        self.next_id = 1

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
                    'hits': 1, 'missed': 0,
                }
                self.trajectories[tid] = [{
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                }]
                self.next_id += 1
            return []

        pred_boxes = []
        for tid in tids:
            data = self.tracks[tid]
            pred_cx, pred_cy = data['kf'].predict()
            x1, y1, x2, y2 = data['box']
            bw = x2 - x1
            bh = y2 - y1
            pred_boxes.append([
                pred_cx - bw / 2.0,
                pred_cy - bh / 2.0,
                pred_cx + bw / 2.0,
                pred_cy + bh / 2.0,
            ])

        det_boxes = np.array([d['box'] for d in detections], dtype=np.float64) if detections else np.zeros((0, 4))
        pred_arr = np.array(pred_boxes, dtype=np.float64)
        iou_mat = compute_iou_matrix(pred_arr, det_boxes)
        cost_matrix = np.where(iou_mat >= self.IOU_THRESH, 1.0 - iou_mat, 1e5)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat.size > 0 and iou_mat[r, c] >= self.IOU_THRESH:
                tid = tids[r]
                det = detections[c]
                data = self.tracks[tid]
                vx = det['cx'] - data['cx']
                vy = det['cy'] - data['cy']
                data['cx'] = det['cx']
                data['cy'] = det['cy']
                data['box'] = det['box']
                data['kf'].update(det['cx'], det['cy'])
                data['hits'] += 1
                data['missed'] = 0

                self.trajectories.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': float(vx), 'vy': float(vy),
                    'box': [float(b) for b in det['box']],
                })
                matched_tracks.add(tid)
                matched_dets.add(c)
                if data['hits'] >= 2:
                    active.append({'track_id': tid})

        for tid in tids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > MAX_MISSED:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.tracks[tid] = {
                    'kf': SimpleKF(det['cx'], det['cy']),
                    'box': det['box'],
                    'cx': det['cx'], 'cy': det['cy'],
                    'hits': 1, 'missed': 0,
                }
                self.trajectories.setdefault(tid, []).append({
                    'frame': frame_idx,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'box': [float(b) for b in det['box']],
                })
                self.next_id += 1

        return active


# ===================== DATA LOADING & EXECUTION =====================

def load_sequence(seq_dir: Path) -> dict:
    """Load seqinfo.ini and MOT16 gt.txt for one test sequence."""
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    sec = cfg["Sequence"]
    seq_len = int(sec["seqLength"])
    width = int(sec["imWidth"])
    height = int(sec["imHeight"])
    fps = float(sec.get("frameRate", 25))

    gt_frames: Dict[int, List[dict]] = {}
    gt_tracks = {}
    box_diags = []

    with open(seq_dir / "gt" / "gt.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            f_idx = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            if len(parts) >= 7:
                try:
                    if float(parts[6]) == 0:
                        continue
                except ValueError:
                    pass
            x2 = x1 + w
            y2 = y1 + h
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            gt_frames.setdefault(f_idx, []).append({
                'id': gt_id,
                'cx': cx,
                'cy': cy,
                'box': [x1, y1, x2, y2],
            })
            gt_tracks.setdefault(gt_id, []).append((f_idx, cx, cy))
            box_diags.append(math.hypot(w, h))

    # Compute ground-truth kinematic statistics for failure/scaling analysis
    speeds = []
    for tid, pts in gt_tracks.items():
        pts.sort(key=lambda x: x[0])
        for k in range(1, len(pts)):
            dt = pts[k][0] - pts[k - 1][0]
            if dt > 0:
                speeds.append(math.hypot(pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2]) / dt)

    nn_dists = []
    frame_counts = []
    for f_idx in range(1, seq_len + 1):
        objs = gt_frames.get(f_idx, [])
        frame_counts.append(len(objs))
        if len(objs) >= 2:
            pts = np.array([[o['cx'], o['cy']] for o in objs])
            diff = pts[:, None, :] - pts[None, :, :]
            dmat = np.hypot(diff[:, :, 0], diff[:, :, 1])
            np.fill_diagonal(dmat, np.inf)
            nn_dists.extend(np.min(dmat, axis=1).tolist())

    max_frame = max([seq_len] + list(gt_frames.keys()))
    return {
        'name': seq_dir.name,
        'seq_len': max_frame,
        'width': width,
        'height': height,
        'fps': fps,
        'gt_frames': gt_frames,
        'unique_gt_ids': len(gt_tracks),
        'total_gt_boxes': sum(len(v) for v in gt_frames.values()),
        'mean_drones_per_frame': float(np.mean(frame_counts)) if frame_counts else 0.0,
        'peak_drones_per_frame': int(np.max(frame_counts)) if frame_counts else 0,
        'mean_gt_speed_px': float(np.mean(speeds)) if speeds else 0.0,
        'p95_gt_speed_px': float(np.percentile(speeds, 95)) if speeds else 0.0,
        'max_gt_speed_px': float(np.max(speeds)) if speeds else 0.0,
        'median_nn_dist_px': float(np.median(nn_dists)) if nn_dists else 0.0,
        'mean_box_diag_px': float(np.mean(box_diags)) if box_diags else 0.0,
    }


def trajectories_to_pred_frames(trajectories: dict, min_len: int = 2) -> Dict[int, List[dict]]:
    """
    Convert tracker.trajectories into frame-indexed predictions matching run2.py's
    exact export rule (`if len(pts) >= min_len`).
    """
    pred_frames: Dict[int, List[dict]] = {}
    for tid, pts in trajectories.items():
        if len(pts) < min_len:
            continue
        for pt in pts:
            pred_frames.setdefault(int(pt['frame']), []).append({
                'id': int(tid),
                'box': list(pt['box']),
            })
    return pred_frames


def write_mot_predictions(seq_name: str, trajectories: dict, min_len: int = 2):
    """Write standard MOT16 prediction file to evaluations/results/predictions/mot_oracle/."""
    rows = []
    for tid, pts in trajectories.items():
        if len(pts) < min_len:
            continue
        for pt in pts:
            x1, y1, x2, y2 = pt['box']
            rows.append((int(pt['frame']), int(tid), x1, y1, x2 - x1, y2 - y1))
    rows.sort(key=lambda r: (r[0], r[1]))
    out_path = PRED_MOT_DIR / f"{seq_name}.txt"
    with open(out_path, "w") as f:
        for fr, tid, x1, y1, w, h in rows:
            f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def run_tracker_on_sequence(tracker, seq_data: dict, seq_idx: int):
    """
    Run tracker with ZERO GT ID leakage: passes only {'cx','cy','box','score'}
    and shuffles detections deterministically per (seq_idx, frame_idx).
    """
    t0 = time.perf_counter()
    for f_idx in range(1, seq_data['seq_len'] + 1):
        raw_gt = seq_data['gt_frames'].get(f_idx, [])
        detections = [{
            'cx': float(d['cx']),
            'cy': float(d['cy']),
            'box': list(d['box']),
            'score': 1.0,
        } for d in raw_gt]
        rng = random.Random(42 + seq_idx * 10000 + f_idx)
        rng.shuffle(detections)
        tracker.process_frame(detections, f_idx)
    elapsed = time.perf_counter() - t0
    return tracker.trajectories, elapsed


def flatten_metrics_row(method: str, seq_data: dict, eval_res: dict, elapsed: float, trajectories: dict) -> dict:
    c = eval_res['clear']
    idf = eval_res['idf1']
    h = eval_res['hota']
    k = eval_res['kinematics']
    n_tracks_exported = sum(1 for pts in trajectories.values() if len(pts) >= 2)
    return {
        'method': method,
        'sequence': seq_data['name'],
        'is_local_4seq': seq_data['name'] in LOCAL_SUBSET,
        'width': seq_data['width'],
        'height': seq_data['height'],
        'frames': seq_data['seq_len'],
        'unique_gt_ids': seq_data['unique_gt_ids'],
        'total_gt': c['total_gt'],
        'total_pred': c['total_pred'],
        'pred_tracks': n_tracks_exported,
        'mean_drones_per_frame': round(seq_data['mean_drones_per_frame'], 2),
        'peak_drones': seq_data['peak_drones_per_frame'],
        'mean_gt_speed_px': round(seq_data['mean_gt_speed_px'], 3),
        'p95_gt_speed_px': round(seq_data['p95_gt_speed_px'], 3),
        'max_gt_speed_px': round(seq_data['max_gt_speed_px'], 3),
        'median_nn_dist_px': round(seq_data['median_nn_dist_px'], 2),
        'MOTA': round(c['mota'] * 100.0, 2),
        'HOTA': round(h['hota'] * 100.0, 2),
        'DetA': round(h['deta'] * 100.0, 2),
        'AssA': round(h['assa'] * 100.0, 2),
        'LocA': round(h['loca'] * 100.0, 2),
        'AssRe': round(h['assre'] * 100.0, 2),
        'AssPr': round(h['asspr'] * 100.0, 2),
        'IDF1': round(idf['idf1'] * 100.0, 2),
        'IDP': round(idf['idp'] * 100.0, 2),
        'IDR': round(idf['idr'] * 100.0, 2),
        'MOTP': round(c['motp'] * 100.0, 2),
        'Precision': round(c['precision'] * 100.0, 2),
        'Recall': round(c['recall'] * 100.0, 2),
        'TP': c['tp'],
        'FP': c['fp'],
        'FN': c['fn'],
        'IDSW': c['idsw'],
        'Frag': c['frag'],
        'MT': c['mt'],
        'PT': c['pt'],
        'ML': c['ml'],
        'Avg_Purity': round(k['avg_track_purity'] * 100.0, 2),
        'Pure_100_Pct': round(k['pure_100_ratio'] * 100.0, 2),
        'Pure_100_Count': k['pure_100_count'],
        'Matched_Tracks': k['matched_track_count'],
        'Erratic_Jumps_gt25': k['erratic_jumps_gt25'],
        'Mean_Step_Disp': round(k['mean_step_disp'], 2),
        'Median_Step_Disp': round(k['median_step_disp'], 2),
        'Runtime_Sec': round(elapsed, 4),
        'FPS': round(seq_data['seq_len'] / elapsed, 1) if elapsed > 0 else 0.0,
    }


def summarize_aggregate(method: str, split_name: str, seq_results: List[dict], total_frames: int, total_time: float) -> dict:
    agg = aggregate_sequence_results(seq_results)
    c = agg['clear']
    idf = agg['idf1']
    h = agg['hota']
    k = agg['kinematics']
    return {
        'method': method,
        'split': split_name,
        'sequences': len(seq_results),
        'frames': total_frames,
        'unique_gt_ids': c['num_gt_ids'],
        'total_gt': c['total_gt'],
        'total_pred': c['total_pred'],
        'MOTA': round(c['mota'] * 100.0, 2),
        'HOTA': round(h['hota'] * 100.0, 2),
        'DetA': round(h['deta'] * 100.0, 2),
        'AssA': round(h['assa'] * 100.0, 2),
        'LocA': round(h['loca'] * 100.0, 2),
        'AssRe': round(h['assre'] * 100.0, 2),
        'AssPr': round(h['asspr'] * 100.0, 2),
        'IDF1': round(idf['idf1'] * 100.0, 2),
        'IDP': round(idf['idp'] * 100.0, 2),
        'IDR': round(idf['idr'] * 100.0, 2),
        'MOTP': round(c['motp'] * 100.0, 2),
        'Precision': round(c['precision'] * 100.0, 2),
        'Recall': round(c['recall'] * 100.0, 2),
        'TP': c['tp'],
        'FP': c['fp'],
        'FN': c['fn'],
        'IDSW': c['idsw'],
        'Frag': c['frag'],
        'MT': c['mt'],
        'PT': c['pt'],
        'ML': c['ml'],
        'Avg_Purity': round(k['avg_track_purity'] * 100.0, 2),
        'Pure_100_Pct': round(k['pure_100_ratio'] * 100.0, 2),
        'Pure_100_Count': k['pure_100_count'],
        'Matched_Tracks': k['matched_track_count'],
        'Erratic_Jumps_gt25': k['erratic_jumps_gt25'],
        'Mean_Step_Disp': round(k['mean_step_disp'], 2),
        'Median_Step_Disp': round(k['median_step_disp'], 2),
        'Runtime_Sec': round(total_time, 3),
        'FPS': round(total_frames / total_time, 1) if total_time > 0 else 0.0,
    }


def generate_plots(summary_df: pd.DataFrame, all_rows_df: pd.DataFrame):
    """Generate 3 publication-quality charts in evaluations/plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # 1. Ablation Comparison Chart (36-seq Full Test Set)
    full_df = summary_df[summary_df['split'] == 'All_36_Test_Sequences'].copy()
    methods = [
        'DroneRLTracker (Ours)',
        'DynGate_Hungarian_NoRL',
        'Kalman_IoU_Hungarian',
        'Static250_GreedyRL',
    ]
    short_labels = [
        'DroneRLTracker\n(Ours)',
        'DynGate+Hungarian\n(No RL)',
        'Kalman+IoU\n(SORT-style)',
        'Static250+Greedy\n(Shrimp Base)',
    ]
    full_df = full_df.set_index('method').loc[methods].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={'width_ratios': [2.2, 1.0]})
    metrics = ['MOTA', 'HOTA', 'IDF1', 'DetA', 'AssA']
    colors = ['#2563eb', '#059669', '#d97706', '#7c3aed', '#0891b2']
    x = np.arange(len(methods))
    width = 0.15

    for idx, (m, col) in enumerate(zip(metrics, colors)):
        vals = full_df[m].values
        bars = ax1.bar(x + (idx - 2) * width, vals, width, label=m, color=col, alpha=0.92)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_ylim(0, 113)
    ax1.set_ylabel('Score (%)')
    ax1.set_title('36-Sequence UAVSwarm Tracking Ablation (GT Detections)')
    ax1.legend(loc='upper right', ncol=5, frameon=True, fontsize=8.5)

    idsw_vals = full_df['IDSW'].values
    bar_colors = ['#16a34a', '#2563eb', '#64748b', '#dc2626']
    bars2 = ax2.bar(x, idsw_vals, width=0.55, color=bar_colors, alpha=0.92)
    for b, v in zip(bars2, idsw_vals):
        ax2.text(b.get_x() + b.get_width() / 2, max(v, 1) * 1.12, f"{int(v):,}",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Ours', 'No RL', 'SORT IoU', 'Greedy250'], fontsize=8.5)
    ax2.set_yscale('log')
    ax2.set_ylim(1, max(idsw_vals) * 3.5)
    ax2.set_ylabel('Identity Switches (IDSW, log scale)')
    ax2.set_title('Identity Switches Across 37,947 GT Boxes')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 2. Per-Sequence Scores for DroneRLTracker (All 36 Sequences sorted by HOTA)
    ours_df = all_rows_df[all_rows_df['method'] == 'DroneRLTracker (Ours)'].sort_values('HOTA', ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 11.0))
    y = np.arange(len(ours_df))
    h_bar = 0.27

    ax.barh(y - h_bar, ours_df['MOTA'], h_bar, label='MOTA (%)', color='#2563eb', alpha=0.9)
    ax.barh(y, ours_df['HOTA'], h_bar, label='HOTA (%)', color='#059669', alpha=0.9)
    ax.barh(y + h_bar, ours_df['IDF1'], h_bar, label='IDF1 (%)', color='#d97706', alpha=0.9)

    y_labels = [
        f"{row['sequence']} ({row['width']}x{row['height']}){' ★' if row['is_local_4seq'] else ''}"
        for _, row in ours_df.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    for tick, is_loc in zip(ax.get_yticklabels(), ours_df['is_local_4seq']):
        if is_loc:
            tick.set_fontweight('bold')
            tick.set_color('#1d4ed8')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Metric Score (%)')
    ax.set_title('DroneRLTracker (Ours) Per-Sequence Performance — 36 Test Sequences (★ = Local 4 Videos)', pad=28)
    ax.legend(bbox_to_anchor=(0.5, 1.005), loc='lower center', ncol=3, frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'per_sequence_scores.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 3. Failure & Scaling Analysis (Resolution, Speed, Neighbor Distance vs HOTA/IDF1/IDSW)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # Panel A: Image Width vs HOTA
    sc1 = axes[0].scatter(ours_df['width'], ours_df['HOTA'],
                          c=ours_df['mean_gt_speed_px'], cmap='viridis',
                          s=65, edgecolors='black', linewidth=0.5, alpha=0.9)
    axes[0].axvline(812, color='#dc2626', linestyle='--', linewidth=1.2, label='Calibration width (812px)')
    axes[0].set_xlabel('Image Width (px)')
    axes[0].set_ylabel('HOTA (%)')
    axes[0].set_title('HOTA vs. Video Width (Color = Speed)')
    axes[0].legend(loc='lower right', fontsize=8)
    cbar = plt.colorbar(sc1, ax=axes[0])
    cbar.set_label('Mean Drone Speed (px/f)', fontsize=8)

    # Panel B: 95th-Percentile Drone Speed vs IDF1
    sc2 = axes[1].scatter(ours_df['p95_gt_speed_px'], ours_df['IDF1'],
                          c=ours_df['width'], cmap='plasma',
                          s=65, edgecolors='black', linewidth=0.5, alpha=0.9)
    axes[1].set_xlabel('95th-Percentile Drone Speed (px/frame)')
    axes[1].set_ylabel('IDF1 (%)')
    axes[1].set_title('IDF1 vs. High-Speed Motion (Color = Width)')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Image Width (px)', fontsize=8)

    # Panel C: Median Nearest-Neighbor Distance vs IDSW
    sc3 = axes[2].scatter(ours_df['median_nn_dist_px'], ours_df['IDSW'],
                          c=ours_df['mean_drones_per_frame'], cmap='coolwarm',
                          s=65, edgecolors='black', linewidth=0.5, alpha=0.9)
    axes[2].axvline(65.0, color='#dc2626', linestyle='--', linewidth=1.2, label='MAX_GATE = 65 px')
    axes[2].set_xlabel('Median Nearest-Neighbor Spacing (px)')
    axes[2].set_ylabel('Identity Switches (IDSW)')
    axes[2].set_title('Remaining IDSW vs. Neighbor Spacing')
    axes[2].legend(loc='upper right', fontsize=8)
    cbar3 = plt.colorbar(sc3, ax=axes[2])
    cbar3.set_label('Mean Swarm Size', fontsize=8)

    for ax in axes:
        for _, row in ours_df.iterrows():
            short = row['sequence'].replace('UAVSwarm-', 'S')
            if ax is axes[0] and (row['HOTA'] < 95.0 or row['sequence'] == 'UAVSwarm-40'):
                ax.annotate(short, (row['width'], row['HOTA']), fontsize=7, xytext=(4, -4), textcoords='offset points')
            elif ax is axes[1] and (row['IDF1'] < 95.0 or row['p95_gt_speed_px'] > 20):
                ax.annotate(short, (row['p95_gt_speed_px'], row['IDF1']), fontsize=7, xytext=(4, -4), textcoords='offset points')
            elif ax is axes[2] and row['IDSW'] >= 2:
                ax.annotate(short, (row['median_nn_dist_px'], row['IDSW']), fontsize=7, xytext=(4, 2), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'failure_and_scaling_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    PRED_MOT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TIER_A_DIR.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in GT_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(seq_dirs)} official UAVSwarm test sequences in {GT_ROOT}")

    device = torch.device("cpu")
    print(f"Running evaluation on device={device} using policy {POLICY_PATH}")

    sequences = [load_sequence(d) for d in seq_dirs]

    def build_tracker(name: str):
        if name == 'DroneRLTracker (Ours)':
            return DroneRLTracker(str(POLICY_PATH), device)
        elif name == 'DynGate_Hungarian_NoRL':
            return DynGateHungarianNoRLTracker(str(POLICY_PATH), device)
        elif name == 'Static250_GreedyRL':
            return Static250GreedyRLTracker(str(POLICY_PATH), device)
        elif name == 'Kalman_IoU_Hungarian':
            return KalmanIoUHungarianTracker()
        raise ValueError(name)

    method_names = [
        'DroneRLTracker (Ours)',
        'DynGate_Hungarian_NoRL',
        'Kalman_IoU_Hungarian',
        'Static250_GreedyRL',
    ]

    all_rows = []
    summary_rows = []
    detailed_out = {'methods': {}, 'sequences': {}}

    # Store sequence metadata
    for s in sequences:
        detailed_out['sequences'][s['name']] = {
            k: v for k, v in s.items() if k != 'gt_frames'
        }

    # Diagnostic check: run2.py len>=2 vs len>=1 for DroneRLTracker
    ours_len1_seq_results = []
    ours_len1_time = 0.0

    for method in method_names:
        print(f"\n=== Evaluating: {method} ===")
        all_seq_res = []
        local4_seq_res = []
        target4_seq_res = []
        all_frames_count = 0
        local4_frames_count = 0
        target4_frames_count = 0
        all_time = 0.0
        local4_time = 0.0
        target4_time = 0.0

        for seq_idx, seq_data in enumerate(sequences):
            tracker = build_tracker(method)
            trajs, elapsed = run_tracker_on_sequence(tracker, seq_data, seq_idx)

            all_frames = list(range(1, seq_data['seq_len'] + 1))
            pred_frames = trajectories_to_pred_frames(trajs, min_len=2)
            eval_res = evaluate_sequence(seq_data['gt_frames'], pred_frames, all_frames=all_frames, iou_thresh=0.50)

            if method == 'DroneRLTracker (Ours)':
                write_mot_predictions(seq_data['name'], trajs, min_len=2)
                pred_frames_l1 = trajectories_to_pred_frames(trajs, min_len=1)
                eval_l1 = evaluate_sequence(seq_data['gt_frames'], pred_frames_l1, all_frames=all_frames, iou_thresh=0.50)
                ours_len1_seq_results.append(eval_l1)
                ours_len1_time += elapsed

            all_seq_res.append(eval_res)
            all_frames_count += seq_data['seq_len']
            all_time += elapsed

            if seq_data['name'] in LOCAL_SUBSET:
                local4_seq_res.append(eval_res)
                local4_frames_count += seq_data['seq_len']
                local4_time += elapsed

            if seq_data['name'] in TARGET_SUBSET_06_28_30_46:
                target4_seq_res.append(eval_res)
                target4_frames_count += seq_data['seq_len']
                target4_time += elapsed

            row = flatten_metrics_row(method, seq_data, eval_res, elapsed, trajs)
            all_rows.append(row)

        agg_all = summarize_aggregate(method, 'All_36_Test_Sequences', all_seq_res, all_frames_count, all_time)
        agg_loc = summarize_aggregate(method, 'Local_4_Sequences', local4_seq_res, local4_frames_count, local4_time)
        agg_tgt = summarize_aggregate(method, 'Subset_06_28_30_46', target4_seq_res, target4_frames_count, target4_time)
        summary_rows.extend([agg_all, agg_loc, agg_tgt])

        print(f"  [36 Seqs] MOTA={agg_all['MOTA']:.2f}% | HOTA={agg_all['HOTA']:.2f}% "
              f"(DetA={agg_all['DetA']:.2f}%, AssA={agg_all['AssA']:.2f}%) | "
              f"IDF1={agg_all['IDF1']:.2f}% | IDSW={agg_all['IDSW']} | FPS={agg_all['FPS']}")
        print(f"  [ 4 Seqs] MOTA={agg_loc['MOTA']:.2f}% | HOTA={agg_loc['HOTA']:.2f}% | "
              f"IDF1={agg_loc['IDF1']:.2f}% | IDSW={agg_loc['IDSW']}")
        print(f"  [06,28,30,46] MOTA={agg_tgt['MOTA']:.2f}% | HOTA={agg_tgt['HOTA']:.2f}% "
              f"(DetA={agg_tgt['DetA']:.2f}%, AssA={agg_tgt['AssA']:.2f}%) | "
              f"IDF1={agg_tgt['IDF1']:.2f}% | IDSW={agg_tgt['IDSW']} | Purity={agg_tgt['Avg_Purity']:.2f}%")

    # Also add len>=1 diagnostic summary for DroneRLTracker
    agg_l1 = summarize_aggregate(
        'DroneRLTracker (Ours, min_len=1 diag)',
        'All_36_Test_Sequences',
        ours_len1_seq_results,
        sum(s['seq_len'] for s in sequences),
        ours_len1_time,
    )
    summary_rows.append(agg_l1)
    print(f"\n  [Diagnostic min_len=1 on DroneRLTracker 36 Seqs] "
          f"MOTA={agg_l1['MOTA']:.2f}% | HOTA={agg_l1['HOTA']:.2f}% | "
          f"IDF1={agg_l1['IDF1']:.2f}% | FP={agg_l1['FP']} | FN={agg_l1['FN']} | IDSW={agg_l1['IDSW']}")

    all_rows_df = pd.DataFrame(all_rows)
    ours_only_df = all_rows_df[all_rows_df['method'] == 'DroneRLTracker (Ours)'].copy()
    summary_df = pd.DataFrame(summary_rows)
    subset_summary_df = summary_df[summary_df['split'] == 'Subset_06_28_30_46'].copy()
    subset_per_seq_df = all_rows_df[all_rows_df['sequence'].isin(TARGET_SUBSET_06_28_30_46)].copy()

    ours_only_df.to_csv(TIER_A_DIR / "per_sequence_metrics.csv", index=False)
    all_rows_df.to_csv(TIER_A_DIR / "per_sequence_all_methods.csv", index=False)
    summary_df.to_csv(TIER_A_DIR / "ablation_summary.csv", index=False)
    subset_summary_df.to_csv(TIER_A_DIR / "subset_06_28_30_46_summary.csv", index=False)
    subset_per_seq_df.to_csv(TIER_A_DIR / "subset_06_28_30_46_per_sequence.csv", index=False)

    detailed_out['summary'] = summary_rows
    detailed_out['per_sequence_ours'] = ours_only_df.to_dict(orient='records')
    with open(TIER_A_DIR / "detailed_results.json", "w") as f:
        json.dump(detailed_out, f, indent=2)

    # Generate plots (exclude diagnostic min_len=1 row from ablation plot)
    plot_summary_df = summary_df[summary_df['method'].isin(method_names)].copy()
    generate_plots(plot_summary_df, all_rows_df)

    print(f"\nSaved all Tier-A evaluation artifacts to {TIER_A_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()
