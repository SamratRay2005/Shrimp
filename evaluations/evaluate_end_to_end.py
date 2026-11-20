#!/usr/bin/env python3
"""
Full 36-Sequence End-to-End (Tier-B) MOT Evaluation Suite
=========================================================
Evaluates real detector outputs (`data/detections_yolox/`, `data/detections_dfine/`) paired
with DroneRLTracker and our 3 controlled tracking ablations across all 36 official
UAVSwarm test sequences (5,754 frames, 325 GT tracks, 37,947 GT boxes).

Outputs saved to evaluations/results/:
  - results/tier_b/end_to_end_summary.csv          (All 36, Local 4, and {06,28,30,46} aggregates)
  - results/tier_b/end_to_end_per_sequence.csv     (36 sequences x detectors x trackers)
  - results/tier_b/end_to_end_subset_06_28_30_46.csv
  - results/tier_b/end_to_end_results.json         (Full summary + val_15-selected threshold + sensitivity)
  - results/predictions/mot_yolox/<seq>.txt
  - results/predictions/mot_dfine/<seq>.txt
  - results/plots/end_to_end_comparison.png
  - results/plots/end_to_end_per_sequence.png
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from run2 import DroneRLTracker
from mot_metrics import evaluate_sequence
from evaluate_tracking import (
    DynGateHungarianNoRLTracker,
    Static250GreedyRLTracker,
    KalmanIoUHungarianTracker,
    load_sequence,
    trajectories_to_pred_frames,
    flatten_metrics_row,
    summarize_aggregate,
    GT_ROOT,
    POLICY_PATH,
    LOCAL_SUBSET,
    TARGET_SUBSET_06_28_30_46,
)

DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
TIER_B_DIR = RESULTS_DIR / "tier_b"
PRED_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_sequence_detections(det_file: Path, conf_thresh: float) -> Dict[int, List[dict]]:
    """Parse MOT detection file (frame,-1,x,y,w,h,score,...) filtered by conf_thresh."""
    frames: Dict[int, List[dict]] = {}
    if not det_file.exists():
        return frames
    with open(det_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            f_idx = int(float(parts[0]))
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])
            if score < conf_thresh or w <= 0.0 or h <= 0.0:
                continue
            frames.setdefault(f_idx, []).append({
                'cx': x1 + w / 2.0,
                'cy': y1 + h / 2.0,
                'box': [x1, y1, x1 + w, y1 + h],
                'score': score,
            })
    return frames


def build_tracker(method_name: str, device: torch.device):
    if method_name == 'DroneRLTracker (Ours)':
        return DroneRLTracker(str(POLICY_PATH), device)
    if method_name == 'DynGate_Hungarian_NoRL':
        return DynGateHungarianNoRLTracker(str(POLICY_PATH), device)
    if method_name == 'Kalman_IoU_Hungarian':
        return KalmanIoUHungarianTracker()
    if method_name == 'Static250_GreedyRL':
        return Static250GreedyRLTracker(str(POLICY_PATH), device)
    raise ValueError(method_name)


def run_tracker_on_detections(tracker, det_frames: Dict[int, List[dict]], seq_len: int):
    t0 = time.perf_counter()
    for f_idx in range(1, seq_len + 1):
        dets = [dict(d) for d in det_frames.get(f_idx, [])]
        tracker.process_frame(dets, f_idx)
    return tracker.trajectories, time.perf_counter() - t0


def write_mot_file(out_dir: Path, seq_name: str, trajectories: dict, min_len: int = 2):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, pts in trajectories.items():
        if len(pts) < min_len:
            continue
        for pt in pts:
            x1, y1, x2, y2 = pt['box']
            rows.append((int(pt['frame']), int(tid), x1, y1, x2 - x1, y2 - y1))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out_dir / f"{seq_name}.txt", "w") as f:
        for fr, tid, x1, y1, w, h in rows:
            f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def generate_e2e_plots(summary_df: pd.DataFrame, per_seq_df: pd.DataFrame):
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

    # Plot 1: End-to-End 36-Sequence Detector x Tracker Comparison
    full = summary_df[summary_df['split'] == 'All_36_Test_Sequences'].copy()
    detectors = list(full['detector'].unique())
    trackers = ['DroneRLTracker (Ours)', 'DynGate_Hungarian_NoRL', 'Kalman_IoU_Hungarian', 'Static250_GreedyRL']

    fig, axes = plt.subplots(1, len(detectors), figsize=(7.2 * len(detectors), 5.2), sharey=True)
    if len(detectors) == 1:
        axes = [axes]

    metrics = ['MOTA', 'HOTA', 'IDF1']
    colors = ['#2563eb', '#059669', '#d97706']
    short_trk = ['DroneRL\n(Ours)', 'DynGate\n(No RL)', 'Kalman+IoU\n(SORT)', 'Static250\n(Greedy)']

    for ax, det_name in zip(axes, detectors):
        sub = full[full['detector'] == det_name].set_index('method').loc[trackers].reset_index()
        x = np.arange(len(trackers))
        width = 0.24
        for idx, (m, col) in enumerate(zip(metrics, colors)):
            vals = sub[m].values
            bars = ax.bar(x + (idx - 1) * width, vals, width, label=m, color=col, alpha=0.92)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 1.0, f"{v:.1f}",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_trk, fontsize=9)
        ax.set_ylim(0, 108)
        ax.set_title(f"{det_name} + Tracker Ablation (36 Seqs)", pad=22)
        ax.set_ylabel("Score (%)")
        ax.legend(bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=3, fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "end_to_end_comparison.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Per-Sequence HOTA / MOTA / IDF1 for Primary Detector + DroneRLTracker
    ours_rows = per_seq_df[per_seq_df['method'] == 'DroneRLTracker (Ours)'].copy()
    if not ours_rows.empty:
        piv = ours_rows.pivot(index='sequence', columns='detector', values='HOTA')
        primary_det = detectors[0]
        order = piv.sort_values(primary_det, ascending=True).index.tolist()

        fig, ax = plt.subplots(figsize=(11.5, 11.0))
        y = np.arange(len(order))
        bar_h = 0.38 if len(detectors) == 2 else 0.55
        det_colors = ['#2563eb', '#059669', '#d97706']

        for i, det_name in enumerate(detectors):
            vals = [piv.loc[seq, det_name] for seq in order]
            offset = (i - (len(detectors) - 1) / 2.0) * bar_h
            ax.barh(y + offset, vals, bar_h, label=f"{det_name} + DroneRL (HOTA)", color=det_colors[i % len(det_colors)], alpha=0.9)

        y_labels = [f"{s}{' ★' if s in LOCAL_SUBSET else ''}" for s in order]
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, fontsize=8.5)
        ax.set_xlim(0, 105)
        ax.set_xlabel("End-to-End HOTA (%)")
        ax.set_title("End-to-End Per-Sequence HOTA Across 36 Unseen UAVSwarm Test Videos", pad=38)
        ax.legend(bbox_to_anchor=(0.5, 1.005), loc='lower center', ncol=len(detectors), frameon=True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "end_to_end_per_sequence.png", dpi=200, bbox_inches='tight')
        plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TIER_B_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    seq_dirs = sorted([d for d in GT_ROOT.iterdir() if d.is_dir()])
    sequences = [load_sequence(d) for d in seq_dirs]
    device = torch.device("cpu")

    detector_configs = []
    yolox_meta_path = TIER_B_DIR / "yolox_eval_metrics.json"
    dfine_meta_path = TIER_B_DIR / "dfine_eval_metrics.json"

    if yolox_meta_path.exists() and (DATA_DIR / "detections_yolox").exists():
        meta = json.loads(yolox_meta_path.read_text())
        detector_configs.append({
            'name': 'YOLOX-M (640x640)',
            'det_dir': DATA_DIR / "detections_yolox",
            'pred_out_dir': PRED_DIR / "mot_yolox",
            'conf_thresh': float(meta['selected_conf_thresh_val15']),
            'meta': meta,
        })

    if dfine_meta_path.exists() and (DATA_DIR / "detections_dfine").exists():
        meta = json.loads(dfine_meta_path.read_text())
        detector_configs.append({
            'name': 'D-FINE-M (640x640)',
            'det_dir': DATA_DIR / "detections_dfine",
            'pred_out_dir': PRED_DIR / "mot_dfine",
            'conf_thresh': float(meta['selected_conf_thresh_val15']),
            'meta': meta,
        })

    if not detector_configs:
        raise FileNotFoundError(f"No detections_yolox/ or detections_dfine/ found in {DATA_DIR}")

    methods = [
        'DroneRLTracker (Ours)',
        'DynGate_Hungarian_NoRL',
        'Kalman_IoU_Hungarian',
        'Static250_GreedyRL',
    ]

    summary_rows = []
    all_seq_rows = []
    thresh_sensitivity = {}

    for dcfg in detector_configs:
        dname = dcfg['name']
        det_dir = dcfg['det_dir']
        conf_thr = dcfg['conf_thresh']
        print(f"\n==================== {dname} (val_15 conf={conf_thr}) ====================")

        # Preload detections once at val_15-selected threshold
        seq_dets = {
            s['name']: load_sequence_detections(det_dir / f"{s['name']}.txt", conf_thr)
            for s in sequences
        }

        for method in methods:
            all_res, loc4_res, sub4_res = [], [], []
            f_all = f_loc4 = f_sub4 = 0
            t_all = t_loc4 = t_sub4 = 0.0

            for seq_data in sequences:
                sname = seq_data['name']
                tracker = build_tracker(method, device)
                trajs, elapsed = run_tracker_on_detections(tracker, seq_dets[sname], seq_data['seq_len'])
                all_frames = list(range(1, seq_data['seq_len'] + 1))
                pred_frames = trajectories_to_pred_frames(trajs, min_len=2)
                eval_res = evaluate_sequence(seq_data['gt_frames'], pred_frames, all_frames=all_frames, iou_thresh=0.50)

                if method == 'DroneRLTracker (Ours)':
                    write_mot_file(dcfg['pred_out_dir'], sname, trajs, min_len=2)

                all_res.append(eval_res)
                f_all += seq_data['seq_len']
                t_all += elapsed
                if sname in LOCAL_SUBSET:
                    loc4_res.append(eval_res)
                    f_loc4 += seq_data['seq_len']
                    t_loc4 += elapsed
                if sname in TARGET_SUBSET_06_28_30_46:
                    sub4_res.append(eval_res)
                    f_sub4 += seq_data['seq_len']
                    t_sub4 += elapsed

                row = flatten_metrics_row(method, seq_data, eval_res, elapsed, trajs)
                row['detector'] = dname
                row['conf_thresh'] = conf_thr
                all_seq_rows.append(row)

            for split_name, res_list, f_cnt, t_cnt in [
                ('All_36_Test_Sequences', all_res, f_all, t_all),
                ('Local_4_Sequences', loc4_res, f_loc4, t_loc4),
                ('Subset_06_28_30_46', sub4_res, f_sub4, t_sub4),
            ]:
                agg = summarize_aggregate(method, split_name, res_list, f_cnt, t_cnt)
                agg['detector'] = dname
                agg['conf_thresh'] = conf_thr
                summary_rows.append(agg)

            agg36 = summary_rows[-3]
            print(f"  {method:26s} | MOTA={agg36['MOTA']:.2f}% HOTA={agg36['HOTA']:.2f}% "
                  f"(DetA={agg36['DetA']:.2f} AssA={agg36['AssA']:.2f}) IDF1={agg36['IDF1']:.2f}% "
                  f"FP={agg36['FP']} FN={agg36['FN']} IDSW={agg36['IDSW']} FPS={agg36['FPS']}")

        # Post-hoc threshold sensitivity analysis for DroneRLTracker (diagnostic only)
        sens_list = []
        for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            s_res = []
            tot_f = 0
            tot_t = 0.0
            for seq_data in sequences:
                sdets = load_sequence_detections(det_dir / f"{seq_data['name']}.txt", thr)
                trk = DroneRLTracker(str(POLICY_PATH), device)
                trajs, elapsed = run_tracker_on_detections(trk, sdets, seq_data['seq_len'])
                pred_frames = trajectories_to_pred_frames(trajs, min_len=2)
                eval_res = evaluate_sequence(
                    seq_data['gt_frames'], pred_frames,
                    all_frames=list(range(1, seq_data['seq_len'] + 1)), iou_thresh=0.50
                )
                s_res.append(eval_res)
                tot_f += seq_data['seq_len']
                tot_t += elapsed
            agg = summarize_aggregate('DroneRLTracker (Ours)', 'All_36_Test_Sequences', s_res, tot_f, tot_t)
            agg['detector'] = dname
            agg['conf_thresh'] = thr
            sens_list.append(agg)
        thresh_sensitivity[dname] = sens_list

    summary_df = pd.DataFrame(summary_rows)
    per_seq_df = pd.DataFrame(all_seq_rows)
    subset_df = per_seq_df[per_seq_df['sequence'].isin(TARGET_SUBSET_06_28_30_46)].copy()

    summary_df.to_csv(TIER_B_DIR / "end_to_end_summary.csv", index=False)
    per_seq_df.to_csv(TIER_B_DIR / "end_to_end_per_sequence.csv", index=False)
    subset_df.to_csv(TIER_B_DIR / "end_to_end_subset_06_28_30_46.csv", index=False)

    with open(TIER_B_DIR / "end_to_end_results.json", "w") as f:
        json.dump({
            'summary': summary_rows,
            'threshold_sensitivity': thresh_sensitivity,
            'detectors': {d['name']: d['meta'] for d in detector_configs},
        }, f, indent=2)

    generate_e2e_plots(summary_df, per_seq_df)
    print(f"\nSaved all end-to-end Tier-B evaluation artifacts to {TIER_B_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()
