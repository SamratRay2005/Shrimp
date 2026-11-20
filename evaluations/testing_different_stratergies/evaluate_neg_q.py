#!/usr/bin/env python3
"""
Negative-Q Hungarian Bipartite Matching Ablation (Tier A & Tier B)
==================================================================
Evaluates `DynGate_Hungarian_NegQ` across all 36 official UAVSwarm test sequences
(5,754 frames, 325 GT tracks, 37,947 MOT GT boxes) using:
  cost_matrix[i, j] = -float(q)
inside the velocity-adaptive dynamic gate (`dist <= dyn_gate`), leaving all
other components (SimpleKF, [20, 65] px dynamic gate, step_d guard, 0.7/0.3
EMA velocity, MAX_MISSED=8, and min_len>=2 export rule) identical to run2.py.

Benchmarks three inputs on CPU:
  1. Tier A: Oracle GT boxes (track IDs stripped, shuffled per frame)
  2. Tier B: D-FINE-M (640x640) at val_15-selected threshold (conf=0.55)
  3. Tier B: YOLOX-M (640x640) at val_15-selected threshold (conf=0.50)

Saves non-destructive artifacts to evaluations/testing_different_stratergies/results/:
  - neg_q_summary.csv
  - neg_q_per_sequence.csv
  - neg_q_subset_06_28_30_46.csv
  - neg_q_results.json
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

STRAT_DIR = Path(__file__).resolve().parent
STRAT_RESULTS_DIR = STRAT_DIR / "results"
EVAL_DIR = STRAT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"
TIER_A_DIR = EVAL_DIR / "results" / "tier_a"
TIER_B_DIR = EVAL_DIR / "results" / "tier_b"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from run2 import DroneRLTracker, MAX_MISSED, SimpleKF, TRAIL_LENGTH
from mot_metrics import evaluate_sequence
from evaluate_tracking import (
    GT_ROOT,
    LOCAL_SUBSET,
    POLICY_PATH,
    TARGET_SUBSET_06_28_30_46,
    flatten_metrics_row,
    load_sequence,
    run_tracker_on_sequence,
    summarize_aggregate,
    trajectories_to_pred_frames,
)
from evaluate_end_to_end import load_sequence_detections, run_tracker_on_detections


METHOD_NAME = "DynGate_Hungarian_NegQ"


class DynGateHungarianNegQTracker(DroneRLTracker):
    """
    Controlled ablation of DroneRLTracker using `cost_matrix[i, j] = -float(q)`
    for Hungarian bipartite assignment inside the dynamic spatial gate.
    """

    def __init__(self, policy_path: str, device: torch.device):
        super().__init__(policy_path, device)
        self.q_min = float("inf")
        self.q_max = float("-inf")
        self.q_sum = 0.0
        self.q_sq_sum = 0.0
        self.q_count = 0

    def process_frame(self, detections: list, frame_idx: int) -> list:
        tids = list(self.tracks.keys())
        active = []

        if not tids:
            for det in detections:
                tid = self.next_id
                gt_id = det.get("gt_id", tid)
                self.tracks[tid] = {
                    "kf": SimpleKF(det["cx"], det["cy"]),
                    "box": det["box"],
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "vx": 0.0,
                    "vy": 0.0,
                    "hits": 1,
                    "missed": 0,
                    "history": [(int(det["cx"]), int(det["cy"]))],
                    "gt_id": gt_id,
                }
                self.trajectories[tid] = [{
                    "frame": frame_idx,
                    "cx": float(det["cx"]),
                    "cy": float(det["cy"]),
                    "vx": 0.0,
                    "vy": 0.0,
                    "box": [float(b) for b in det["box"]],
                    "gt_id": gt_id,
                }]
                self.next_id += 1
            return []

        cost_matrix = np.full((len(tids), len(detections)), 1e5)
        gates = []

        for i, tid in enumerate(tids):
            data = self.tracks[tid]
            pred_cx, pred_cy = data["kf"].predict()
            vx = data["vx"]
            vy = data["vy"]
            dyn_gate = self.compute_dynamic_gate(vx, vy, data["missed"])
            gates.append(dyn_gate)

            cands = []
            for j, det in enumerate(detections):
                dx = det["cx"] - pred_cx
                dy = det["cy"] - pred_cy
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

                for (j, _, _, _, _), q in zip(cands, q_vals):
                    q_val = float(q)
                    if not math.isfinite(q_val) or -q_val >= 1e4:
                        raise RuntimeError(f"Invalid Q value encountered: q={q_val}")
                    self.q_min = min(self.q_min, q_val)
                    self.q_max = max(self.q_max, q_val)
                    self.q_sum += q_val
                    self.q_sq_sum += q_val * q_val
                    self.q_count += 1
                    cost_matrix[i, j] = -q_val

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            tid = tids[r]
            dyn_gate = gates[r]
            if cost_matrix[r, c] < 1e4:
                det = detections[c]
                data = self.tracks[tid]
                step_d = math.hypot(det["cx"] - data["cx"], det["cy"] - data["cy"])
                if step_d > dyn_gate + 5.0:
                    continue

                vx = det["cx"] - data["cx"]
                vy = det["cy"] - data["cy"]
                data["vx"] = 0.7 * data["vx"] + 0.3 * vx
                data["vy"] = 0.7 * data["vy"] + 0.3 * vy
                data["cx"] = det["cx"]
                data["cy"] = det["cy"]
                data["box"] = det["box"]
                data["gt_id"] = det.get("gt_id", data["gt_id"])
                data["kf"].update(det["cx"], det["cy"])
                data["history"].append((int(det["cx"]), int(det["cy"])))
                if len(data["history"]) > TRAIL_LENGTH:
                    data["history"].pop(0)

                data["hits"] += 1
                data["missed"] = 0

                if tid not in self.trajectories:
                    self.trajectories[tid] = []
                self.trajectories[tid].append({
                    "frame": frame_idx,
                    "cx": float(det["cx"]),
                    "cy": float(det["cy"]),
                    "vx": float(vx),
                    "vy": float(vy),
                    "box": [float(b) for b in det["box"]],
                    "gt_id": data["gt_id"],
                })

                matched_tracks.add(tid)
                matched_dets.add(c)

                if data["hits"] >= 2:
                    active.append({
                        "track_id": tid,
                        "gt_id": data["gt_id"],
                        "box": det["box"],
                        "cx": det["cx"],
                        "cy": det["cy"],
                        "history": list(data["history"]),
                        "gate": dyn_gate,
                        "speed": math.hypot(data["vx"], data["vy"]),
                    })

        for tid in tids:
            if tid not in matched_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > MAX_MISSED:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                gt_id = det.get("gt_id", tid)
                self.tracks[tid] = {
                    "kf": SimpleKF(det["cx"], det["cy"]),
                    "box": det["box"],
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "vx": 0.0,
                    "vy": 0.0,
                    "hits": 1,
                    "missed": 0,
                    "history": [(int(det["cx"]), int(det["cy"]))],
                    "gt_id": gt_id,
                }
                if tid not in self.trajectories:
                    self.trajectories[tid] = []
                self.trajectories[tid].append({
                    "frame": frame_idx,
                    "cx": float(det["cx"]),
                    "cy": float(det["cy"]),
                    "vx": 0.0,
                    "vy": 0.0,
                    "box": [float(b) for b in det["box"]],
                    "gt_id": gt_id,
                })
                self.next_id += 1

        return active


def evaluate_mode(
    tier: str,
    detector_label: str,
    sequences: List[dict],
    device: torch.device,
    det_dir: Optional[Path] = None,
    conf_thresh: Optional[float] = None,
) -> tuple:
    """Run DynGateHungarianNegQTracker across all 36 sequences for one input mode."""
    all_res, loc4_res, sub4_res = [], [], []
    f_all = f_loc4 = f_sub4 = 0
    t_all = t_loc4 = t_sub4 = 0.0
    per_seq_rows = []

    q_min = float("inf")
    q_max = float("-inf")
    q_sum = 0.0
    q_sq_sum = 0.0
    q_count = 0

    for seq_idx, seq_data in enumerate(sequences):
        sname = seq_data["name"]
        tracker = DynGateHungarianNegQTracker(str(POLICY_PATH), device)

        if det_dir is None:
            trajs, elapsed = run_tracker_on_sequence(tracker, seq_data, seq_idx)
        else:
            assert conf_thresh is not None
            sdets = load_sequence_detections(det_dir / f"{sname}.txt", conf_thresh)
            trajs, elapsed = run_tracker_on_detections(tracker, sdets, seq_data["seq_len"])

        if tracker.q_count > 0:
            q_min = min(q_min, tracker.q_min)
            q_max = max(q_max, tracker.q_max)
            q_sum += tracker.q_sum
            q_sq_sum += tracker.q_sq_sum
            q_count += tracker.q_count

        all_frames = list(range(1, seq_data["seq_len"] + 1))
        pred_frames = trajectories_to_pred_frames(trajs, min_len=2)
        eval_res = evaluate_sequence(
            seq_data["gt_frames"], pred_frames, all_frames=all_frames, iou_thresh=0.50
        )

        all_res.append(eval_res)
        f_all += seq_data["seq_len"]
        t_all += elapsed
        if sname in LOCAL_SUBSET:
            loc4_res.append(eval_res)
            f_loc4 += seq_data["seq_len"]
            t_loc4 += elapsed
        if sname in TARGET_SUBSET_06_28_30_46:
            sub4_res.append(eval_res)
            f_sub4 += seq_data["seq_len"]
            t_sub4 += elapsed

        row = flatten_metrics_row(METHOD_NAME, seq_data, eval_res, elapsed, trajs)
        row["tier"] = tier
        row["detector"] = detector_label
        row["conf_thresh"] = conf_thresh if conf_thresh is not None else 1.0
        per_seq_rows.append(row)

    summary_rows = []
    for split_name, res_list, f_cnt, t_cnt in [
        ("All_36_Test_Sequences", all_res, f_all, t_all),
        ("Local_4_Sequences", loc4_res, f_loc4, t_loc4),
        ("Subset_06_28_30_46", sub4_res, f_sub4, t_sub4),
    ]:
        agg = summarize_aggregate(METHOD_NAME, split_name, res_list, f_cnt, t_cnt)
        agg["tier"] = tier
        agg["detector"] = detector_label
        agg["conf_thresh"] = conf_thresh if conf_thresh is not None else 1.0
        summary_rows.append(agg)

    mean_q = q_sum / max(1, q_count)
    var_q = max(0.0, (q_sq_sum / max(1, q_count)) - mean_q * mean_q)
    q_stats = {
        "q_count": q_count,
        "q_min": round(q_min, 4) if q_count else None,
        "q_max": round(q_max, 4) if q_count else None,
        "q_mean": round(mean_q, 4) if q_count else None,
        "q_std": round(math.sqrt(var_q), 4) if q_count else None,
        "neg_q_max": round(-q_min, 4) if q_count else None,
        "neg_q_min": round(-q_max, 4) if q_count else None,
    }
    return summary_rows, per_seq_rows, q_stats


def format_comparison_block(
    title: str,
    neg_q_row: dict,
    ref_df: pd.DataFrame,
) -> Dict[str, dict]:
    """Print side-by-side table against stored baselines and return delta dict."""
    print(f"\n=== {title} ===")
    cols = ["method", "MOTA", "HOTA", "DetA", "AssA", "IDF1", "FP", "FN", "IDSW", "Avg_Purity", "FPS"]
    neg_df = pd.DataFrame([{k: neg_q_row[k] for k in cols}])
    comp_df = pd.concat([neg_df, ref_df[cols]], ignore_index=True)
    print(comp_df.to_string(index=False))

    ours = ref_df[ref_df["method"] == "DroneRLTracker (Ours)"].iloc[0]
    deltas = {}
    for k in ["MOTA", "HOTA", "DetA", "AssA", "IDF1", "IDSW", "FN", "FP", "Avg_Purity"]:
        diff = float(neg_q_row[k]) - float(ours[k])
        deltas[k] = round(diff, 2) if isinstance(neg_q_row[k], float) else int(diff)
    return deltas


def main():
    STRAT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seq_dirs = sorted([d for d in GT_ROOT.iterdir() if d.is_dir()])
    sequences = [load_sequence(d) for d in seq_dirs]
    device = torch.device("cpu")

    dfine_meta = json.loads((TIER_B_DIR / "dfine_eval_metrics.json").read_text())
    yolox_meta = json.loads((TIER_B_DIR / "yolox_eval_metrics.json").read_text())
    dfine_conf = float(dfine_meta["selected_conf_thresh_val15"])
    yolox_conf = float(yolox_meta["selected_conf_thresh_val15"])

    modes = [
        ("Tier A", "Oracle_GT", None, None),
        ("Tier B", "D-FINE-M (640x640)", DATA_DIR / "detections_dfine", dfine_conf),
        ("Tier B", "YOLOX-M (640x640)", DATA_DIR / "detections_yolox", yolox_conf),
    ]

    all_summary = []
    all_per_seq = []
    q_stats_by_mode = {}

    for tier, det_label, det_dir, conf in modes:
        s_rows, p_rows, q_stats = evaluate_mode(tier, det_label, sequences, device, det_dir, conf)
        all_summary.extend(s_rows)
        all_per_seq.extend(p_rows)
        q_stats_by_mode[f"{tier} | {det_label}"] = q_stats

    summary_df = pd.DataFrame(all_summary)
    per_seq_df = pd.DataFrame(all_per_seq)
    subset_df = per_seq_df[per_seq_df["sequence"].isin(TARGET_SUBSET_06_28_30_46)].copy()

    summary_df.to_csv(STRAT_RESULTS_DIR / "neg_q_summary.csv", index=False)
    per_seq_df.to_csv(STRAT_RESULTS_DIR / "neg_q_per_sequence.csv", index=False)
    subset_df.to_csv(STRAT_RESULTS_DIR / "neg_q_subset_06_28_30_46.csv", index=False)

    # Load stored reference summaries for comparison
    tier_a_ref = pd.read_csv(TIER_A_DIR / "ablation_summary.csv")
    tier_b_ref = pd.read_csv(TIER_B_DIR / "end_to_end_summary.csv")
    tier_a_seq_ref = pd.read_csv(TIER_A_DIR / "per_sequence_metrics.csv")
    tier_b_seq_ref = pd.read_csv(TIER_B_DIR / "end_to_end_per_sequence.csv")

    deltas_vs_ours = {}
    top_seq_diffs = {}

    for det_label, ref_sub, seq_ref in [
        (
            "Oracle_GT",
            tier_a_ref[
                (tier_a_ref["split"] == "All_36_Test_Sequences")
                & (~tier_a_ref["method"].str.contains("diag"))
            ],
            tier_a_seq_ref,
        ),
        (
            "D-FINE-M (640x640)",
            tier_b_ref[
                (tier_b_ref["split"] == "All_36_Test_Sequences")
                & (tier_b_ref["detector"] == "D-FINE-M (640x640)")
            ],
            tier_b_seq_ref[
                (tier_b_seq_ref["detector"] == "D-FINE-M (640x640)")
                & (tier_b_seq_ref["method"] == "DroneRLTracker (Ours)")
            ],
        ),
        (
            "YOLOX-M (640x640)",
            tier_b_ref[
                (tier_b_ref["split"] == "All_36_Test_Sequences")
                & (tier_b_ref["detector"] == "YOLOX-M (640x640)")
            ],
            tier_b_seq_ref[
                (tier_b_seq_ref["detector"] == "YOLOX-M (640x640)")
                & (tier_b_seq_ref["method"] == "DroneRLTracker (Ours)")
            ],
        ),
    ]:
        neg_row = summary_df[
            (summary_df["detector"] == det_label) & (summary_df["split"] == "All_36_Test_Sequences")
        ].iloc[0].to_dict()
        deltas_vs_ours[det_label] = format_comparison_block(
            f"{det_label} — All 36 Sequences", neg_row, ref_sub
        )

        # Per-sequence deltas vs DroneRLTracker (Ours)
        neg_seq = per_seq_df[per_seq_df["detector"] == det_label][
            ["sequence", "MOTA", "HOTA", "IDF1", "IDSW", "Avg_Purity"]
        ].rename(columns=lambda c: c if c == "sequence" else f"NegQ_{c}")
        ours_seq = seq_ref[["sequence", "MOTA", "HOTA", "IDF1", "IDSW", "Avg_Purity"]].rename(
            columns=lambda c: c if c == "sequence" else f"Ours_{c}"
        )
        merged = neg_seq.merge(ours_seq, on="sequence")
        merged["dHOTA"] = (merged["NegQ_HOTA"] - merged["Ours_HOTA"]).round(2)
        merged["dMOTA"] = (merged["NegQ_MOTA"] - merged["Ours_MOTA"]).round(2)
        merged["dIDSW"] = merged["NegQ_IDSW"] - merged["Ours_IDSW"]
        changed = merged[
            (merged["dHOTA"].abs() > 0) | (merged["dMOTA"].abs() > 0) | (merged["dIDSW"] != 0)
        ].copy()
        changed["abs_dHOTA"] = changed["dHOTA"].abs()
        changed = changed.sort_values(["abs_dHOTA", "dIDSW"], ascending=[False, False]).drop(
            columns=["abs_dHOTA"]
        )
        top_seq_diffs[det_label] = changed.head(8).to_dict(orient="records")
        print(f"\n  [{det_label}] Changed sequences vs DroneRLTracker: {len(changed)} / 36")
        if not changed.empty:
            print(
                changed[["sequence", "NegQ_MOTA", "Ours_MOTA", "NegQ_HOTA", "Ours_HOTA", "dHOTA", "NegQ_IDSW", "Ours_IDSW", "dIDSW"]]
                .head(8)
                .to_string(index=False)
            )

    print("\n=== Q-Value Statistics During DynGate_Hungarian_NegQ ===")
    print(json.dumps(q_stats_by_mode, indent=2))

    with open(STRAT_RESULTS_DIR / "neg_q_results.json", "w") as f:
        json.dump(
            {
                "cost_formula": "cost_matrix[i, j] = -float(q) for dist <= dyn_gate",
                "summary": all_summary,
                "q_stats": q_stats_by_mode,
                "deltas_vs_drone_rl_tracker": deltas_vs_ours,
                "top_per_sequence_diffs": top_seq_diffs,
            },
            f,
            indent=2,
        )

    print(f"\nSaved Negative-Q evaluation artifacts to {STRAT_RESULTS_DIR}")


if __name__ == "__main__":
    main()
