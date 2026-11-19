#!/usr/bin/env python3
"""
Exact MOT Evaluation Metrics (CLEAR MOT / MOTA, IDF1, HOTA, Purity & Kinematics)
=================================================================================
Implements standard multi-object tracking metrics directly from published specs:
  1. CLEAR MOT (Bernardin & Stiefelhagen, EURASIP 2008): MOTA, MOTP, IDSW, Frag, MT/PT/ML
  2. Identity Metrics (Ristani et al., ECCV 2016): IDF1, IDP, IDR, IDTP, IDFP, IDFN
  3. Higher Order Tracking Accuracy (Luiten et al., IJCV 2021): HOTA, DetA, AssA, LocA, AssRe, AssPr
     evaluated across all 19 localization thresholds alpha in {0.05, 0.10, ..., 0.95}.
  4. Domain Kinematic & Track Purity Metrics: Track Purity, 100%-Pure Tracks, Erratic Jumps (>25px).

Supports both per-sequence evaluation and exact dataset-level accumulator aggregation
across multiple sequences (matching TrackEval's `combine_sequences` convention).
"""

import math
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


HOTA_ALPHAS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix between boxes_a (Na, 4) and boxes_b (Nb, 4)
    in [x1, y1, x2, y2] format.
    """
    na = len(boxes_a)
    nb = len(boxes_b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float64)

    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)

    x1 = np.maximum(a[:, 0:1], b[None, :, 0])
    y1 = np.maximum(a[:, 1:2], b[None, :, 1])
    x2 = np.minimum(a[:, 2:3], b[None, :, 2])
    y2 = np.minimum(a[:, 3:4], b[None, :, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.zeros((na, nb), dtype=np.float64)
    valid = union > 0.0
    iou[valid] = inter[valid] / union[valid]
    return iou


def _prepare_frames(
    gt_frames: Dict[int, List[dict]],
    pred_frames: Dict[int, List[dict]],
    all_frames: List[int] = None,
):
    """Normalize frame keys and precompute IoU matrices once per frame."""
    if all_frames is None:
        frame_set = set(gt_frames.keys()) | set(pred_frames.keys())
        all_frames = sorted(frame_set)

    frame_data = []
    for f in all_frames:
        gts = gt_frames.get(f, [])
        preds = pred_frames.get(f, [])
        gt_ids = [int(d['id']) for d in gts]
        pred_ids = [int(d['id']) for d in preds]
        gt_boxes = np.array([d['box'] for d in gts], dtype=np.float64) if gts else np.zeros((0, 4), dtype=np.float64)
        pred_boxes = np.array([d['box'] for d in preds], dtype=np.float64) if preds else np.zeros((0, 4), dtype=np.float64)
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        frame_data.append((f, gt_ids, gt_boxes, pred_ids, pred_boxes, iou_mat))
    return frame_data


def evaluate_clear_mot(frame_data, iou_thresh: float = 0.50) -> dict:
    """
    CLEAR MOT evaluation (Bernardin & Stiefelhagen, 2008) at IoU >= iou_thresh.
    Returns raw accumulators + computed metrics.
    """
    prev_gt_to_pred: Dict[int, int] = {}
    last_match: Dict[int, int] = {}
    was_ever_tracked = set()
    currently_tracked: Dict[int, bool] = {}

    gt_total_count: Dict[int, int] = {}
    gt_tracked_count: Dict[int, int] = {}

    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    frag = 0
    motp_sum = 0.0

    for _, gt_ids, _, pred_ids, _, iou_mat in frame_data:
        n_gt = len(gt_ids)
        n_pred = len(pred_ids)
        total_gt += n_gt
        total_pred += n_pred

        for g_id in gt_ids:
            gt_total_count[g_id] = gt_total_count.get(g_id, 0) + 1

        gt_idx_map = {g: i for i, g in enumerate(gt_ids)}
        pred_idx_map = {p: j for j, p in enumerate(pred_ids)}

        matched_gt = set()
        matched_pred = set()
        curr_matches: Dict[int, Tuple[int, float]] = {}

        # Step 1: Preserve correspondences from t-1 when still valid (IoU >= iou_thresh)
        for g_id, prev_p_id in prev_gt_to_pred.items():
            if g_id in gt_idx_map and prev_p_id in pred_idx_map:
                gi = gt_idx_map[g_id]
                pj = pred_idx_map[prev_p_id]
                iou = float(iou_mat[gi, pj])
                if iou >= iou_thresh:
                    matched_gt.add(gi)
                    matched_pred.add(pj)
                    curr_matches[g_id] = (prev_p_id, iou)

        # Step 2: Optimal Hungarian matching on remaining unmatched objects
        rem_gt = [i for i in range(n_gt) if i not in matched_gt]
        rem_pred = [j for j in range(n_pred) if j not in matched_pred]

        if rem_gt and rem_pred:
            sub_iou = iou_mat[np.ix_(rem_gt, rem_pred)]
            cost = np.where(sub_iou >= iou_thresh, 1.0 - sub_iou, 1e6)
            r_ind, c_ind = linear_sum_assignment(cost)
            for r, c in zip(r_ind, c_ind):
                if cost[r, c] < 1e5:
                    gi = rem_gt[r]
                    pj = rem_pred[c]
                    iou = float(iou_mat[gi, pj])
                    if iou >= iou_thresh:
                        g_id = gt_ids[gi]
                        p_id = pred_ids[pj]
                        matched_gt.add(gi)
                        matched_pred.add(pj)
                        curr_matches[g_id] = (p_id, iou)

        # Step 3: Accumulate TP, FP, FN, IDSW, Frag, MOTP
        next_prev: Dict[int, int] = {}
        for g_id in gt_ids:
            if g_id in curr_matches:
                p_id, iou = curr_matches[g_id]
                tp += 1
                motp_sum += iou
                gt_tracked_count[g_id] = gt_tracked_count.get(g_id, 0) + 1

                if g_id in last_match and last_match[g_id] != p_id:
                    idsw += 1
                if g_id in was_ever_tracked and not currently_tracked.get(g_id, False):
                    frag += 1

                last_match[g_id] = p_id
                was_ever_tracked.add(g_id)
                currently_tracked[g_id] = True
                next_prev[g_id] = p_id
            else:
                fn += 1
                currently_tracked[g_id] = False

        fp += (n_pred - len(matched_pred))
        prev_gt_to_pred = next_prev

    mt = pt = ml = 0
    for g_id, tot in gt_total_count.items():
        ratio = gt_tracked_count.get(g_id, 0) / tot if tot > 0 else 0.0
        if ratio >= 0.80:
            mt += 1
        elif ratio >= 0.20:
            pt += 1
        else:
            ml += 1

    return finalize_clear_mot({
        'total_gt': total_gt,
        'total_pred': total_pred,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'idsw': idsw,
        'frag': frag,
        'motp_sum': motp_sum,
        'mt': mt,
        'pt': pt,
        'ml': ml,
        'num_gt_ids': len(gt_total_count),
    })


def finalize_clear_mot(acc: dict) -> dict:
    """Compute CLEAR MOT ratios from raw accumulator counts."""
    res = dict(acc)
    total_gt = acc['total_gt']
    tp = acc['tp']
    fp = acc['fp']
    fn = acc['fn']
    idsw = acc['idsw']

    res['mota'] = (1.0 - (fp + fn + idsw) / total_gt) if total_gt > 0 else 0.0
    res['motp'] = (acc['motp_sum'] / tp) if tp > 0 else 0.0
    res['precision'] = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    res['recall'] = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return res


def evaluate_idf1(frame_data, iou_thresh: float = 0.50) -> dict:
    """
    Identity Metrics (IDF1, IDP, IDR) following Ristani et al. (ECCV 2016) / TrackEval.
    Computes global 1-to-1 optimal bipartite matching between GT IDs and Pred IDs.
    """
    gt_to_idx: Dict[int, int] = {}
    pred_to_idx: Dict[int, int] = {}
    gt_counts: List[int] = []
    pred_counts: List[int] = []

    for _, gt_ids, _, pred_ids, _, _ in frame_data:
        for g in gt_ids:
            if g not in gt_to_idx:
                gt_to_idx[g] = len(gt_counts)
                gt_counts.append(0)
            gt_counts[gt_to_idx[g]] += 1
        for p in pred_ids:
            if p not in pred_to_idx:
                pred_to_idx[p] = len(pred_counts)
                pred_counts.append(0)
            pred_counts[pred_to_idx[p]] += 1

    n_g = len(gt_counts)
    n_p = len(pred_counts)
    total_gt = int(sum(gt_counts))
    total_pred = int(sum(pred_counts))

    if n_g == 0 or n_p == 0:
        return finalize_idf1({
            'idtp': 0,
            'idfp': total_pred,
            'idfn': total_gt,
            'total_gt': total_gt,
            'total_pred': total_pred,
        })

    # Count frames where GT g and Pred p have IoU >= iou_thresh.
    # Since global matching pairs each GT ID with at most 1 Pred ID and each Pred ID
    # with at most 1 GT ID, and each frame contains unique IDs, IDTP <= min(total_gt, total_pred).
    co_occur = np.zeros((n_g, n_p), dtype=np.int64)
    for _, gt_ids, _, pred_ids, _, iou_mat in frame_data:
        if len(gt_ids) == 0 or len(pred_ids) == 0:
            continue
        g_idx = np.array([gt_to_idx[g] for g in gt_ids], dtype=np.int64)
        p_idx = np.array([pred_to_idx[p] for p in pred_ids], dtype=np.int64)
        valid = iou_mat >= iou_thresh
        rows, cols = np.where(valid)
        for r, c in zip(rows, cols):
            co_occur[g_idx[r], p_idx[c]] += 1

    r_ind, c_ind = linear_sum_assignment(-co_occur)
    idtp = int(co_occur[r_ind, c_ind].sum())
    idfp = total_pred - idtp
    idfn = total_gt - idtp

    return finalize_idf1({
        'idtp': idtp,
        'idfp': idfp,
        'idfn': idfn,
        'total_gt': total_gt,
        'total_pred': total_pred,
    })


def finalize_idf1(acc: dict) -> dict:
    """Compute IDF1, IDP, IDR from raw IDTP, IDFP, IDFN counts."""
    res = dict(acc)
    idtp = acc['idtp']
    idfp = acc['idfp']
    idfn = acc['idfn']
    res['idp'] = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
    res['idr'] = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
    denom = 2 * idtp + idfp + idfn
    res['idf1'] = (2.0 * idtp / denom) if denom > 0 else 0.0
    return res


def evaluate_hota(frame_data, alphas: np.ndarray = HOTA_ALPHAS) -> dict:
    """
    HOTA evaluation (Luiten et al., IJCV 2021 / TrackEval specification)
    across all localization IoU thresholds `alphas` (0.05..0.95 step 0.05).

    Returns raw per-alpha accumulators plus sequence-level HOTA, DetA, AssA, LocA, AssRe, AssPr.
    """
    gt_to_idx: Dict[int, int] = {}
    pred_to_idx: Dict[int, int] = {}
    gt_counts: List[int] = []
    pred_counts: List[int] = []

    indexed_frames = []
    for _, gt_ids, _, pred_ids, _, iou_mat in frame_data:
        g_arr = np.empty(len(gt_ids), dtype=np.int64)
        for i, g in enumerate(gt_ids):
            if g not in gt_to_idx:
                gt_to_idx[g] = len(gt_counts)
                gt_counts.append(0)
            idx = gt_to_idx[g]
            gt_counts[idx] += 1
            g_arr[i] = idx

        p_arr = np.empty(len(pred_ids), dtype=np.int64)
        for j, p in enumerate(pred_ids):
            if p not in pred_to_idx:
                pred_to_idx[p] = len(pred_counts)
                pred_counts.append(0)
            idx = pred_to_idx[p]
            pred_counts[idx] += 1
            p_arr[j] = idx

        indexed_frames.append((g_arr, p_arr, iou_mat))

    n_g = len(gt_counts)
    n_p = len(pred_counts)
    n_a = len(alphas)
    gt_cnt_arr = np.array(gt_counts, dtype=np.float64)
    pred_cnt_arr = np.array(pred_counts, dtype=np.float64)
    total_gt = int(sum(gt_counts))
    total_pred = int(sum(pred_counts))

    tp_arr = np.zeros(n_a, dtype=np.float64)
    fn_arr = np.zeros(n_a, dtype=np.float64)
    fp_arr = np.zeros(n_a, dtype=np.float64)
    loca_sum_arr = np.zeros(n_a, dtype=np.float64)
    sum_ass_iou_arr = np.zeros(n_a, dtype=np.float64)
    sum_ass_re_arr = np.zeros(n_a, dtype=np.float64)
    sum_ass_pr_arr = np.zeros(n_a, dtype=np.float64)

    if n_g == 0 or n_p == 0:
        fn_arr[:] = float(total_gt)
        fp_arr[:] = float(total_pred)
        return finalize_hota({
            'alphas': alphas.tolist(),
            'tp': tp_arr,
            'fn': fn_arr,
            'fp': fp_arr,
            'loca_sum': loca_sum_arr,
            'sum_ass_iou': sum_ass_iou_arr,
            'sum_ass_re': sum_ass_re_arr,
            'sum_ass_pr': sum_ass_pr_arr,
        })

    # Precompute global potential similarity matrix (TrackEval Pass 1)
    # Note: In HOTA, global_alignment_score uses per-frame normalized similarity across valid pairs.
    # For threshold alpha, we compute potential_matches_count using frames where iou >= alpha.
    for a_idx, alpha in enumerate(alphas):
        potential = np.zeros((n_g, n_p), dtype=np.float64)
        for g_arr, p_arr, iou_mat in indexed_frames:
            if len(g_arr) == 0 or len(p_arr) == 0:
                continue
            sim = np.where(iou_mat >= alpha, iou_mat, 0.0)
            row_sum = sim.sum(axis=1, keepdims=True)
            col_sum = sim.sum(axis=0, keepdims=True)
            denom = row_sum + col_sum - sim
            sim_iou = np.zeros_like(sim)
            nz = denom > 0.0
            sim_iou[nz] = sim[nz] / denom[nz]
            potential[np.ix_(g_arr, p_arr)] += sim_iou

        denom_global = gt_cnt_arr[:, None] + pred_cnt_arr[None, :] - potential
        global_align = np.zeros_like(potential)
        nz_g = denom_global > 0.0
        global_align[nz_g] = potential[nz_g] / denom_global[nz_g]

        # Pass 2: Per-frame Hungarian maximizing global_alignment_score (with tiny IoU tie-break)
        matches = np.zeros((n_g, n_p), dtype=np.float64)
        tp_a = 0.0
        loca_sum_a = 0.0

        for g_arr, p_arr, iou_mat in indexed_frames:
            if len(g_arr) == 0 or len(p_arr) == 0:
                continue
            valid = iou_mat >= alpha
            if not np.any(valid):
                continue
            score = global_align[np.ix_(g_arr, p_arr)] + 1e-6 * iou_mat
            # Disallow invalid matches (< alpha)
            cost = np.where(valid, -score, 1e6)
            r_ind, c_ind = linear_sum_assignment(cost)
            for r, c in zip(r_ind, c_ind):
                if valid[r, c]:
                    matches[g_arr[r], p_arr[c]] += 1.0
                    tp_a += 1.0
                    loca_sum_a += float(iou_mat[r, c])

        tp_arr[a_idx] = tp_a
        fn_arr[a_idx] = float(total_gt) - tp_a
        fp_arr[a_idx] = float(total_pred) - tp_a
        loca_sum_arr[a_idx] = loca_sum_a

        # Pass 3: Association IoU / Re / Pr across all matched pairs
        nz_m = matches > 0.0
        if np.any(nz_m):
            m_vals = matches[nz_m]
            g_i, p_i = np.where(nz_m)
            g_tot = gt_cnt_arr[g_i]
            p_tot = pred_cnt_arr[p_i]
            ass_iou = m_vals / (g_tot + p_tot - m_vals)
            ass_re = m_vals / g_tot
            ass_pr = m_vals / p_tot
            sum_ass_iou_arr[a_idx] = float(np.sum(m_vals * ass_iou))
            sum_ass_re_arr[a_idx] = float(np.sum(m_vals * ass_re))
            sum_ass_pr_arr[a_idx] = float(np.sum(m_vals * ass_pr))

    return finalize_hota({
        'alphas': alphas.tolist(),
        'tp': tp_arr,
        'fn': fn_arr,
        'fp': fp_arr,
        'loca_sum': loca_sum_arr,
        'sum_ass_iou': sum_ass_iou_arr,
        'sum_ass_re': sum_ass_re_arr,
        'sum_ass_pr': sum_ass_pr_arr,
    })


def finalize_hota(acc: dict) -> dict:
    """Compute mean HOTA, DetA, AssA, LocA, AssRe, AssPr across all 19 alphas."""
    tp = np.asarray(acc['tp'], dtype=np.float64)
    fn = np.asarray(acc['fn'], dtype=np.float64)
    fp = np.asarray(acc['fp'], dtype=np.float64)
    loca_sum = np.asarray(acc['loca_sum'], dtype=np.float64)
    sum_ass_iou = np.asarray(acc['sum_ass_iou'], dtype=np.float64)
    sum_ass_re = np.asarray(acc['sum_ass_re'], dtype=np.float64)
    sum_ass_pr = np.asarray(acc['sum_ass_pr'], dtype=np.float64)

    det_denom = tp + fn + fp
    deta_a = np.where(det_denom > 0.0, tp / det_denom, 0.0)
    assa_a = np.where(tp > 0.0, sum_ass_iou / tp, 0.0)
    assre_a = np.where(tp > 0.0, sum_ass_re / tp, 0.0)
    asspr_a = np.where(tp > 0.0, sum_ass_pr / tp, 0.0)
    loca_a = np.where(tp > 0.0, loca_sum / tp, 0.0)
    hota_a = np.sqrt(deta_a * assa_a)

    return {
        'raw': {
            'alphas': acc['alphas'],
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'loca_sum': loca_sum,
            'sum_ass_iou': sum_ass_iou,
            'sum_ass_re': sum_ass_re,
            'sum_ass_pr': sum_ass_pr,
        },
        'hota': float(np.mean(hota_a)),
        'deta': float(np.mean(deta_a)),
        'assa': float(np.mean(assa_a)),
        'loca': float(np.mean(loca_a)),
        'assre': float(np.mean(assre_a)),
        'asspr': float(np.mean(asspr_a)),
        'hota_per_alpha': hota_a.tolist(),
        'deta_per_alpha': deta_a.tolist(),
        'assa_per_alpha': assa_a.tolist(),
    }


def evaluate_purity_and_kinematics(frame_data, iou_thresh: float = 0.50) -> dict:
    """
    Domain metrics matching DEVLOG.md:
      - Track Purity: fraction of a predicted track's matched points belonging to its dominant GT ID.
      - 100%-Pure Tracks: fraction of matched predicted tracks with zero ID switches.
      - Erratic Jumps (> 25 px): step displacements > 25 px between consecutive track observations.
      - Mean / Median step displacement (px/frame).
    """
    pred_gt_labels: Dict[int, List[int]] = {}
    pred_positions: Dict[int, List[Tuple[int, float, float]]] = {}

    for f_idx, gt_ids, _, pred_ids, pred_boxes, iou_mat in frame_data:
        # Record centers for kinematics
        for j, p_id in enumerate(pred_ids):
            x1, y1, x2, y2 = pred_boxes[j]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            pred_positions.setdefault(p_id, []).append((f_idx, cx, cy))

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            continue
        cost = np.where(iou_mat >= iou_thresh, 1.0 - iou_mat, 1e6)
        r_ind, c_ind = linear_sum_assignment(cost)
        for r, c in zip(r_ind, c_ind):
            if iou_mat[r, c] >= iou_thresh:
                pred_gt_labels.setdefault(pred_ids[c], []).append(gt_ids[r])

    purities = []
    pure_100 = 0
    for p_id, g_list in pred_gt_labels.items():
        if not g_list:
            continue
        vals, counts = np.unique(g_list, return_counts=True)
        purity = float(np.max(counts)) / len(g_list)
        purities.append(purity)
        if len(vals) == 1:
            pure_100 += 1

    steps = []
    jumps_25 = 0
    for p_id, pts in pred_positions.items():
        pts.sort(key=lambda x: x[0])
        for k in range(1, len(pts)):
            d = math.hypot(pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2])
            steps.append(d)
            if d > 25.0:
                jumps_25 += 1

    return {
        'avg_track_purity': float(np.mean(purities)) if purities else 0.0,
        'pure_100_count': pure_100,
        'matched_track_count': len(purities),
        'pure_100_ratio': (pure_100 / len(purities)) if purities else 0.0,
        'erratic_jumps_gt25': jumps_25,
        'mean_step_disp': float(np.mean(steps)) if steps else 0.0,
        'median_step_disp': float(np.median(steps)) if steps else 0.0,
        'purities_list': purities,
        'steps_list': steps,
    }


def evaluate_sequence(
    gt_frames: Dict[int, List[dict]],
    pred_frames: Dict[int, List[dict]],
    all_frames: List[int] = None,
    iou_thresh: float = 0.50,
) -> dict:
    """Run CLEAR MOT, IDF1, HOTA, and Purity/Kinematics on a single sequence."""
    frame_data = _prepare_frames(gt_frames, pred_frames, all_frames)
    clear = evaluate_clear_mot(frame_data, iou_thresh=iou_thresh)
    idf1 = evaluate_idf1(frame_data, iou_thresh=iou_thresh)
    hota = evaluate_hota(frame_data, alphas=HOTA_ALPHAS)
    kin = evaluate_purity_and_kinematics(frame_data, iou_thresh=iou_thresh)
    return {
        'clear': clear,
        'idf1': idf1,
        'hota': hota,
        'kinematics': kin,
    }


def aggregate_sequence_results(seq_results: List[dict]) -> dict:
    """
    Aggregate raw accumulators across multiple sequences (exact TrackEval dataset aggregation).
    """
    clear_acc = {
        'total_gt': 0, 'total_pred': 0, 'tp': 0, 'fp': 0, 'fn': 0,
        'idsw': 0, 'frag': 0, 'motp_sum': 0.0,
        'mt': 0, 'pt': 0, 'ml': 0, 'num_gt_ids': 0,
    }
    idf1_acc = {'idtp': 0, 'idfp': 0, 'idfn': 0, 'total_gt': 0, 'total_pred': 0}

    n_a = len(HOTA_ALPHAS)
    hota_acc = {
        'alphas': HOTA_ALPHAS.tolist(),
        'tp': np.zeros(n_a, dtype=np.float64),
        'fn': np.zeros(n_a, dtype=np.float64),
        'fp': np.zeros(n_a, dtype=np.float64),
        'loca_sum': np.zeros(n_a, dtype=np.float64),
        'sum_ass_iou': np.zeros(n_a, dtype=np.float64),
        'sum_ass_re': np.zeros(n_a, dtype=np.float64),
        'sum_ass_pr': np.zeros(n_a, dtype=np.float64),
    }

    all_purities = []
    all_steps = []
    pure_100_total = 0
    matched_tracks_total = 0
    jumps_25_total = 0

    for sr in seq_results:
        c = sr['clear']
        for k in clear_acc:
            clear_acc[k] += c[k]

        idf = sr['idf1']
        for k in idf1_acc:
            idf1_acc[k] += idf[k]

        hr = sr['hota']['raw']
        for k in ('tp', 'fn', 'fp', 'loca_sum', 'sum_ass_iou', 'sum_ass_re', 'sum_ass_pr'):
            hota_acc[k] += hr[k]

        kin = sr['kinematics']
        all_purities.extend(kin['purities_list'])
        all_steps.extend(kin['steps_list'])
        pure_100_total += kin['pure_100_count']
        matched_tracks_total += kin['matched_track_count']
        jumps_25_total += kin['erratic_jumps_gt25']

    return {
        'clear': finalize_clear_mot(clear_acc),
        'idf1': finalize_idf1(idf1_acc),
        'hota': finalize_hota(hota_acc),
        'kinematics': {
            'avg_track_purity': float(np.mean(all_purities)) if all_purities else 0.0,
            'pure_100_count': pure_100_total,
            'matched_track_count': matched_tracks_total,
            'pure_100_ratio': (pure_100_total / matched_tracks_total) if matched_tracks_total > 0 else 0.0,
            'erratic_jumps_gt25': jumps_25_total,
            'mean_step_disp': float(np.mean(all_steps)) if all_steps else 0.0,
            'median_step_disp': float(np.median(all_steps)) if all_steps else 0.0,
        },
    }


def run_unit_tests():
    """Verify CLEAR MOT, IDF1, and HOTA exact mathematical identities."""
    b1 = [10.0, 10.0, 30.0, 30.0]
    b2 = [100.0, 100.0, 120.0, 120.0]

    # Test 1: Perfect tracking (2 tracks x 10 frames)
    gt_1 = {t: [{'id': 1, 'box': b1}, {'id': 2, 'box': b2}] for t in range(1, 11)}
    pred_1 = {t: [{'id': 10, 'box': b1}, {'id': 20, 'box': b2}] for t in range(1, 11)}
    r1 = evaluate_sequence(gt_1, pred_1)
    assert math.isclose(r1['clear']['mota'], 1.0, abs_tol=1e-9)
    assert math.isclose(r1['idf1']['idf1'], 1.0, abs_tol=1e-9)
    assert math.isclose(r1['hota']['hota'], 1.0, abs_tol=1e-9)
    assert r1['clear']['idsw'] == 0

    # Test 2: Exact mid-sequence ID swap (frames 6..10 swap pred IDs)
    pred_2 = {}
    for t in range(1, 11):
        if t <= 5:
            pred_2[t] = [{'id': 10, 'box': b1}, {'id': 20, 'box': b2}]
        else:
            pred_2[t] = [{'id': 20, 'box': b1}, {'id': 10, 'box': b2}]
    r2 = evaluate_sequence(gt_1, pred_2)
    assert r2['clear']['idsw'] == 2, f"Expected 2 IDSW, got {r2['clear']['idsw']}"
    assert math.isclose(r2['clear']['mota'], 0.90, abs_tol=1e-9)
    assert math.isclose(r2['idf1']['idf1'], 0.50, abs_tol=1e-9)
    assert math.isclose(r2['hota']['deta'], 1.0, abs_tol=1e-9)
    assert math.isclose(r2['hota']['assa'], 1.0 / 3.0, abs_tol=1e-7)
    assert math.isclose(r2['hota']['hota'], math.sqrt(1.0 / 3.0), abs_tol=1e-7)

    # Test 3: 1-frame missed detection (FN) then same ID resumes -> IDSW=0, Frag=1, FN=1
    gt_3 = {t: [{'id': 1, 'box': b1}] for t in range(1, 6)}
    pred_3 = {t: [{'id': 1, 'box': b1}] for t in [1, 2, 4, 5]}
    r3 = evaluate_sequence(gt_3, pred_3, all_frames=[1, 2, 3, 4, 5])
    assert r3['clear']['idsw'] == 0
    assert r3['clear']['frag'] == 1
    assert r3['clear']['fn'] == 1
    assert math.isclose(r3['clear']['mota'], 0.80, abs_tol=1e-9)
    assert math.isclose(r3['idf1']['idtp'], 4, abs_tol=1e-9)
    assert math.isclose(r3['idf1']['idr'], 0.80, abs_tol=1e-9)
    assert math.isclose(r3['idf1']['idp'], 1.00, abs_tol=1e-9)

    print("All MOT metric unit tests passed (CLEAR MOT / MOTA, IDF1, HOTA, Frag, IDSW).")


if __name__ == '__main__':
    run_unit_tests()
