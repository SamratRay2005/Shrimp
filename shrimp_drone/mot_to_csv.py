"""
MOT16 GT → Drone Training CSV Converter
=========================================
WHY THIS SCRIPT EXISTS:
  The Shrimp architecture trains from a CSV with columns:
    frame, track_id, cx, cy, x1, y1, x2, y2, score

  The UAVSwarm dataset provides annotations in MOT16 gt.txt format:
    frame_id, track_id, bb_left, bb_top, width, height, conf, class, visibility

  This script converts MOT16 → Shrimp-compatible CSV so the drone training
  pipeline (DroneTrackingEnv + DroneAgent) can be run directly on UAVSwarm data.

  clean_data.py logic is also re-implemented here with drone-appropriate thresholds:
    - Min length:  20 frames (drones appear/disappear faster than shrimp)
    - Max gap:     2 frames (allow 1-frame occlusion, reject long gaps)
    - Max jump:    400px  (drones move faster than shrimp, larger allowed)

USAGE:
  python mot_to_csv.py --seq_dir ./data/uavswarm/sequences \
                       --output ./drone_training_data.csv
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_sequence(seq_dir: Path, seq_name: str) -> pd.DataFrame:
    """Parse one MOT16 sequence gt.txt into a dataframe of detections."""
    gt_file = seq_dir / seq_name / 'gt' / 'gt.txt'
    img_dir  = seq_dir / seq_name / 'img1'

    if not gt_file.exists():
        print(f"  [SKIP] {seq_name}: no gt.txt")
        return pd.DataFrame()

    rows = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue

            frame_id  = int(parts[0])
            track_id  = int(parts[1])
            bb_left   = float(parts[2])
            bb_top    = float(parts[3])
            width     = float(parts[4])
            height    = float(parts[5])
            conf      = int(float(parts[6]))

            if conf == 0:   # Ignore flag in MOT16
                continue

            cx = bb_left + width  / 2.0
            cy = bb_top  + height / 2.0
            x1 = bb_left
            y1 = bb_top
            x2 = bb_left + width
            y2 = bb_top  + height

            # Build a frame identifier similar to the shrimp CSV format
            frame_name = f"{seq_name}_frame_{frame_id:06d}.jpg"

            rows.append({
                'frame':    frame_name,
                'track_id': f"{seq_name}_t{track_id}",  # unique across sequences
                'cx': cx, 'cy': cy,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'score': 1.0,   # GT annotations always have score=1
                'frame_id': frame_id,
                'seq': seq_name,
            })

    return pd.DataFrame(rows)


def clean_tracks(df: pd.DataFrame, min_length: int = 20,
                 max_gap: int = 2, max_jump: float = 400.0) -> pd.DataFrame:
    """
    Filter to 'perfect' trajectories suitable for RL training.

    Drone thresholds (vs Shrimp):
      min_length: 20 frames  (shrimp: 16 — drones appear/disappear faster)
      max_gap:    2 frames   (shrimp: 0  — allow 1-frame occlusion)
      max_jump:   400px      (shrimp: 200px — drones move faster)

    WHY STRICT CLEANING:
      The RL agent trains on trajectory continuity — if there are frame gaps
      or teleportations in the training data, the velocity feature (vx, vy)
      becomes noise, corrupting the learning signal.
    """
    valid_ids = []

    for tid, group in df.groupby('track_id'):
        group = group.sort_values('frame_id')
        frame_ids = group['frame_id'].values
        cxs = group['cx'].values
        cys = group['cy'].values

        # Condition A: Minimum length
        if len(group) < min_length:
            continue

        # Condition B: No large gaps (allow up to max_gap consecutive missing frames)
        gaps = np.diff(frame_ids)
        if np.any(gaps > max_gap):
            continue

        # Condition C: No teleportation
        dists = np.sqrt(np.diff(cxs)**2 + np.diff(cys)**2)
        if len(dists) > 0 and np.max(dists) > max_jump:
            continue

        valid_ids.append(tid)

    clean = df[df['track_id'].isin(valid_ids)].copy()
    # Drop internal columns not needed for training
    clean = clean.drop(columns=['frame_id', 'seq'], errors='ignore')
    return clean


def mot_to_csv(seq_dir: str, output_csv: str,
               min_length: int = 20, max_gap: int = 2, max_jump: float = 400.0):
    """Convert all sequences in seq_dir to a single training CSV."""
    seq_base = Path(seq_dir)
    sequences = sorted([d.name for d in seq_base.iterdir() if d.is_dir()])

    if not sequences:
        raise ValueError(f"No sequence directories found in {seq_dir}")

    print(f"[mot_to_csv] Processing {len(sequences)} sequences...")
    all_dfs = []

    for seq_name in sequences:
        df = parse_sequence(seq_base, seq_name)
        if df.empty:
            continue
        print(f"  {seq_name}: {df['track_id'].nunique()} raw tracks, "
              f"{len(df)} rows")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data parsed from any sequence!")

    raw_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[mot_to_csv] Raw: {raw_df['track_id'].nunique()} tracks, "
          f"{len(raw_df)} rows")

    # Clean tracks
    clean_df = clean_tracks(raw_df, min_length=min_length,
                             max_gap=max_gap, max_jump=max_jump)
    print(f"[mot_to_csv] Clean: {clean_df['track_id'].nunique()} tracks, "
          f"{len(clean_df)} rows")

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    print(f"[mot_to_csv] Saved: {output_csv}")

    return clean_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOT16 GT → Drone Training CSV')
    parser.add_argument('--seq_dir',    required=True, help='Path to sequences/')
    parser.add_argument('--output',     required=True, help='Output CSV path')
    parser.add_argument('--min_length', type=int,   default=20)
    parser.add_argument('--max_gap',    type=int,   default=2)
    parser.add_argument('--max_jump',   type=float, default=400.0)
    args = parser.parse_args()

    mot_to_csv(args.seq_dir, args.output,
               args.min_length, args.max_gap, args.max_jump)
