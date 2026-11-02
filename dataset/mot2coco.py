"""
MOT16 → COCO Converter for UAVSwarm Dataset
=============================================
WHY: D-FINE requires COCO JSON format. UAVSwarm uses MOT16 (gt.txt per sequence).
This script bridges that gap so the same D-FINE training infrastructure works
out of the box without any modification to the D-FINE codebase itself.

MOT16 format per line:
  frame_id, track_id, bb_left, bb_top, width, height, conf, class, visibility

COCO JSON output structure:
  { "images": [...], "annotations": [...], "categories": [...] }

USAGE:
  python mot2coco.py --seq_dir ./data/uavswarm/sequences \
                     --output_dir ./data/uavswarm/coco \
                     --train_ratio 0.8 --val_ratio 0.1
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image


CATEGORY = [{"id": 1, "name": "drone", "supercategory": "vehicle"}]


def parse_mot_gt(gt_file: Path):
    """
    Parse a MOT16 gt.txt file into a list of annotation dicts.
    Each line: frame_id, track_id, bb_left, bb_top, width, height, conf, class, visibility
    We keep only conf > 0 and class == 1 (active targets, ignore distractor classes).
    """
    annotations = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bb_left  = float(parts[2])
            bb_top   = float(parts[3])
            width    = float(parts[4])
            height   = float(parts[5])
            conf     = int(float(parts[6]))
            cls      = int(float(parts[7]))

            # Skip distractors/occluded (conf=0 means ignore in MOT16)
            if conf == 0:
                continue
            # Class 1 = pedestrian in standard MOT, but in UAVSwarm all are drones
            # Keep all non-ignored annotations
            annotations.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'bbox': [bb_left, bb_top, width, height],  # COCO [x, y, w, h]
                'area': width * height,
            })
    return annotations


def get_image_size(img_path: Path):
    """Return (width, height) of an image without fully loading it."""
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def build_coco_from_sequences(sequences: list, seq_base_dir: Path,
                               output_img_dir: Path, output_img_dir_name: str,
                               start_img_id: int = 1, start_ann_id: int = 1):
    """
    Convert a list of MOT16 sequences into COCO format.
    Copies images to output_img_dir and builds annotation lists.

    Returns: (images_list, annotations_list, next_img_id, next_ann_id)
    """
    images = []
    annotations = []
    img_id = start_img_id
    ann_id = start_ann_id

    for seq_name in sorted(sequences):
        seq_dir = seq_base_dir / seq_name
        img_dir = seq_dir / 'img1'
        gt_file = seq_dir / 'gt' / 'gt.txt'

        if not img_dir.exists() or not gt_file.exists():
            print(f"  [SKIP] {seq_name}: missing img1/ or gt/gt.txt")
            continue

        # Parse ground truth
        raw_anns = parse_mot_gt(gt_file)

        # Build frame → annotations mapping
        frame_to_anns = {}
        for ann in raw_anns:
            fid = ann['frame_id']
            frame_to_anns.setdefault(fid, []).append(ann)

        # Find all image files (sorted numerically)
        img_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))

        for img_path in img_files:
            # Extract frame number from filename (e.g., 000042.jpg → 42)
            stem = img_path.stem
            try:
                frame_num = int(stem)
            except ValueError:
                # Handle non-numeric stems like UAVSwarm (frame_NNNN.jpg)
                import re
                m = re.search(r'(\d+)', stem)
                frame_num = int(m.group(1)) if m else None
                if frame_num is None:
                    continue

            # Get or infer image dimensions
            try:
                w, h = get_image_size(img_path)
            except Exception:
                w, h = 1920, 1080  # fallback

            # Build unique filename: seqname_NNNNNN.jpg
            out_fname = f"{seq_name}_{stem}{img_path.suffix}"
            dst_path = output_img_dir / out_fname

            # Copy image
            if not dst_path.exists():
                shutil.copy2(img_path, dst_path)

            images.append({
                'id': img_id,
                'file_name': out_fname,
                'width': w,
                'height': h,
                'seq_name': seq_name,
                'frame_id': frame_num,
            })

            # Add annotations for this frame
            if frame_num in frame_to_anns:
                for ann in frame_to_anns[frame_num]:
                    x, y, bw, bh = ann['bbox']
                    # Clamp to image bounds
                    x  = max(0.0, x)
                    y  = max(0.0, y)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    if bw <= 0 or bh <= 0:
                        continue
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': [x, y, bw, bh],
                        'area': bw * bh,
                        'iscrowd': 0,
                        'track_id': ann['track_id'],
                    })
                    ann_id += 1

            img_id += 1

        print(f"  ✓ {seq_name}: {len(img_files)} frames processed")

    return images, annotations, img_id, ann_id


def mot2coco(seq_dir: str, output_dir: str,
             train_ratio: float = 0.8, val_ratio: float = 0.1,
             seed: int = 42):
    """
    Main conversion entry point.

    Splits sequences (not frames!) into train/val/test to avoid data leakage.
    WHY sequences not frames: frames from the same sequence are temporally
    correlated. Splitting at sequence level gives a more honest val/test set.
    """
    seq_base = Path(seq_dir)
    out_base = Path(output_dir)

    # Discover all sequences
    sequences = sorted([d.name for d in seq_base.iterdir() if d.is_dir()])
    if not sequences:
        raise ValueError(f"No sequence directories found in {seq_dir}")

    print(f"[INFO] Found {len(sequences)} sequences: {sequences}")

    # Split sequences into train/val/test
    random.seed(seed)
    random.shuffle(sequences)

    n_train = max(1, int(len(sequences) * train_ratio))
    n_val   = max(1, int(len(sequences) * val_ratio))

    train_seqs = sequences[:n_train]
    val_seqs   = sequences[n_train:n_train + n_val]
    test_seqs  = sequences[n_train + n_val:]

    print(f"  Train sequences ({len(train_seqs)}): {train_seqs}")
    print(f"  Val   sequences ({len(val_seqs)}):   {val_seqs}")
    print(f"  Test  sequences ({len(test_seqs)}):  {test_seqs}")

    splits = {
        'train2017': train_seqs,
        'val2017':   val_seqs,
        'test2017':  test_seqs,
    }

    ann_dir = out_base / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)

    img_id, ann_id = 1, 1

    for split_name, seqs in splits.items():
        if not seqs:
            print(f"[SKIP] No sequences for split '{split_name}'")
            continue

        print(f"\n[BUILDING] {split_name} ({len(seqs)} sequences)...")
        img_out_dir = out_base / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)

        images, anns, img_id, ann_id = build_coco_from_sequences(
            seqs, seq_base, img_out_dir, split_name,
            start_img_id=img_id, start_ann_id=ann_id
        )

        coco_dict = {
            'info': {
                'description': f'UAVSwarm Dataset - {split_name}',
                'url': 'https://github.com/UAVSwarm/UAVSwarm-dataset',
                'version': '1.0',
                'year': 2023,
                'contributor': 'UAVSwarm',
            },
            'licenses': [],
            'images': images,
            'annotations': anns,
            'categories': CATEGORY,
        }

        out_json = ann_dir / f'instances_{split_name}.json'
        with open(out_json, 'w') as f:
            json.dump(coco_dict, f)

        print(f"  → {len(images)} images, {len(anns)} annotations")
        print(f"  → Saved: {out_json}")

    print(f"\n[DONE] COCO dataset written to: {out_base}")
    print(f"  Structure:")
    print(f"    {out_base}/")
    print(f"      annotations/instances_train2017.json")
    print(f"      annotations/instances_val2017.json")
    print(f"      train2017/  (images)")
    print(f"      val2017/    (images)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert UAVSwarm MOT16 → COCO')
    parser.add_argument('--seq_dir',    required=True,  help='Path to sequences/ folder')
    parser.add_argument('--output_dir', required=True,  help='Where to write COCO dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    mot2coco(
        seq_dir=args.seq_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
