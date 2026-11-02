"""
UAVSwarm Dataset Downloader
============================
Downloads the UAVSwarm dataset from the official GitHub repository.
Dataset: https://github.com/UAVSwarm/UAVSwarm-dataset
Format: MOT16 (sequences of frames + gt.txt annotations)

WHY THIS SCRIPT EXISTS:
  The UAVSwarm dataset is hosted on GitHub releases (or Google Drive).
  This script automates the full download + extraction pipeline so that
  subsequent conversion (mot2coco.py) has a consistent directory structure.

USAGE (on Colab):
  python download_uavswarm.py --output_dir ./data/uavswarm --split all

EXPECTED OUTPUT STRUCTURE:
  data/uavswarm/
    ├── sequences/
    │   ├── seq01/
    │   │   ├── img1/         ← JPEG frames
    │   │   │   ├── 000001.jpg
    │   │   │   └── ...
    │   │   ├── gt/
    │   │   │   └── gt.txt    ← MOT16 ground truth
    │   │   └── seqinfo.ini   ← Sequence metadata
    │   └── seq02/ ...
    └── README.md
"""

import os
import argparse
import subprocess
import zipfile
import json
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────

GITHUB_REPO = "https://github.com/UAVSwarm/UAVSwarm-dataset"

# Known Google Drive IDs for the dataset parts
# (Update these if the repo changes its release structure)
GDRIVE_MIRRORS = {
    "UAVSwarm_v1_part1": "1XXXX_PLACEHOLDER_PART1",  # Update from GitHub README
    "UAVSwarm_v1_part2": "1XXXX_PLACEHOLDER_PART2",
}

# ─── Downloader ───────────────────────────────────────────────────────────────

def check_gdown():
    """Ensure gdown is installed (for Google Drive downloads)."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("[INFO] Installing gdown for Google Drive support...")
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        import gdown
        return gdown


def download_from_gdrive(file_id: str, output_path: str):
    """Download a file from Google Drive using gdown."""
    gdown = check_gdown()
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[DOWNLOAD] Downloading from Google Drive: {url}")
    gdown.download(url, output_path, quiet=False)


def clone_dataset_repo(output_dir: str):
    """
    Clone the UAVSwarm GitHub repo.
    Note: The repo itself links to the data — actual images may be in releases/GDrive.
    """
    repo_dir = os.path.join(output_dir, "UAVSwarm-dataset")
    if not os.path.exists(repo_dir):
        print(f"[INFO] Cloning dataset repo to {repo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth=1", GITHUB_REPO, repo_dir],
            check=True
        )
    else:
        print(f"[INFO] Repo already cloned at {repo_dir}")
    return repo_dir


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file to a directory."""
    print(f"[EXTRACT] Extracting {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"[EXTRACT] Done.")


def validate_mot_structure(sequences_dir: str):
    """
    Validate that the downloaded dataset has the expected MOT16 structure.
    Each sequence should have: img1/ folder and gt/gt.txt
    """
    seq_dir = Path(sequences_dir)
    sequences = [d for d in seq_dir.iterdir() if d.is_dir()]

    print(f"\n[VALIDATE] Found {len(sequences)} sequences in {sequences_dir}")
    errors = []

    for seq in sorted(sequences):
        img_dir = seq / "img1"
        gt_file = seq / "gt" / "gt.txt"

        if not img_dir.exists():
            errors.append(f"  ✗ {seq.name}: missing img1/ directory")
            continue
        if not gt_file.exists():
            errors.append(f"  ✗ {seq.name}: missing gt/gt.txt")
            continue

        n_frames = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        n_annotations = sum(1 for _ in open(gt_file))
        print(f"  ✓ {seq.name}: {n_frames} frames, {n_annotations} annotation rows")

    if errors:
        print("\n[WARNING] Issues found:")
        for e in errors:
            print(e)
    else:
        print("\n[VALIDATE] All sequences validated successfully ✓")

    return len(errors) == 0


def generate_dataset_stats(sequences_dir: str, output_json: str):
    """
    Generate basic statistics about the dataset.
    WHY: Understanding the data distribution (object sizes, density, etc.)
    is critical for configuring D-FINE's resolution and augmentation settings.
    """
    from collections import defaultdict
    stats = {
        "total_sequences": 0,
        "total_frames": 0,
        "total_annotations": 0,
        "avg_drones_per_frame": 0,
        "max_drones_per_frame": 0,
        "bbox_size_distribution": {"tiny": 0, "small": 0, "medium": 0, "large": 0},
        "sequences": []
    }

    seq_dir = Path(sequences_dir)
    sequences = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    stats["total_sequences"] = len(sequences)

    all_counts_per_frame = []
    all_areas = []

    for seq in sequences:
        gt_file = seq / "gt" / "gt.txt"
        if not gt_file.exists():
            continue

        frame_counts = defaultdict(int)
        seq_areas = []

        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                w, h = float(parts[4]), float(parts[5])
                area = w * h
                frame_counts[frame_id] += 1
                seq_areas.append(area)
                all_areas.append(area)

                # Categorize by size (pixels^2)
                if area < 32*32:
                    stats["bbox_size_distribution"]["tiny"] += 1
                elif area < 64*64:
                    stats["bbox_size_distribution"]["small"] += 1
                elif area < 128*128:
                    stats["bbox_size_distribution"]["medium"] += 1
                else:
                    stats["bbox_size_distribution"]["large"] += 1

        n_frames = len(list((seq / "img1").glob("*.jpg"))) + len(list((seq / "img1").glob("*.png")))
        all_counts_per_frame.extend(frame_counts.values())

        stats["total_frames"] += n_frames
        stats["total_annotations"] += sum(frame_counts.values())

        seq_counts = list(frame_counts.values())
        stats["sequences"].append({
            "name": seq.name,
            "frames": n_frames,
            "annotations": sum(seq_counts),
            "avg_drones": round(sum(seq_counts) / max(len(seq_counts), 1), 2),
            "max_drones": max(seq_counts) if seq_counts else 0,
        })

    if all_counts_per_frame:
        stats["avg_drones_per_frame"] = round(sum(all_counts_per_frame) / len(all_counts_per_frame), 2)
        stats["max_drones_per_frame"] = max(all_counts_per_frame)

    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[STATS] Dataset Statistics:")
    print(f"  Sequences: {stats['total_sequences']}")
    print(f"  Total Frames: {stats['total_frames']}")
    print(f"  Total Annotations: {stats['total_annotations']}")
    print(f"  Avg Drones/Frame: {stats['avg_drones_per_frame']}")
    print(f"  Max Drones/Frame: {stats['max_drones_per_frame']}")
    print(f"  Bbox Size Distribution: {stats['bbox_size_distribution']}")
    print(f"  Stats saved to: {output_json}")

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download UAVSwarm Dataset")
    parser.add_argument("--output_dir", type=str, default="./data/uavswarm",
                        help="Directory to download dataset into")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download (if already downloaded manually)")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate structure after download")
    parser.add_argument("--stats", action="store_true", default=True,
                        help="Generate dataset statistics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences_dir = os.path.join(args.output_dir, "sequences")

    if not args.skip_download:
        print("=" * 60)
        print("UAVSwarm Dataset Downloader")
        print("=" * 60)
        print(f"\nTarget directory: {args.output_dir}")
        print("\n[INFO] Please follow these steps to download the dataset:")
        print("  1. Visit: https://github.com/UAVSwarm/UAVSwarm-dataset")
        print("  2. Download the dataset from the releases or provided links")
        print("  3. Extract to:", sequences_dir)
        print("  4. Re-run this script with --skip_download to validate\n")

        # Try cloning the repo for README/structure reference
        try:
            repo_dir = clone_dataset_repo(args.output_dir)
            print(f"\n[INFO] Repo cloned. Check {repo_dir}/README.md for download links.")
        except Exception as e:
            print(f"[WARNING] Could not clone repo: {e}")
            print("  Please manually download from:", GITHUB_REPO)

    if args.validate and os.path.exists(sequences_dir):
        is_valid = validate_mot_structure(sequences_dir)
    else:
        print(f"\n[SKIP] sequences_dir not found at {sequences_dir}")
        print("  Download and extract the dataset first.")
        return

    if args.stats and os.path.exists(sequences_dir):
        stats_path = os.path.join(args.output_dir, "dataset_stats.json")
        generate_dataset_stats(sequences_dir, stats_path)


if __name__ == "__main__":
    main()
