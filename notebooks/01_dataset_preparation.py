# ============================================================
#  NOTEBOOK 1: Dataset Preparation — UAVSwarm → COCO + CSV
#  Run this on Google Colab / Kaggle before any training
# ============================================================
# Cell 1: Mount Drive & Clone Repo
# ----------------------------------
"""
# In Colab, run these cells sequentially.
# This notebook prepares:
#   1. UAVSwarm dataset in COCO format (for D-FINE)
#   2. Drone training CSV (for RL agent)
"""

# ── Cell 1: Mount Google Drive ────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os
BASE_DIR = '/content/drive/MyDrive/drone_tracker'
os.makedirs(BASE_DIR, exist_ok=True)
print(f"Base directory: {BASE_DIR}")

# ── Cell 2: Clone project repo ────────────────────────────────────────────────
os.chdir('/content')
if not os.path.exists('RM_Shrimp_extension'):
    # Option A: Clone from GitHub (if public)
    # !git clone https://github.com/SamratRay2005/RM_Shrimp_extension
    
    # Option B: Copy from Drive (if uploaded as zip)
    import shutil
    shutil.copy(f'{BASE_DIR}/RM_Shrimp_extension.zip', '/content/')
    import zipfile
    with zipfile.ZipFile('/content/RM_Shrimp_extension.zip') as z:
        z.extractall('/content/')

print("✅ Project files ready")
os.listdir('/content/RM_Shrimp_extension')

# ── Cell 3: Install dependencies ─────────────────────────────────────────────
import subprocess
subprocess.run(['pip', 'install', '-q', 'gdown', 'Pillow', 'pycocotools',
                'wandb', 'pandas', 'numpy'], check=True)
print("✅ Dependencies installed")

# ── Cell 4: Download UAVSwarm dataset ─────────────────────────────────────────
"""
The UAVSwarm dataset is available at:
  https://github.com/UAVSwarm/UAVSwarm-dataset

Download the sequences manually from the GitHub releases or Google Drive links
in the README, then upload the zip to your Google Drive and set DATASET_ZIP below.

Expected structure after extraction:
  sequences/
    seq01/
      img1/         ← JPEG frames
      gt/gt.txt     ← MOT16 ground truth
    seq02/ ...
"""

DATASET_ZIP = f'{BASE_DIR}/UAVSwarm_sequences.zip'    # ← Set this path!
SEQUENCES_DIR = f'{BASE_DIR}/data/uavswarm/sequences'

if os.path.exists(DATASET_ZIP):
    import zipfile
    os.makedirs(SEQUENCES_DIR, exist_ok=True)
    print(f"Extracting {DATASET_ZIP}...")
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(os.path.dirname(SEQUENCES_DIR))
    print(f"✅ Dataset extracted to {SEQUENCES_DIR}")
else:
    print(f"⚠️  Dataset zip not found at {DATASET_ZIP}")
    print("   Please upload UAVSwarm_sequences.zip to your Drive and update the path.")

# ── Cell 5: Convert MOT16 → COCO ─────────────────────────────────────────────
COCO_DIR = f'{BASE_DIR}/data/uavswarm/coco'

if os.path.exists(SEQUENCES_DIR):
    os.chdir('/content/RM_Shrimp_extension/dataset')
    result = subprocess.run([
        'python', 'mot2coco.py',
        '--seq_dir', SEQUENCES_DIR,
        '--output_dir', COCO_DIR,
        '--train_ratio', '0.8',
        '--val_ratio', '0.1',
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
    else:
        print("✅ COCO dataset ready!")
        # Show structure
        for root, dirs, files in os.walk(COCO_DIR):
            level = root.replace(COCO_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            if level < 2:
                for f in files[:3]:
                    print(f'{indent}  {f}')

# ── Cell 6: Convert MOT16 → Drone Training CSV ───────────────────────────────
CSV_PATH = f'{BASE_DIR}/data/drone_training_data.csv'

if os.path.exists(SEQUENCES_DIR):
    os.chdir('/content/RM_Shrimp_extension/shrimp_drone')
    result = subprocess.run([
        'python', 'mot_to_csv.py',
        '--seq_dir', SEQUENCES_DIR,
        '--output', CSV_PATH,
        '--min_length', '20',
        '--max_gap', '2',
        '--max_jump', '400',
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
    else:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Training CSV ready: {len(df)} rows, {df['track_id'].nunique()} tracks")
        print(df.head())

# ── Cell 7: Dataset Statistics ────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(CSV_PATH)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('UAVSwarm Drone Training Data Statistics', fontsize=14)

# Track length distribution
track_lengths = df.groupby('track_id').size()
axes[0].hist(track_lengths, bins=20, color='steelblue', edgecolor='white')
axes[0].set_title('Track Length Distribution')
axes[0].set_xlabel('Frames per Track')
axes[0].set_ylabel('Count')

# Drone position heatmap
axes[1].hist2d(df['cx'], df['cy'], bins=50, cmap='hot')
axes[1].set_title('Drone Position Heatmap')
axes[1].set_xlabel('cx (pixels)')
axes[1].set_ylabel('cy (pixels)')
axes[1].invert_yaxis()

# Track displacement (speed proxy)
def track_speeds(group):
    dists = np.sqrt(np.diff(group['cx'])**2 + np.diff(group['cy'])**2)
    return dists.mean() if len(dists) > 0 else 0

speeds = df.groupby('track_id').apply(track_speeds)
axes[2].hist(speeds, bins=20, color='tomato', edgecolor='white')
axes[2].set_title('Avg Inter-frame Displacement')
axes[2].set_xlabel('pixels/frame')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/dataset_stats.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Stats saved to {BASE_DIR}/dataset_stats.png")
print(f"\nSummary:")
print(f"  Total tracks:     {df['track_id'].nunique()}")
print(f"  Total rows:       {len(df)}")
print(f"  Avg track length: {track_lengths.mean():.1f} frames")
print(f"  Avg drone speed:  {speeds.mean():.1f} px/frame")
