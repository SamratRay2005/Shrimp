# ============================================================
#  NOTEBOOK 2: D-FINE Fine-tuning on UAVSwarm Dataset
#  Run on Google Colab (T4/A100) or Kaggle (T4/P100)
# ============================================================

# ── Cell 1: Check GPU ─────────────────────────────────────────────────────────
import subprocess, os

result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Cell 2: Mount Drive ───────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR  = '/content/drive/MyDrive/drone_tracker'
COCO_DIR  = f'{BASE_DIR}/data/uavswarm/coco'
OUT_DIR   = f'{BASE_DIR}/results/dfine_uavswarm'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Cell 3: Clone D-FINE ──────────────────────────────────────────────────────
os.chdir('/content')
if not os.path.exists('D-FINE'):
    subprocess.run(['git', 'clone', 'https://github.com/Peterande/D-FINE.git'], check=True)

os.chdir('/content/D-FINE')
subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
print("✅ D-FINE cloned and dependencies installed")

# ── Cell 4: Download Pretrained COCO Weights ─────────────────────────────────
"""
Download pretrained D-FINE weights from the official model zoo.
Using dfine_m_coco.pth (medium size) — best for T4 16GB VRAM.

If you have an A100 (40GB), use dfine_l_coco.pth for better accuracy.
"""
import subprocess

WEIGHTS_URL = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth'
WEIGHTS_PATH = f'{BASE_DIR}/dfine_m_coco.pth'

if not os.path.exists(WEIGHTS_PATH):
    print(f"Downloading pretrained weights...")
    subprocess.run(['wget', '-q', '-O', WEIGHTS_PATH, WEIGHTS_URL], check=True)
    print(f"✅ Weights saved to {WEIGHTS_PATH}")
else:
    print(f"✅ Weights already at {WEIGHTS_PATH}")

# Verify size
size_mb = os.path.getsize(WEIGHTS_PATH) / 1e6
print(f"   File size: {size_mb:.1f} MB")

# ── Cell 5: Set up D-FINE config ──────────────────────────────────────────────
"""
Copy our custom UAVSwarm config to D-FINE's configs directory.
"""
import shutil

# Copy our config
CUSTOM_CONFIG_SRC = '/content/RM_Shrimp_extension/dfine_finetune/configs/dfine_uav_swarm.yml'
CUSTOM_CONFIG_DST = '/content/D-FINE/configs/dfine/dfine_uav_swarm.yml'

# We need to create a self-contained config (no __include__ to our relative paths)
CONFIG_CONTENT = f"""
__include__: [
  './include/dataloader.yml',
  './include/optimizer.yml',
  './include/runtime.yml',
  './dfine_hgnetv2_m.yml',
]

output_dir: {OUT_DIR}

model:
  decoder:
    num_classes: 1

train_dataloader:
  dataset:
    img_folder: {COCO_DIR}/train2017/
    ann_file: {COCO_DIR}/annotations/instances_train2017.json
    return_masks: False
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.8}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: RandomHorizontalFlip}}
        - {{type: RandomVerticalFlip, p: 0.3}}
        - type: RandomResize
          sizes: [640, 672, 704, 736, 768, 800]
          max_size: 1024
        - {{type: ToImageTensor}}
        - {{type: ConvertDtype}}
        - type: SanitizeBoundingBoxes
          min_size: 1
  batch_size: 6
  shuffle: True
  num_workers: 2
  drop_last: True

val_dataloader:
  dataset:
    img_folder: {COCO_DIR}/val2017/
    ann_file: {COCO_DIR}/annotations/instances_val2017.json
    return_masks: False
    transforms:
      ops:
        - {{type: Resize, size: [800, 800], max_size: 1024}}
        - {{type: ToImageTensor}}
        - {{type: ConvertDtype}}
  batch_size: 4
  shuffle: False
  num_workers: 2

solver:
  max_epoch: 72
  val_epoch: 2
  warmup_iter: 500
  warmup_bias_lr: 0.0001
  optimizer:
    type: AdamW
    params:
      - params: '^(?=.*backbone)(?=.*norm).*$'
        lr: 1.0e-05
        weight_decay: 0.00
      - params: '^(?=.*backbone).*$'
        lr: 1.0e-05
      - params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
        weight_decay: 0.00
    lr: 5.0e-05
    betas: [0.9, 0.999]
    weight_decay: 1.0e-04
  lr_scheduler:
    type: MultiStepLR
    milestones: [55, 68]
    gamma: 0.1
  clip_max_norm: 0.1

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

checkpoint_freq: 1
"""

with open(CUSTOM_CONFIG_DST, 'w') as f:
    f.write(CONFIG_CONTENT)

print(f"✅ Config written to {CUSTOM_CONFIG_DST}")

# ── Cell 6: (Optional) Test config + data loading ─────────────────────────────
"""
Run a quick sanity check to ensure data loading works before full training.
"""
test_result = subprocess.run([
    'python', 'train.py',
    '-c', CUSTOM_CONFIG_DST,
    '-r', WEIGHTS_PATH,
    '--test-only',
], capture_output=True, text=True, cwd='/content/D-FINE')

print("STDOUT:", test_result.stdout[-3000:])
if test_result.returncode != 0:
    print("STDERR:", test_result.stderr[-2000:])
else:
    print("✅ Data loading and config OK!")

# ── Cell 7: Launch Fine-tuning ────────────────────────────────────────────────
"""
Start fine-tuning. This will run for ~72 epochs.
On a T4 GPU: ~2-4 hours depending on dataset size.
On an A100:  ~45-90 minutes.

Checkpoints save every epoch to OUT_DIR.
"""
print("🚀 Starting D-FINE fine-tuning...")
print(f"   Config:  {CUSTOM_CONFIG_DST}")
print(f"   Weights: {WEIGHTS_PATH}")
print(f"   Output:  {OUT_DIR}")
print()

train_result = subprocess.Popen([
    'torchrun',
    '--nproc_per_node=1',
    'train.py',
    '-c', CUSTOM_CONFIG_DST,
    '-r', WEIGHTS_PATH,
], cwd='/content/D-FINE',
stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
text=True, bufsize=1)

# Stream output in real-time
for line in train_result.stdout:
    print(line, end='')

train_result.wait()
print(f"\nExit code: {train_result.returncode}")
if train_result.returncode == 0:
    print("✅ Fine-tuning complete!")
else:
    print("❌ Training failed — check output above")

# ── Cell 8: Evaluate Best Checkpoint ─────────────────────────────────────────
import glob

# Find best checkpoint
ckpt_files = sorted(glob.glob(f'{OUT_DIR}/*.pth'))
if ckpt_files:
    best_ckpt = ckpt_files[-1]
    print(f"Evaluating: {best_ckpt}")

    eval_result = subprocess.run([
        'torchrun', '--nproc_per_node=1', 'train.py',
        '-c', CUSTOM_CONFIG_DST,
        '-r', best_ckpt,
        '--test-only',
    ], capture_output=True, text=True, cwd='/content/D-FINE')

    print(eval_result.stdout)
else:
    print("No checkpoints found!")

# ── Cell 9: Export to ONNX ────────────────────────────────────────────────────
"""
Export the fine-tuned model to ONNX for:
  1. Edge deployment (NVIDIA Jetson)
  2. Use with the RL tracker inference pipeline (run_drone_tracker.py)
"""
ONNX_OUT = f'{BASE_DIR}/results/dfine_uavswarm.onnx'

if ckpt_files:
    export_result = subprocess.run([
        'python', 'tools/deployment/export_onnx.py',
        '-c', CUSTOM_CONFIG_DST,
        '-r', best_ckpt,
        '--output-file', ONNX_OUT,
    ], capture_output=True, text=True, cwd='/content/D-FINE')

    print(export_result.stdout)
    if export_result.returncode == 0:
        size_mb = os.path.getsize(ONNX_OUT) / 1e6
        print(f"✅ ONNX model exported: {ONNX_OUT} ({size_mb:.1f} MB)")
    else:
        print("Export error:", export_result.stderr)
