# ============================================================
#  NOTEBOOK 4: YOLOX Drone Swarm Fine-Tuning Pipeline
#  Run on Google Colab (Tesla T4 / A100 GPU)
# ============================================================

"""
Fine-tuning Megvii YOLOX-S on the UAVSwarm Drone Swarm Dataset.
Outputs:
  - best_ckpt.pth (Fine-tuned PyTorch checkpoint)
  - yolox_drone_finetuned.onnx (Deployed ONNX model for run3.py live inference)
"""

import os
import sys
import re
import subprocess
import shutil
from pathlib import Path

# ── Cell 1: Check GPU & Hardware Acceleration ─────────────────────────────────
print("=" * 60)
print("1. HARDWARE ACCELERATION CHECK")
print("=" * 60)
try:
    res = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(res.stdout)
except Exception as e:
    print(f"nvidia-smi not available: {e}")

import torch
print(f"PyTorch Version  : {torch.__version__}")
print(f"CUDA Available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device       : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: GPU not detected. Please enable T4/A100 in Runtime > Change runtime type!")

# ── Cell 2: Setup Workspace Directories ───────────────────────────────────────
WORK_DIR = Path("/content")
YOLOX_DIR = WORK_DIR / "YOLOX"
DATA_DIR = WORK_DIR / "UAVSwarm-dataset"
OUTPUT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cell 3: Clone & Install YOLOX ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. INSTALLING YOLOX & DEPENDENCIES")
print("=" * 60)
os.chdir(WORK_DIR)
if not YOLOX_DIR.exists():
    print("📥 Cloning Megvii YOLOX repository...")
    subprocess.run(["git", "clone", "https://github.com/Megvii-BaseDetection/YOLOX.git", str(YOLOX_DIR)], check=True)

# Patch requirements to remove onnx-simplifier build issue on modern Python
req_file = YOLOX_DIR / "requirements.txt"
if req_file.exists():
    text = req_file.read_text()
    req_file.write_text(re.sub(r"onnx-simplifier.*", "", text))

os.chdir(YOLOX_DIR)
subprocess.run(["pip", "install", "-q", "--no-build-isolation", "-e", "."], check=True)
subprocess.run(["pip", "install", "-q", "onnx", "onnxruntime", "pycocotools", "loguru", "thop"], check=True)
print("✅ YOLOX successfully installed in environment.")

# ── Cell 4: Download Pretrained COCO Base Weights ────────────────────────────
print("\n" + "=" * 60)
print("3. DOWNLOADING PRETRAINED BASE WEIGHTS")
print("=" * 60)
BASE_WEIGHTS = WORK_DIR / "yolox_s.pth"
if not BASE_WEIGHTS.exists():
    url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    print(f"📥 Downloading YOLOX-S COCO weights from {url}...")
    subprocess.run(["wget", "-q", "-O", str(BASE_WEIGHTS), url], check=True)
    print(f"✅ Downloaded base weights ({BASE_WEIGHTS.stat().st_size / 1e6:.1f} MB).")
else:
    print(f"✅ Base weights already present at: {BASE_WEIGHTS}")

# ── Cell 5: Clone Official UAVSwarm Dataset ──────────────────────────────────
print("\n" + "=" * 60)
print("4. PREPARING UAVSWARM DATASET")
print("=" * 60)
if not DATA_DIR.exists():
    print("📥 Cloning official UAVSwarm dataset repository...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/UAVSwarm/UAVSwarm-dataset.git", str(DATA_DIR)], check=True)
print(f"✅ Dataset ready at {DATA_DIR} (3,450 train images / 3,394 val images).")

# ── Cell 6: Create UAVSwarm Experiment Config (yolox_uavswarm.py) ──────────────
print("\n" + "=" * 60)
print("5. GENERATING YOLOX UAVSWARM EXPERIMENT SPECIFICATION")
print("=" * 60)
EXP_CODE = '''import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Drone Swarm Single-Class Configuration
        self.num_classes = 1
        self.depth = 0.33  # YOLOX-S depth multiplier
        self.width = 0.50  # YOLOX-S width multiplier
        self.exp_name = "yolox_uavswarm"

        # Dataset directories (COCO format)
        self.data_dir = "/content/UAVSwarm-dataset"
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.train_name = "train"
        self.val_name = "train"

        # Training Hyperparameters optimized for T4 GPU (16GB VRAM)
        self.max_epoch = 12
        self.data_num_workers = 4
        self.eval_interval = 2
        self.print_interval = 50
        self.save_history_ckpt = False
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # Learning Rate Schedule (Cosine Annealing)
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 1
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # Augmentation suited for tiny aerial drones
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "test.json",
            name=self.val_name if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
'''

exp_path = YOLOX_DIR / "exps" / "custom" / "yolox_uavswarm.py"
exp_path.parent.mkdir(parents=True, exist_ok=True)
with open(exp_path, "w") as f:
    f.write(EXP_CODE)
print(f"✅ Custom experiment specification written to: {exp_path}")

# ── Cell 7: Launch YOLOX Fine-Tuning ───────────────────────────────────────────
print("\n" + "=" * 60)
print("6. LAUNCHING YOLOX FINE-TUNING ON GPU")
print("=" * 60)
train_cmd = [
    sys.executable, "-m", "yolox.tools.train",
    "-f", str(exp_path),
    "-c", str(BASE_WEIGHTS),
    "-b", "16",
    "-d", "1",
    "--fp16",
    "-o"
]
print(f"Executing: {' '.join(train_cmd)}")
print("🚀 Training started. Check outputs in YOLOX_outputs/yolox_uavswarm/...")
subprocess.run(train_cmd, check=True)

# ── Cell 8: Export Best Checkpoint to ONNX ────────────────────────────────────
print("\n" + "=" * 60)
print("7. EXPORT FINE-TUNED MODEL TO ONNX")
print("=" * 60)
BEST_CKPT = YOLOX_DIR / "YOLOX_outputs" / "yolox_uavswarm" / "best_ckpt.pth"
EXPORT_ONNX = OUTPUT_DIR / "yolox_drone_finetuned.onnx"

export_cmd = [
    sys.executable, "-m", "yolox.tools.export_onnx",
    "--output-name", str(EXPORT_ONNX),
    "-f", str(exp_path),
    "-c", str(BEST_CKPT)
]
print(f"Export command: {' '.join(export_cmd)}")
subprocess.run(export_cmd, check=True)
print(f"✅ Fine-tuned ONNX model exported to: {EXPORT_ONNX}")
