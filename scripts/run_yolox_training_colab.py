import os
import sys
import subprocess
from pathlib import Path

# 1. Create custom experiment config
exp_dir = Path("/content/YOLOX/exps/custom")
exp_dir.mkdir(parents=True, exist_ok=True)
exp_file = exp_dir / "yolox_uavswarm.py"

exp_content = '''import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33   # YOLOX-S depth
        self.width = 0.50   # YOLOX-S width
        self.exp_name = "yolox_uavswarm"

        # Dataset directories
        self.data_dir = "/content/UAVSwarm-dataset"
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.train_name = "train"
        self.val_name = "train"

        # Training schedule
        self.max_epoch = 12
        self.data_num_workers = 4
        self.eval_interval = 2
        self.print_interval = 50
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # Learning Rate Schedule
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 1
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # Aerial drone augmentation
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

with open(exp_file, "w") as f:
    f.write(exp_content)
print(f"✅ Experiment written to {exp_file}")

# 2. Check training invocation
os.chdir("/content/YOLOX")
train_cmd = [
    sys.executable, "-m", "yolox.tools.train",
    "-f", str(exp_file),
    "-c", "/content/yolox_s.pth",
    "-b", "16",
    "-d", "1",
    "--fp16",
    "-o"
]
print("Executing train command:", " ".join(train_cmd))
subprocess.run(train_cmd, check=True)
