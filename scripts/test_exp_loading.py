import os
import sys
import torch
from pathlib import Path

# Download base pretrained weights if not present
weights_path = Path("/content/yolox_s.pth")
if not weights_path.exists():
    print("Downloading YOLOX-S COCO pretrained weights...")
    import urllib.request
    url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    urllib.request.urlretrieve(url, str(weights_path))
    print(f"Downloaded yolox_s.pth ({weights_path.stat().st_size / 1e6:.1f} MB)")
else:
    print("yolox_s.pth already exists.")

# Check experiment definition
sys.path.insert(0, "/content/YOLOX")
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_uavswarm"

        # Dataset directories
        self.data_dir = "/content/UAVSwarm-dataset"
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"

        # In UAVSwarm-dataset, train/ and test/ folders contain the sequence subfolders
        # train_half.json and val_half.json both reference images in train/
        self.train_name = "train"
        self.val_name = "train"

        # Hyperparameters
        self.max_epoch = 15
        self.data_num_workers = 2
        self.eval_interval = 1
        self.print_interval = 20
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 1

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

exp = Exp()
train_dataset = exp.get_dataset()
print(f"Dataset successfully initialized! Total train images: {len(train_dataset)}")
sample_img, sample_target, sample_info, sample_id = train_dataset[0]
print(f"Sample image shape: {sample_img.shape}, target shape: {sample_target.shape}")

val_dataset = exp.get_eval_dataset()
print(f"Val dataset successfully initialized! Total val images: {len(val_dataset)}")
