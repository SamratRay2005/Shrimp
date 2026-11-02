import subprocess
import os
from pathlib import Path

yolox_dir = Path("/content/YOLOX")
if not yolox_dir.exists():
    print("Cloning YOLOX...")
    subprocess.run(["git", "clone", "https://github.com/Megvii-BaseDetection/YOLOX.git", str(yolox_dir)], check=True)

os.chdir(yolox_dir)
print("Installing YOLOX in editable mode...")
subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
subprocess.run(["pip", "install", "-q", "onnx", "onnxruntime", "pycocotools", "loguru", "thop"], check=True)

import yolox
print("YOLOX installed successfully! Version:", yolox.__file__)
