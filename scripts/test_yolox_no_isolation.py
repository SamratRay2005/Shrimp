import subprocess
import os

os.chdir("/content/YOLOX")
res = subprocess.run(["pip", "install", "--no-build-isolation", "-e", "."], capture_output=True, text=True)
print("Return code:", res.returncode)
if res.returncode != 0:
    print("STDERR:", res.stderr[-500:])
else:
    print("SUCCESS: YOLOX installed with --no-build-isolation!")
    subprocess.run(["pip", "install", "-q", "onnx", "onnxruntime", "pycocotools", "loguru", "thop"])
    import yolox
    print("Imported yolox successfully:", yolox.__file__)
