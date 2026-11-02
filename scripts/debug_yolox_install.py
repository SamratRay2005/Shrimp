import subprocess
import os

os.chdir("/content/YOLOX")
res = subprocess.run(["pip", "install", "-v", "-e", "."], capture_output=True, text=True)
print("Return code:", res.returncode)
print("STDOUT:\n", res.stdout[-1000:])
print("STDERR:\n", res.stderr[-1000:])
