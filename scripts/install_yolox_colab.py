import subprocess
import os

os.chdir("/content/YOLOX")
res = subprocess.run(["pip", "install", "--no-build-isolation", "-e", "."], capture_output=True, text=True)
print("Return code:", res.returncode)
if res.returncode != 0:
    print("STDOUT:\n", res.stdout[-600:])
    print("STDERR:\n", res.stderr[-600:])
else:
    print("SUCCESS: YOLOX installed into environment!")
    import yolox
    print("Imported YOLOX from:", yolox.__file__)
