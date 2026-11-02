import subprocess
import time
import os

print("=== Testing git clone of UAVSwarm dataset on Colab ===")
t0 = time.time()
p = subprocess.run([
    "git", "clone", "--depth", "1", "https://github.com/UAVSwarm/UAVSwarm-dataset.git", "/content/UAVSwarm-dataset"
], capture_output=True, text=True)

print("Returncode:", p.returncode)
print("Stdout:", p.stdout)
print("Stderr:", p.stderr[-500:] if len(p.stderr) > 500 else p.stderr)
print(f"Time taken: {time.time() - t0:.2f} seconds")

if os.path.exists("/content/UAVSwarm-dataset"):
    res = subprocess.run(["du", "-sh", "/content/UAVSwarm-dataset"], capture_output=True, text=True)
    print("Dataset size:", res.stdout)
