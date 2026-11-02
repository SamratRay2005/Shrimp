import os
import subprocess

print("=== Colab Environment Inspection ===")
print("Current dir:", os.getcwd())
print("Files in /content:")
try:
    for item in os.listdir("/content"):
        p = os.path.join("/content", item)
        sz = os.path.getsize(p) if os.path.isfile(p) else "dir"
        print(f"  {item} ({sz})")
except Exception as e:
    print("Error listing /content:", e)

print("\nDisk space:")
subprocess.run(["df", "-h", "/content"])
