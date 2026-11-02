import re
from pathlib import Path

req_file = Path("/content/YOLOX/requirements.txt")
if req_file.exists():
    text = req_file.read_text()
    print("Original requirements:\n", text)
    # Remove onnx-simplifier or make it optional
    filtered = re.sub(r"onnx-simplifier.*", "", text)
    req_file.write_text(filtered)
    print("\nUpdated requirements:\n", req_file.read_text())

# Also check setup.py
setup_file = Path("/content/YOLOX/setup.py")
if setup_file.exists():
    text = setup_file.read_text()
    if "onnx-simplifier" in text:
        text = re.sub(r'["\']onnx-simplifier.*?["\']', '', text)
        setup_file.write_text(text)
        print("Removed onnx-simplifier from setup.py")
