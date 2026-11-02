import json
from pathlib import Path

root = Path("/content/UAVSwarm-dataset")

for name in ["test.json", "val_half.json", "train_half.json"]:
    p = root / "annotations" / name
    with open(p) as f:
        d = json.load(f)
    first_img = d["images"][0]["file_name"] if d.get("images") else "None"
    total_imgs = len(d.get("images", []))
    total_anns = len(d.get("annotations", []))
    in_train = (root / "train" / first_img).exists()
    in_test = (root / "test" / first_img).exists()
    print(f"{name}: {total_imgs} images, {total_anns} anns. Sample: '{first_img}'. Exists in train: {in_train}, in test: {in_test}")
