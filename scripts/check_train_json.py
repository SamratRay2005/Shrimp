import json
from pathlib import Path

train_json_path = Path("/content/UAVSwarm-dataset/annotations/train.json")
with open(train_json_path) as f:
    data = json.load(f)

print("Categories:", data.get("categories"))
print("Total images:", len(data.get("images", [])))
print("Total annotations:", len(data.get("annotations", [])))
print("Sample image record:", data["images"][0])
sample_ann = data["annotations"][0] if data["annotations"] else None
print("Sample annotation:", sample_ann)

# Verify if image file exists on disk
img_path = Path("/content/UAVSwarm-dataset/train") / data["images"][0]["file_name"]
print(f"Check sample image file {img_path} exists:", img_path.exists())
