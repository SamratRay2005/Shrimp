import os
from pathlib import Path

root = Path("/content/UAVSwarm-dataset")
print("Folders in UAVSwarm-dataset:", [p.name for p in root.iterdir() if p.is_dir()])

annotations = root / "annotations"
if annotations.exists():
    print("Annotations files:", [p.name for p in annotations.iterdir()])

train_dir = root / "train"
if train_dir.exists():
    train_seqs = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    print(f"Train sequences ({len(train_seqs)}):", train_seqs[:5], "...", train_seqs[-2:])
    sample_seq = train_dir / train_seqs[0]
    print(f"Contents of {sample_seq.name}:", [p.name for p in sample_seq.iterdir()])
    if (sample_seq / "img1").exists():
        imgs = sorted(os.listdir(sample_seq / "img1"))
        print(f"Sample images in {sample_seq.name}/img1 ({len(imgs)}):", imgs[:3])

test_dir = root / "test"
if test_dir.exists():
    test_seqs = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    print(f"Test sequences ({len(test_seqs)}):", test_seqs[:5], "...", test_seqs[-2:])
