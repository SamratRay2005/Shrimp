#!/usr/bin/env python3
"""
Automated Backup Synchronizer for Colab Runtime.
Protects model weights and logs against sudden runtime disconnects.
Continuously or one-shot mirrors checkpoints and logs from Colab to local host.
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Paths to mirror
SYNC_TARGETS = [
    # (remote_path, local_path)
    ("/content/dfine_train.log", "logs/dfine_train.log"),
    ("/content/rl_train.log", "logs/rl_train.log"),
    ("/content/drone_tracker_policy_best.pth", "checkpoints/drone_tracker_policy_best.pth"),
    ("/content/drone_tracker_policy.pth", "checkpoints/drone_tracker_policy.pth"),
    ("/content/output/dfine_hgnetv2_m_uavswarm/best_stg1.pth", "checkpoints/dfine_best_stg1.pth"),
    ("/content/output/dfine_hgnetv2_m_uavswarm/last.pth", "checkpoints/dfine_last.pth"),
    ("/content/output/dfine_hgnetv2_m_uavswarm/checkpoint0000.pth", "checkpoints/dfine_epoch0.pth"),
    ("/content/output/dfine_hgnetv2_m_uavswarm/checkpoint0001.pth", "checkpoints/dfine_epoch1.pth"),
    ("/content/output/dfine_hgnetv2_m_uavswarm/dfine_drone.onnx", "checkpoints/dfine_drone.onnx"),
    ("/content/results/drone_trajectories.csv", "results/drone_trajectories.csv"),
    ("/content/results/drone_trajectories.json", "results/drone_trajectories.json"),
]

def format_size(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} TB"

def sync_once(session_name="trainer"):
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Starting backup sync from Colab session '{session_name}'...")
    
    synced_count = 0
    for remote, local in SYNC_TARGETS:
        local_path = repo_root / local
        cmd = ["colab", "download", "-s", session_name, remote, str(local_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode == 0 and local_path.exists():
                size = format_size(local_path.stat().st_size)
                print(f"  ✓ {local} ({size})")
                synced_count += 1
            else:
                # File may not exist yet remotely, which is normal for later epochs
                pass
        except subprocess.TimeoutExpired:
            print(f"  ⏳ Timeout downloading {remote}")
        except Exception as e:
            print(f"  ✗ Error syncing {remote}: {e}")
            
    print(f"[{timestamp}] Backup sync complete. Successfully mirrored {synced_count} items locally.")

def main():
    parser = argparse.ArgumentParser(description="Mirror Colab checkpoints locally")
    parser.add_argument("--session", default="trainer", help="Colab session name")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=180, help="Sync interval in seconds (default: 180s)")
    args = parser.parse_args()
    
    if args.loop:
        print(f"Starting auto-backup daemon (interval: {args.interval}s, session: {args.session})...")
        while True:
            try:
                sync_once(args.session)
            except KeyboardInterrupt:
                print("\nStopping auto-backup daemon.")
                break
            except Exception as e:
                print(f"Sync error: {e}")
            time.sleep(args.interval)
    else:
        sync_once(args.session)

if __name__ == "__main__":
    main()
