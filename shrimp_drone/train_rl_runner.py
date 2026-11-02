import os
import sys
import time
import argparse

# Ensure shrimp_drone is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_drone_agent import train

def main():
    parser = argparse.ArgumentParser(description="Drone RL Data Association Tracker Trainer")
    parser.add_argument('--csv', default='/content/drone_training_data.csv', help='Path to trajectories CSV')
    parser.add_argument('--epochs', type=int, default=600, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gate_radius', type=float, default=250.0, help='Gate radius (pixels)')
    parser.add_argument('--num_clutter', type=int, default=4, help='Clutter count')
    parser.add_argument('--save_path', default='/content/drone_tracker_policy.pth', help='Model output path')
    parser.add_argument('--wandb', action='store_true', default=False, help='WandB logging')
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"🚁 Starting Drone RL Agent Training ({args.epochs} epochs)", flush=True)
    print(f"Data: {args.csv}", flush=True)
    print(f"Gate: {args.gate_radius}px | Clutter: {args.num_clutter} | LR: {args.lr}", flush=True)
    print(f"Save: {args.save_path}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    train(args)
    elapsed = time.time() - t0
    print(f"\n🎉 RL Training Finished Successfully in {elapsed / 60:.2f} minutes!", flush=True)

if __name__ == '__main__':
    main()
