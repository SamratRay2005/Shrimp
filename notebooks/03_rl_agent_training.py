# ============================================================
#  NOTEBOOK 3: RL Agent Training — Drone Shrimp Architecture
#  Run on Google Colab / Kaggle after notebooks 1 & 2
# ============================================================

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, sys, subprocess, shutil
BASE_DIR = '/content/drive/MyDrive/drone_tracker'
CSV_PATH = f'{BASE_DIR}/data/drone_training_data.csv'
SAVE_DIR = f'{BASE_DIR}/results'
os.makedirs(SAVE_DIR, exist_ok=True)

# Copy shrimp_drone source to /content for import
SRC = '/content/RM_Shrimp_extension/shrimp_drone'
DST = '/content/shrimp_drone'
if not os.path.exists(DST):
    shutil.copytree(SRC, DST)
sys.path.insert(0, DST)

subprocess.run(['pip', 'install', '-q', 'wandb', 'pandas', 'numpy'], check=True)
print("✅ Ready")

# ── Cell 2: Verify training data ──────────────────────────────────────────────
import pandas as pd
import numpy as np

df = pd.read_csv(CSV_PATH)
print(f"Training data: {len(df):,} rows, {df['track_id'].nunique()} tracks")
print(df[['frame', 'track_id', 'cx', 'cy']].head())

# ── Cell 3: Quick sanity check of DroneTrackingEnv ────────────────────────────
from drone_env import DroneTrackingEnv

env = DroneTrackingEnv(
    csv_path=CSV_PATH,
    gate_radius=250.0,
    num_clutter=4,
    use_swarm_clutter=True,
)

state, candidates, true_idx, done = env.reset()
print(f"\nEnv test: {len(candidates)} candidates, true_idx={true_idx}")
for i, s in enumerate(state):
    tag = " ← TRUTH" if i == true_idx else ""
    print(f"  [{i}] r={s['region_id']} dx={s['dx']:+.1f} dy={s['dy']:+.1f} "
          f"vx={s['vx']:+.1f} vy={s['vy']:+.1f}{tag}")

# ── Cell 4: WandB login (optional) ────────────────────────────────────────────
# import wandb
# wandb.login()   # Will prompt for API key

# ── Cell 5: Train the RL Agent ────────────────────────────────────────────────
import torch
import random
from train_drone_agent import DroneAgent, compute_reward, DroneTrackingEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# Hyperparameters
EPOCHS      = 600      # More epochs than shrimp (500) — harder problem
GATE_RADIUS = 250.0
NUM_CLUTTER = 4
LR          = 1e-3
SAVE_PATH   = f'{SAVE_DIR}/drone_tracker_policy.pth'

# Init
env = DroneTrackingEnv(CSV_PATH, gate_radius=GATE_RADIUS,
                       num_clutter=NUM_CLUTTER, use_swarm_clutter=True)
agent = DroneAgent(device=device, gate_radius=GATE_RADIUS, lr=LR)

best_accuracy = 0.0
history = {'reward': [], 'accuracy': [], 'epsilon': [], 'loss': []}

print(f"Starting training: {EPOCHS} epochs | gate={GATE_RADIUS}px | lr={LR}")
print("-" * 70)

for epoch in range(EPOCHS):
    state, _, true_index, done = env.reset()
    total_reward = 0.0
    correct = 0
    steps = 0
    last_loss = 0.0

    while not done:
        action = agent.select_action(state)
        reward = compute_reward(action, true_index, state, GATE_RADIUS)

        if action == true_index:
            correct += 1

        next_state, _, next_true_index, done = env.step()
        agent.memory.push(state, action, reward, next_state, done)
        last_loss = agent.train_step()

        state      = next_state
        true_index = next_true_index
        total_reward += reward
        steps += 1

    agent.decay_epsilon()
    acc = correct / max(steps, 1) * 100

    history['reward'].append(total_reward)
    history['accuracy'].append(acc)
    history['epsilon'].append(agent.epsilon)
    history['loss'].append(last_loss)

    if (epoch + 1) % 25 == 0:
        print(f"Ep {epoch+1:4d}/{EPOCHS} | "
              f"Rew: {total_reward:+7.2f} | "
              f"Acc: {acc:5.1f}% | "
              f"ε: {agent.epsilon:.3f} | "
              f"Loss: {last_loss:.4f}")

    # Best model checkpoint
    if acc > best_accuracy and steps >= 5:
        best_accuracy = acc
        best_path = SAVE_PATH.replace('.pth', '_best.pth')
        torch.save(agent.q_net.state_dict(), best_path)

    # Periodic save every 100 epochs
    if (epoch + 1) % 100 == 0:
        ckpt = SAVE_PATH.replace('.pth', f'_ep{epoch+1}.pth')
        torch.save(agent.q_net.state_dict(), ckpt)
        print(f"  💾 Checkpoint: {ckpt}")

# Final save
torch.save(agent.q_net.state_dict(), SAVE_PATH)
print(f"\n✅ Training complete!")
print(f"   Best accuracy: {best_accuracy:.1f}%")
print(f"   Saved: {SAVE_PATH}")
print(f"   Best:  {SAVE_PATH.replace('.pth', '_best.pth')}")

# ── Cell 6: Plot Training Curves ──────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Drone RL Agent Training', fontsize=14)

axes[0, 0].plot(history['reward'], color='steelblue', alpha=0.6, linewidth=0.8)
# Smooth curve
window = 20
smoothed = pd.Series(history['reward']).rolling(window).mean()
axes[0, 0].plot(smoothed, color='darkblue', linewidth=2, label=f'{window}-ep avg')
axes[0, 0].set_title('Episode Reward')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()

axes[0, 1].plot(history['accuracy'], color='green', alpha=0.6, linewidth=0.8)
smoothed_acc = pd.Series(history['accuracy']).rolling(window).mean()
axes[0, 1].plot(smoothed_acc, color='darkgreen', linewidth=2)
axes[0, 1].axhline(best_accuracy, color='red', linestyle='--',
                    label=f'Best: {best_accuracy:.1f}%')
axes[0, 1].set_title('Association Accuracy (%)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()

axes[1, 0].plot(history['epsilon'], color='orange')
axes[1, 0].set_title('Epsilon (Exploration)')
axes[1, 0].set_xlabel('Epoch')

axes[1, 1].plot(history['loss'], color='tomato', alpha=0.6, linewidth=0.8)
smoothed_loss = pd.Series(history['loss']).rolling(window).mean()
axes[1, 1].plot(smoothed_loss, color='darkred', linewidth=2)
axes[1, 1].set_title('Training Loss')
axes[1, 1].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/rl_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Curves saved: {SAVE_DIR}/rl_training_curves.png")

# ── Cell 7: Evaluate Trained Agent ───────────────────────────────────────────
"""
Run the trained agent on the full dataset in greedy mode (epsilon=0)
to get a clean accuracy metric.
"""
# Load best weights
agent.q_net.load_state_dict(torch.load(SAVE_PATH.replace('.pth', '_best.pth'),
                                        map_location=device))
agent.epsilon = 0.0  # Pure greedy

total_correct = 0
total_steps = 0
n_episodes = 100  # Evaluate on 100 random tracks

print(f"Evaluating on {n_episodes} random episodes (greedy)...")
for _ in range(n_episodes):
    state, _, true_index, done = env.reset()
    while not done:
        action = agent.select_action(state)
        if action == true_index:
            total_correct += 1
        total_steps += 1
        state, _, true_index, done = env.step()

final_acc = total_correct / max(total_steps, 1) * 100
print(f"\n📊 Greedy Evaluation (n={n_episodes} episodes):")
print(f"   Association Accuracy: {final_acc:.2f}%")
print(f"   (Random baseline: {100 / (NUM_CLUTTER + 1):.1f}%  — "
      f"1/{NUM_CLUTTER+1} chance)")
