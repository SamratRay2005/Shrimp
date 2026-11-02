"""
Drone DQN Agent — Adapted from Shrimp Architecture
====================================================
Original: DQNAgent + QNetwork (shrimp/train_agent.py)
Adapted:  DroneQNetwork + DroneAgent for UAVSwarm tracking

ARCHITECTURE (identical to Shrimp v2, preserved for comparability):
  Input:  5D feature vector per candidate (dx, dy, region_id, vx, vy)
  Hidden: 5 → 128 → 64 → 1  (expanded from Shrimp's 5→64→32→1)
  Output: 1 Q-value per candidate; argmax = selected detection

WHY SAME ARCHITECTURE:
  The task is identical at the mathematical level: given N candidates
  with spatial features, score each one. The shrimp architecture proved
  this MLP-per-candidate approach works. We scale hidden size slightly
  (128 vs 64) to handle harder drone swarm scenarios.

KEY TRAINING CHANGES vs SHRIMP:
  1. Larger replay buffer (50k vs 10k) — drone sequences are longer
  2. WandB logging for training visibility on Colab
  3. Checkpoint saves every 50 epochs + best model
  4. Validation accuracy tracking (not just reward)

USAGE:
  python train_drone_agent.py \
      --csv drone_training_data.csv \
      --epochs 500 \
      --save_path drone_tracker_policy.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
from collections import deque
from drone_env import DroneTrackingEnv

# Optional WandB logging (gracefully disabled if not installed)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[INFO] wandb not installed — logging to console only")


# ── Q-Network ────────────────────────────────────────────────────────────────

class DroneQNetwork(nn.Module):
    """
    Per-candidate scoring MLP.
    Input: 5D feature vector (dx, dy, region_id, vx, vy)
    Output: scalar Q-value

    Slightly wider than Shrimp (128 hidden units) to handle harder
    drone swarm disambiguation. BatchNorm removed — inputs are normalized
    externally; BN would break on single-candidate frames.
    """
    def __init__(self, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular experience replay buffer. Larger than Shrimp (50k vs 10k)."""
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DroneAgent:
    """
    DQN agent for drone swarm data association.

    Identical DQN mechanics to Shrimp:
      - Epsilon-greedy exploration
      - Single Q-network (no target network — episodes are short enough)
      - Per-transition gradient updates
      - Gradient clipping at norm=1.0

    Enhanced features:
      - Tracks accuracy (% correct associations) alongside reward
      - Saves best model checkpoint by validation accuracy
    """

    def __init__(self, device: torch.device, gate_radius: float = 250.0,
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, batch_size: int = 64,
                 buffer_capacity: int = 50_000):
        self.device = device
        self.gate_radius = gate_radius

        self.q_net = DroneQNetwork().to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _featurize(self, state: list) -> torch.Tensor:
        """
        Convert state list to normalized feature tensor.
        Normalization constants match Shrimp exactly for compatibility.
        dx, dy, vx, vy → divided by gate_radius (puts in [-1, 1] approx)
        region_id       → divided by 7 (puts in [0, 1])
        """
        feats = [
            [
                s['dx'] / self.gate_radius,
                s['dy'] / self.gate_radius,
                s['region_id'] / 7.0,
                s['vx'] / self.gate_radius,
                s['vy'] / self.gate_radius,
            ]
            for s in state
        ]
        return torch.FloatTensor(feats).to(self.device)

    def get_q_values(self, state: list) -> np.ndarray:
        with torch.no_grad():
            return self.q_net(self._featurize(state)).cpu().numpy().flatten()

    def select_action(self, state: list) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(state) - 1)
        return int(np.argmax(self.get_q_values(state)))

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        total_loss = 0.0

        for state, action, reward, next_state, done in batch:
            q_vals = self.q_net(self._featurize(state))
            current_q = q_vals[action]

            if done or next_state is None:
                target_q = torch.tensor([reward], dtype=torch.float32,
                                        device=self.device)
            else:
                with torch.no_grad():
                    max_next_q = torch.max(self.q_net(self._featurize(next_state)))
                target_q = reward + self.gamma * max_next_q

            loss = self.loss_fn(current_q.view(1), target_q.view(1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.batch_size

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ── Training Loop ────────────────────────────────────────────────────────────

def compute_reward(action: int, true_index: int, state: list,
                   gate_radius: float) -> float:
    """
    Drone-specific reward function (enhanced from Shrimp's binary +1/-1).

    +1.0  : Exact correct selection
    +0.5  : Near miss — selected drone is within 20% of gate from truth
             (reward partial credit in dense swarms where 2 drones are very close)
    -1.0  : Wrong selection, far from truth

    WHY THE NEAR-MISS REWARD:
      In dense drone swarms, two drones can be very close. Penalizing the
      agent equally for picking a very nearby drone (vs a distant one) creates
      a misleading gradient signal. The 0.5 partial credit acknowledges this.
    """
    if action == true_index:
        return 1.0

    chosen = state[action]
    truth  = state[true_index]
    dist = ((chosen['dx'] - truth['dx'])**2 +
            (chosen['dy'] - truth['dy'])**2) ** 0.5

    near_miss_threshold = gate_radius * 0.2  # 50px at gate=250
    if dist < near_miss_threshold:
        return 0.5
    return -1.0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # Init environment + agent
    env = DroneTrackingEnv(
        csv_path=args.csv,
        gate_radius=args.gate_radius,
        num_clutter=args.num_clutter,
        use_swarm_clutter=True,
    )
    agent = DroneAgent(
        device=device,
        gate_radius=args.gate_radius,
        lr=args.lr,
    )

    # WandB setup
    if WANDB_AVAILABLE and args.wandb:
        wandb.init(project='drone-rl-tracker', config=vars(args))
        wandb.watch(agent.q_net)

    best_accuracy = 0.0
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    print(f"[Train] Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        state, _, true_index, done = env.reset()
        total_reward = 0.0
        correct = 0
        steps = 0
        last_loss = 0.0

        while not done:
            action = agent.select_action(state)
            reward = compute_reward(action, true_index, state, args.gate_radius)

            if action == true_index:
                correct += 1

            next_state, _, next_true_index, done = env.step()
            agent.memory.push(state, action, reward, next_state, done)
            last_loss = agent.train_step()

            state = next_state
            true_index = next_true_index
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        accuracy = correct / max(steps, 1) * 100

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Reward: {total_reward:+7.2f} | "
                  f"Acc: {accuracy:5.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {last_loss:.4f}", flush=True)

        # Log to WandB
        if WANDB_AVAILABLE and args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'reward': total_reward,
                'accuracy': accuracy,
                'epsilon': agent.epsilon,
                'loss': last_loss,
            })

        # Save best model
        if accuracy > best_accuracy and steps >= 5:
            best_accuracy = accuracy
            best_path = args.save_path.replace('.pth', '_best.pth')
            torch.save(agent.q_net.state_dict(), best_path)

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            ckpt_path = args.save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(agent.q_net.state_dict(), ckpt_path)
            print(f"  [Checkpoint] Saved: {ckpt_path}", flush=True)

    # Final save
    torch.save(agent.q_net.state_dict(), args.save_path)
    print(f"\n✅ Training complete! Best accuracy: {best_accuracy:.1f}%", flush=True)
    print(f"   Final model: {args.save_path}", flush=True)
    print(f"   Best model:  {args.save_path.replace('.pth', '_best.pth')}", flush=True)

    if WANDB_AVAILABLE and args.wandb:
        wandb.finish()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drone RL Tracking Agent')
    parser.add_argument('--csv',         required=True,    help='Path to training CSV')
    parser.add_argument('--epochs',      type=int,   default=500)
    parser.add_argument('--gate_radius', type=float, default=250.0)
    parser.add_argument('--num_clutter', type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--save_path',   default='./results/drone_tracker_policy.pth')
    parser.add_argument('--wandb',       action='store_true', help='Enable WandB logging')
    args = parser.parse_args()

    train(args)
