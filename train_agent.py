import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from shrimp_env import ShrimpTrackingEnv


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, device):
        self.device = device
        self.q_net = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.gate_radius = 150.0

    def _featurize(self, state):
        return torch.FloatTensor([
            [
                s['dx']/self.gate_radius,
                s['dy']/self.gate_radius,
                s['region_id']/7.0,
                s['vx']/self.gate_radius,
                s['vy']/self.gate_radius
            ]
            for s in state
        ]).to(self.device)

    def get_q_values(self, state_list):
        with torch.no_grad():
            return self.q_net(self._featurize(state_list)).cpu().numpy().flatten()

    def select_action(self, state_list):
        if random.random() < self.epsilon:
            return random.randint(0, len(state_list)-1)
        return np.argmax(self.get_q_values(state_list))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in batch:
            q_values = self.q_net(self._featurize(state))
            current_q = q_values[action]

            if done or next_state is None:
                target_q = torch.tensor([reward]).to(self.device)
            else:
                with torch.no_grad():
                    max_next_q = torch.max(
                        self.q_net(self._featurize(next_state)))
                target_q = reward + self.gamma * max_next_q

            loss = self.loss_fn(current_q.view(1), target_q.view(1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.batch_size


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ShrimpTrackingEnv('rl_clean_training_data.csv', gate_radius=150)
    agent = DQNAgent(device)

    epochs = 300

    for epoch in range(epochs):
        state, _, true_index, done = env.reset()
        total_reward = 0

        while not done:
            action = agent.select_action(state)

            chosen = state[action]
            true = state[true_index]

            dist = ((chosen['dx']-true['dx'])**2 +
                    (chosen['dy']-true['dy'])**2)**0.5
            threshold = agent.gate_radius * 0.2

            if action == true_index:
                reward = 1.0
            elif dist < threshold:
                reward = 0.5
            else:
                reward = -1.0

            next_state, _, next_true_index, done = env.step()

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            true_index = next_true_index
            total_reward += reward

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (epoch+1) % 20 == 0:
            print(
                f"Epoch {epoch+1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.q_net.state_dict(), "shrimp_tracker_policy.pth")


if __name__ == "__main__":
    main()
