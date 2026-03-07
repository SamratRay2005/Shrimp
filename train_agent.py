import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from shrimp_env import ShrimpTrackingEnv

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Input: 3 NORMALIZED features (dx, dy, region_id) -> Output: 1 Q-Value
        self.fc1 = nn.Linear(3, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

# ==========================================
# 2. REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. DQN AGENT
# ==========================================
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
        
        # Normalization constant (matches the environment gate radius)
        self.gate_radius = 150.0

    def get_q_values(self, state_list):
        # NORMALIZATION APPLIED HERE
        features = [[s['dx']/self.gate_radius, s['dy']/self.gate_radius, s['region_id']/7.0] for s in state_list]
        tensor_features = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(tensor_features)
        return q_values.cpu().numpy().flatten()

    def select_action(self, state_list):
        if random.random() < self.epsilon:
            return random.randint(0, len(state_list) - 1) 
        
        q_values = self.get_q_values(state_list)
        return np.argmax(q_values) 

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0 
            
        batch = self.memory.sample(self.batch_size)
        total_loss = 0
        
        for state, action_idx, reward, next_state, done in batch:
            # NORMALIZATION APPLIED HERE
            state_features = [[s['dx']/self.gate_radius, s['dy']/self.gate_radius, s['region_id']/7.0] for s in state]
            state_tensor = torch.FloatTensor(state_features).to(self.device)
            q_values = self.q_net(state_tensor)
            current_q = q_values[action_idx] 
            
            if done or next_state is None:
                target_q = torch.FloatTensor([reward]).to(self.device)
            else:
                # NORMALIZATION APPLIED HERE
                next_features = [[s['dx']/self.gate_radius, s['dy']/self.gate_radius, s['region_id']/7.0] for s in next_state]
                next_tensor = torch.FloatTensor(next_features).to(self.device)
                with torch.no_grad():
                    max_next_q = torch.max(self.q_net(next_tensor))
                target_q = reward + (self.gamma * max_next_q)
                
            # SHAPE MISMATCH FIX APPLIED HERE (.view(1))
            loss = self.loss_fn(current_q.view(1), target_q.view(1))
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # GRADIENT CLIPPING ADDED HERE (Prevents any remaining instability)
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / self.batch_size

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training on Compute Device: {device}")
    
    env = ShrimpTrackingEnv(csv_path='rl_clean_training_data.csv', gate_radius=150, num_clutter=4)
    agent = DQNAgent(device)
    
    epochs = 300 
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        state, _, true_index, done = env.reset()
        epoch_reward = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state)
            reward = 1.0 if action == true_index else -1.0
            epoch_reward += reward
            
            next_state, _, next_true_index, done = env.step()
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            state = next_state
            true_index = next_true_index
            steps += 1
            
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch: {epoch+1:3d}/{epochs} | Steps: {steps:3d} | Total Reward: {epoch_reward:5.1f} | Epsilon: {agent.epsilon:.3f} | Avg Loss: {loss:.4f}")

    torch.save(agent.q_net.state_dict(), "shrimp_tracker_policy.pth")
    print("\n✅ Training Complete. Model weights saved to 'shrimp_tracker_policy.pth'")

if __name__ == "__main__":
    main()