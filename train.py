import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from env import DynamicRoutingEnv
from model import QNetwork

# Hyperparameters
NUM_EPISODES = 600
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99

env = DynamicRoutingEnv(num_nodes=6, src=0, dst=5)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=0.0005)
memory = deque(maxlen=2000)

epsilon = EPS_START
rewards_history = []

print("Starting RL Agent Training with Action Masking...")

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Get valid action mask from environment
        valid_mask = env.get_valid_actions()
        valid_indices = np.where(valid_mask == 1.0)[0]
        
        # If no valid neighbors exist (isolated node edge-case)
        if len(valid_indices) == 0:
            break

        # Action Selection with Masking
        if random.random() < epsilon:
            action = int(random.choice(valid_indices))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_net(state_tensor).squeeze(0).numpy()
                # Suppress invalid actions by setting Q-values to -infinity
                q_vals[valid_mask == 0.0] = -1e9
                action = int(np.argmax(q_vals))
                
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        # Experience Replay Training Step
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            b_states = torch.FloatTensor(np.array(states))
            b_actions = torch.LongTensor(actions).unsqueeze(1)
            b_rewards = torch.FloatTensor(rewards).unsqueeze(1)
            b_next_states = torch.FloatTensor(np.array(next_states))
            b_dones = torch.FloatTensor(dones).unsqueeze(1)
            
            # Compute Q Targets
            q_values = q_net(b_states).gather(1, b_actions)
            with torch.no_grad():
                max_next_q = q_net(b_next_states).max(1)[0].unsqueeze(1)
                targets = b_rewards + (GAMMA * max_next_q * (1 - b_dones))
                
            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    rewards_history.append(total_reward)
    
    if (episode + 1) % 30 == 0:
        avg_reward = np.mean(rewards_history[-30:])
        print(f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward (Last 30): {avg_reward:.2f} | Epsilon: {epsilon:.2f}")

# Save Trained Model Parameters
torch.save(q_net.state_dict(), "routing_dqn.pt")
print("Model saved to 'routing_dqn.pt'.")

# Save Learning Plot
plt.figure(figsize=(8, 5))
plt.plot(rewards_history, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Learning Curve (With Action Masking)")
plt.grid(True)
plt.savefig("learning_curve.png")
print("Learning curve plot saved to 'learning_curve.png'.")