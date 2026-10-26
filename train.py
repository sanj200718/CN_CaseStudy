"""Train the DQN routing model with valid-action-masked Bellman targets.

Running this file overwrites routing_dqn.pt and learning_curve.png. The checked-
in trained checkpoint is intentionally not overwritten during evaluation.
"""
from collections import deque
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from env import DynamicRoutingEnv
from model import QNetwork

NUM_EPISODES = 600
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99
LEARNING_RATE = 0.0005
MEMORY_SIZE = 5000
SEED = 42
ROOT = Path(__file__).resolve().parent

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
env = DynamicRoutingEnv(num_nodes=6, src=0, dst=5, max_steps=12)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPS_START
rewards_history = []

print("Starting DQN training with visited-node action masks and masked Bellman targets...")
for episode in range(NUM_EPISODES):
    state, _ = env.reset(seed=SEED + episode)
    total_reward = 0.0
    done = False
    while not done:
        valid_mask = env.get_valid_actions()
        valid_indices = np.flatnonzero(valid_mask > 0.5)
        if len(valid_indices) == 0:
            # Defensive; env.step normally flags dead ends on the previous transition.
            break

        if random.random() < epsilon:
            action = int(random.choice(valid_indices.tolist()))
        else:
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                q_values[torch.as_tensor(valid_mask == 0.0)] = -1e9
                action = int(torch.argmax(q_values).item())

        next_state, reward, terminated, truncated, info = env.step(action)
        next_mask = env.get_valid_actions()
        # Mark transitions terminal when next state has no permitted action.
        next_has_action = bool(np.any(next_mask > 0.5))
        done = bool(terminated or truncated or not next_has_action)
        if not next_has_action and not terminated and not truncated:
            reward -= 25.0
        memory.append((state, action, float(reward), next_state, done, next_mask.astype(np.float32)))
        state = next_state
        total_reward += float(reward)

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones, next_masks = zip(*batch)
            b_states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            b_actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
            b_rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
            b_next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
            b_dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)
            b_next_masks = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool)

            q_values = q_net(b_states).gather(1, b_actions)
            with torch.no_grad():
                next_q = q_net(b_next_states)
                next_q[~b_next_masks] = -1e9
                max_next_q = next_q.max(dim=1, keepdim=True).values
                no_next_action = ~b_next_masks.any(dim=1, keepdim=True)
                max_next_q[no_next_action] = 0.0
                targets = b_rewards + GAMMA * max_next_q * (1.0 - b_dones)
            loss = F.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
            optimizer.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    rewards_history.append(total_reward)
    if (episode + 1) % 30 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES} | Avg reward (last 30): "
              f"{np.mean(rewards_history[-30:]):.2f} | epsilon: {epsilon:.3f}")

# Explicitly overwrite only when the user chooses to run this training script.
torch.save(q_net.state_dict(), ROOT / "routing_dqn.pt")
print("Saved model to routing_dqn.pt")
plt.figure(figsize=(8, 5))
plt.plot(rewards_history, label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN learning curve (valid-action-masked targets)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "learning_curve.png", dpi=160)
plt.close()
print("Saved learning curve to learning_curve.png")
