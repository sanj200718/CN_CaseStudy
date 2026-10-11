import networkx as nx
import numpy as np
import torch
from env import DynamicRoutingEnv
from model import QNetwork

# 1. Initialize Environment
env = DynamicRoutingEnv(num_nodes=6, src=0, dst=5)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 2. Load Trained ML Model
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("routing_dqn.pt"))
q_net.eval()

# 3. Compute Standard Dijkstra Route
dijkstra_path = nx.shortest_path(env.G, source=0, target=5, weight=None)
dijkstra_delay = sum(env.link_delays[u][v] for u, v in zip(dijkstra_path[:-1], dijkstra_path[1:]))

# 4. Compute ML Agent Route
state, _ = env.reset()
dqn_path = [0]
total_dqn_delay = 0.0
done = False

while not done and len(dqn_path) < 10:
    curr = env.current_node
    
    # Get valid neighbor mask for action selection
    valid_mask = env.get_valid_actions()
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_vals = q_net(state_tensor).squeeze(0).numpy()
        q_vals[valid_mask == 0.0] = -1e9
        action = int(np.argmax(q_vals))
        
    next_state, reward, done, _, _ = env.step(action)
    
    if action in env.G.neighbors(curr):
        total_dqn_delay += env.link_delays[curr][action]
        dqn_path.append(action)
        
    state = next_state

print("=== Intelligent Routing Comparison Results ===")
print(f"Standard Dijkstra Route: {dijkstra_path} | Cost/Delay: {dijkstra_delay:.2f} ms")
print(f"ML Agent Route       : {dqn_path} | Cost/Delay: {total_dqn_delay:.2f} ms")