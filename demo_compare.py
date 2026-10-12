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

# 3. Lock in a fixed snapshot of the network topology and link delays for a fair test
fixed_graph = env.G.copy()
fixed_delays = env.link_delays.copy()

# --- A. Evaluate Standard Dijkstra Routing ---
# Compute shortest path using the fixed delay matrix as edge weights
for u, v, data in fixed_graph.edges(data=True):
    fixed_graph[u][v]['weight'] = fixed_delays[u][v]

dijkstra_path = nx.shortest_path(fixed_graph, source=0, target=5, weight='weight')
dijkstra_delay = sum(fixed_delays[u][v] for u, v in zip(dijkstra_path[:-1], dijkstra_path[1:]))

# --- B. Evaluate ML Agent Routing ---
# Reset environment with the same delay state snapshot
state, _ = env.reset()
env.G = fixed_graph
env.link_delays = fixed_delays.copy()
env.current_node = 0

dqn_path = [0]
total_dqn_delay = 0.0
done = False

while not done and len(dqn_path) < 10:
    curr = env.current_node
    valid_mask = env.get_valid_actions()
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_vals = q_net(state_tensor).squeeze(0).numpy()
        q_vals[valid_mask == 0.0] = -1e9
        action = int(np.argmax(q_vals))
        
    next_state, reward, done, _, _ = env.step(action)
    
    if action in fixed_graph.neighbors(curr):
        total_dqn_delay += fixed_delays[curr][action]
        dqn_path.append(action)
        
    state = next_state

print("=== Fair Intelligent Routing Comparison Results ===")
print(f"Standard Dijkstra Route: {dijkstra_path} | Cost/Delay: {dijkstra_delay:.2f} ms")
print(f"ML Agent Route       : {dqn_path} | Cost/Delay: {total_dqn_delay:.2f} ms")