import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class DynamicRoutingEnv(gym.Env):
    def __init__(self, num_nodes=6, src=0, dst=5):
        super(DynamicRoutingEnv, self).__init__()
        self.num_nodes = num_nodes
        self.src = src
        self.dst = dst
        
        # Action Space: Select next hop node (0 to num_nodes-1)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation Space: [Current Node ID, Target Node ID, Flattened Link Latency Matrix]
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(2 + num_nodes * num_nodes,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Create a connected network topology
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.6, seed=42)
        while not nx.is_connected(self.G):
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.6)
            
        # Initialize random dynamic link latencies (ms)
        self.link_delays = np.random.uniform(5.0, 50.0, size=(self.num_nodes, self.num_nodes))
        
        self.current_node = self.src
        return self._get_obs(), {}

    def _get_obs(self):
        flattened_matrix = self.link_delays.flatten()
        return np.concatenate(([self.current_node, self.dst], flattened_matrix)).astype(np.float32)

    def get_valid_actions(self):
        """Returns a binary mask array of valid connected neighbors for current_node."""
        valid_mask = np.zeros(self.num_nodes, dtype=np.float32)
        for neighbor in self.G.neighbors(self.current_node):
            valid_mask[neighbor] = 1.0
        return valid_mask

    def step(self, action):
        next_node = action
        
        # Check if the chosen next_node is a valid edge
        if not self.G.has_edge(self.current_node, next_node):
            reward = -100.0
            done = True
            return self._get_obs(), reward, done, False, {"status": "invalid_hop"}
        
        # Calculate latency cost
        delay = self.link_delays[self.current_node][next_node]
        self.current_node = next_node
        
        # Reward is negative latency
        reward = -delay
        
        # Check if destination reached
        if self.current_node == self.dst:
            reward += 100.0  # Reward bonus for reaching target
            done = True
            return self._get_obs(), reward, done, False, {"status": "success"}
        
        # Dynamic traffic fluctuation: link delays slightly shift over time
        if np.random.rand() < 0.2:
            self.link_delays += np.random.uniform(-2.0, 2.0, size=(self.num_nodes, self.num_nodes))
            self.link_delays = np.clip(self.link_delays, 1.0, 100.0)

        done = False
        return self._get_obs(), reward, done, False, {}