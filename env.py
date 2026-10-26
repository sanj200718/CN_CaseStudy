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
        
        self.action_space = spaces.Discrete(self.num_nodes)
        
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(2 + num_nodes * num_nodes,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.6, seed=seed)
        while not nx.is_connected(self.G):
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.6)
            
        # Randomly choose distinct source and destination nodes so the agent learns bidirectional & multi-path routing
        nodes = list(self.G.nodes())
        self.src = np.random.choice(nodes)
        self.dst = np.random.choice([n for n in nodes if n != self.src])
            
        self.link_delays = np.random.uniform(5.0, 50.0, size=(self.num_nodes, self.num_nodes))
        
        self.current_node = self.src
        self.visited_nodes = {self.current_node}
        
        return self._get_obs(), {}

    def _get_obs(self):
        flattened_matrix = self.link_delays.flatten()
        return np.concatenate(([self.current_node, self.dst], flattened_matrix)).astype(np.float32)

    def get_valid_actions(self):
        """Returns a binary mask of valid unvisited neighbors for current_node."""
        valid_mask = np.zeros(self.num_nodes, dtype=np.float32)
        
        # Only allow neighbors we haven't visited yet to prevent loops
        for neighbor in self.G.neighbors(self.current_node):
            if neighbor not in self.visited_nodes:
                valid_mask[neighbor] = 1.0
                
        # SAFETY RULE: If all connected rooms are already visited (dead end), 
        # allow going back so training doesn't break out or crash.
        if np.sum(valid_mask) == 0:
            for neighbor in self.G.neighbors(self.current_node):
                valid_mask[neighbor] = 1.0
                
        return valid_mask

    def step(self, action):
        next_node = action
        
        if not self.G.has_edge(self.current_node, next_node):
            reward = -100.0
            done = True
            return self._get_obs(), reward, done, False, {"status": "invalid_hop"}
        
        delay = self.link_delays[self.current_node][next_node]
        self.current_node = next_node
        self.visited_nodes.add(self.current_node)
        
        reward = -delay
        
        if self.current_node == self.dst:
            reward += 100.0  
            done = True
            return self._get_obs(), reward, done, False, {"status": "success"}
        
        if np.random.rand() < 0.2:
            self.link_delays += np.random.uniform(-2.0, 2.0, size=(self.num_nodes, self.num_nodes))
            self.link_delays = np.clip(self.link_delays, 1.0, 100.0)

        done = False
        return self._get_obs(), reward, done, False, {}