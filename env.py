"""Gymnasium environment for learning next-hop routing on a six-node graph.

This training simulator intentionally remains separate from the Mininet runtime.
It supplies randomized, symmetric link-delay snapshots and avoids selecting a
node already visited in the current episode. The checked-in checkpoint is kept
unchanged; rerun train.py only when intentionally producing a new checkpoint.
"""
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np


class DynamicRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_nodes=6, src=0, dst=5, max_steps=None):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.src = int(src)
        self.dst = int(dst)
        self.max_steps = int(max_steps or (self.num_nodes * 2))
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(2 + self.num_nodes * self.num_nodes,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Fixed graph seed keeps the graph consistent with graph_config.py.
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.6, seed=42)
        if not nx.is_connected(self.G):
            # Defensive fallback; fixed seed graph should be connected.
            rng = self.np_random
            while not nx.is_connected(self.G):
                self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.6,
                                               seed=int(rng.integers(0, 2**31 - 1)))
        raw = self.np_random.uniform(5.0, 50.0, size=(self.num_nodes, self.num_nodes))
        self.link_delays = ((raw + raw.T) / 2.0).astype(np.float32)
        np.fill_diagonal(self.link_delays, 0.0)
        self.current_node = self.src
        self.visited_nodes = {self.current_node}
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate(([self.current_node, self.dst], self.link_delays.flatten())).astype(np.float32)

    def get_valid_actions(self):
        """Binary mask of currently connected neighbours not visited this episode.

        An all-zero mask is a terminal dead end. The environment does not
        silently allow a previously visited node as a fallback action.
        """
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        for neighbor in self.G.neighbors(self.current_node):
            if neighbor not in self.visited_nodes:
                mask[neighbor] = 1.0
        return mask

    def step(self, action):
        self.steps += 1
        next_node = int(action)
        if (next_node not in self.G.neighbors(self.current_node)
                or next_node in self.visited_nodes):
            return self._get_obs(), -100.0, True, False, {"status": "invalid_or_revisited_hop"}

        delay = float(self.link_delays[self.current_node][next_node])
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        reward = -delay
        if self.current_node == self.dst:
            return self._get_obs(), reward + 100.0, True, False, {"status": "success"}

        # Apply small symmetric delay drift with 20% probability.
        if self.np_random.random() < 0.2:
            noise = self.np_random.uniform(-2.0, 2.0, size=(self.num_nodes, self.num_nodes))
            noise = (noise + noise.T) / 2.0
            self.link_delays = np.clip(self.link_delays + noise, 1.0, 100.0).astype(np.float32)
            np.fill_diagonal(self.link_delays, 0.0)

        if self.steps >= self.max_steps:
            return self._get_obs(), reward - 25.0, True, False, {"status": "step_limit_reached"}
        if not np.any(self.get_valid_actions()):
            return self._get_obs(), reward - 25.0, True, False, {"status": "dead_end"}
        return self._get_obs(), reward, False, False, {"status": "continue"}
