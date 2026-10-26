"""Gymnasium environment for learning next-hop routing on the project's
fixed six-node topology.

The graph is imported from graph_config.py so training and the Mininet/Ryu
controller use the same links. Each reset randomizes distinct source and
destination nodes, as well as symmetric delays on existing links. Non-edges
retain the controller's default delay value and are never treated as links.
"""

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

import graph_config as cfg


class DynamicRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_nodes=6, src=0, dst=5, max_steps=None):
        super().__init__()
        self.num_nodes = int(num_nodes)
        if self.num_nodes != cfg.NUM_NODES:
            raise ValueError(
                f"This project environment uses the configured {cfg.NUM_NODES}-node topology; "
                f"received num_nodes={self.num_nodes}."
            )

        self.src = int(src)
        self.dst = int(dst)
        self.max_steps = int(max_steps if max_steps is not None else self.num_nodes * 2)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(2 + self.num_nodes * self.num_nodes,),
            dtype=np.float32,
        )

        # Use the exact fixed topology shared with the controller and Mininet.
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_nodes))
        self.G.add_edges_from(cfg.EDGES)
        if not nx.is_connected(self.G):
            raise ValueError("The topology in graph_config.py must be connected")

        self.link_delays = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.current_node = self.src
        self.visited_nodes = {self.current_node}
        self.steps = 0
        self.reset()

    def _sample_link_delays(self):
        """Return a symmetric delay matrix; non-edges use default delay 50 ms."""
        matrix = np.full(
            (self.num_nodes, self.num_nodes), 50.0, dtype=np.float32
        )
        np.fill_diagonal(matrix, 0.0)

        for u, v in self.G.edges():
            delay = float(self.np_random.uniform(5.0, 50.0))
            matrix[u, v] = delay
            matrix[v, u] = delay

        return matrix

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Select distinct endpoints using Gymnasium's seeded RNG.
        self.src = int(self.np_random.integers(0, self.num_nodes))
        destinations = [node for node in range(self.num_nodes) if node != self.src]
        self.dst = int(self.np_random.choice(destinations))

        self.link_delays = self._sample_link_delays()
        self.current_node = self.src
        self.visited_nodes = {self.current_node}
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate(
            ([self.current_node, self.dst], self.link_delays.flatten())
        ).astype(np.float32)

    def get_valid_actions(self):
        """Return a binary mask for connected neighbors not yet visited."""
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        for neighbor in self.G.neighbors(self.current_node):
            if neighbor not in self.visited_nodes:
                mask[neighbor] = 1.0
        return mask

    def step(self, action):
        self.steps += 1
        next_node = int(action)

        # Reject out-of-range, disconnected, or revisited hops.
        if (
            not 0 <= next_node < self.num_nodes
            or not self.G.has_edge(self.current_node, next_node)
            or next_node in self.visited_nodes
        ):
            return (
                self._get_obs(),
                -100.0,
                True,
                False,
                {"status": "invalid_or_revisited_hop"},
            )

        delay = float(self.link_delays[self.current_node, next_node])
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        reward = -delay

        if self.current_node == self.dst:
            return (
                self._get_obs(),
                reward + 100.0,
                True,
                False,
                {"status": "success"},
            )

        # Apply small symmetric delay drift to existing links only.
        if self.np_random.random() < 0.2:
            for u, v in self.G.edges():
                drift = float(self.np_random.uniform(-2.0, 2.0))
                updated_delay = float(
                    np.clip(self.link_delays[u, v] + drift, 1.0, 100.0)
                )
                self.link_delays[u, v] = updated_delay
                self.link_delays[v, u] = updated_delay

        if self.steps >= self.max_steps:
            return (
                self._get_obs(),
                reward - 25.0,
                False,
                True,
                {"status": "step_limit_reached"},
            )

        if not np.any(self.get_valid_actions()):
            return (
                self._get_obs(),
                reward - 25.0,
                True,
                False,
                {"status": "dead_end"},
            )

        return self._get_obs(), reward, False, False, {"status": "continue"}
