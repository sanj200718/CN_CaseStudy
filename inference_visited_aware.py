"""Inference wrapper for the 44-feature visited-aware DQN candidate."""
from pathlib import Path
import numpy as np
import torch
from model import QNetwork


class VisitedAwareRoutingInference:
    def __init__(self, model_path="routing_dqn_visited_candidate.pt", num_nodes=6):
        self.num_nodes = int(num_nodes)
        self.model_path = Path(model_path)
        state_dim = 2 + self.num_nodes * self.num_nodes + self.num_nodes
        self.q_net = QNetwork(state_dim, self.num_nodes)
        try:
            weights = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(self.model_path, map_location="cpu")
        self.q_net.load_state_dict(weights)
        self.q_net.eval()

    def get_next_hop(self, current_node, target_node, link_delay_matrix,
                     valid_neighbors=None, visited_nodes=None):
        current_node = int(current_node)
        target_node = int(target_node)
        matrix = np.asarray(link_delay_matrix, dtype=np.float32)
        expected = (self.num_nodes, self.num_nodes)
        if matrix.shape != expected:
            raise ValueError(f"link_delay_matrix must have shape {expected}, got {matrix.shape}")
        if not (0 <= current_node < self.num_nodes and 0 <= target_node < self.num_nodes):
            raise ValueError("Node IDs are outside the configured topology")
        if current_node == target_node:
            return target_node

        visited = set(int(n) for n in (visited_nodes or ()))
        visited.add(current_node)
        visited_mask = np.zeros(self.num_nodes, dtype=np.float32)
        for node in visited:
            if 0 <= node < self.num_nodes:
                visited_mask[node] = 1.0

        state = np.concatenate(
            ([current_node, target_node], matrix.flatten(), visited_mask)
        ).astype(np.float32)
        with torch.no_grad():
            q_values = self.q_net(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()

        if valid_neighbors is None:
            candidates = [n for n in range(self.num_nodes) if n not in visited]
        else:
            candidates = sorted({
                int(n) for n in valid_neighbors
                if 0 <= int(n) < self.num_nodes and int(n) not in visited
            })
        if not candidates:
            return None

        masked = np.full(self.num_nodes, -np.inf, dtype=np.float32)
        masked[candidates] = q_values[candidates]
        if not np.isfinite(masked).any():
            return None
        return int(np.argmax(masked))
