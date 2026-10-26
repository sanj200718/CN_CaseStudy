"""Inference wrapper consumed by controller.py and the evaluation scripts."""
from pathlib import Path
import numpy as np
import torch
from model import QNetwork


class RoutingInference:
    def __init__(self, model_path="routing_dqn.pt", num_nodes=6):
        self.num_nodes = int(num_nodes)
        self.model_path = Path(model_path)
        state_dim = 2 + (self.num_nodes * self.num_nodes)
        action_dim = self.num_nodes
        self.q_net = QNetwork(state_dim, action_dim)
        # map_location makes the checkpoint portable across CPU/GPU machines.
        try:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with PyTorch versions before weights_only
            state_dict = torch.load(self.model_path, map_location="cpu")
        self.q_net.load_state_dict(state_dict)
        self.q_net.eval()

    def get_next_hop(self, current_node, target_node, link_delay_matrix,
                     valid_neighbors=None, visited_nodes=None):
        """Return a next-hop node, or None when no permitted neighbor exists.

        valid_neighbors must represent links that are physically/currently
        available. If visited_nodes is supplied, previously visited nodes are
        excluded. This method intentionally does not silently re-enable them.
        """
        current_node, target_node = int(current_node), int(target_node)
        matrix = np.asarray(link_delay_matrix, dtype=np.float32)
        expected_shape = (self.num_nodes, self.num_nodes)
        if matrix.shape != expected_shape:
            raise ValueError(f"link_delay_matrix must have shape {expected_shape}, got {matrix.shape}")
        if not (0 <= current_node < self.num_nodes and 0 <= target_node < self.num_nodes):
            raise ValueError("current_node and target_node must be valid node IDs")
        if current_node == target_node:
            return target_node

        state = np.concatenate(([current_node, target_node], matrix.flatten())).astype(np.float32)
        with torch.no_grad():
            q_values = self.q_net(torch.from_numpy(state).unsqueeze(0)).squeeze(0).cpu().numpy()

        if valid_neighbors is not None:
            visited = set(int(n) for n in (visited_nodes or ()))
            candidates = sorted({int(n) for n in valid_neighbors
                                 if 0 <= int(n) < self.num_nodes and int(n) not in visited})
            if not candidates:
                return None
            masked = np.full(self.num_nodes, -np.inf, dtype=np.float32)
            masked[candidates] = q_values[candidates]
            q_values = masked
        elif visited_nodes is not None:
            # Without a topology-provided neighbor list, the model cannot know
            # which actions are connected; refuse visited candidates only.
            visited = {int(n) for n in visited_nodes if 0 <= int(n) < self.num_nodes}
            for n in visited:
                q_values[n] = -np.inf

        if not np.isfinite(q_values).any():
            return None
        next_hop = int(np.argmax(q_values))
        if not np.isfinite(q_values[next_hop]):
            return None
        return next_hop


if __name__ == "__main__":
    model_path = Path(__file__).with_name("routing_dqn.pt")
    engine = RoutingInference(model_path=model_path, num_nodes=6)
    matrix = np.full((6, 6), 50.0, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    matrix[0, 3] = matrix[3, 0] = 10.0
    matrix[3, 5] = matrix[5, 3] = 15.0
    print("Inference module ready; example next hop:",
          engine.get_next_hop(0, 5, matrix, valid_neighbors=[2, 3, 4], visited_nodes=[0]))
