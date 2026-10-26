import torch
import numpy as np
from model import QNetwork

class RoutingInference:
    def __init__(self, model_path="routing_dqn.pt", num_nodes=6):
        self.num_nodes = num_nodes
        state_dim = 2 + (num_nodes * num_nodes)
        action_dim = num_nodes
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.q_net.load_state_dict(torch.load(model_path))
        self.q_net.eval()

    def get_next_hop(self, current_node, target_node, link_delay_matrix, valid_neighbors=None, visited_nodes=None):
        """
        Accepts state telemetry and returns recommended next hop node safely.
        Returns None if a loop or dead-end trap is detected.
        """
        matrix_flat = np.array(link_delay_matrix, dtype=np.float32).flatten()
        state = np.concatenate(([current_node, target_node], matrix_flat)).astype(np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0).numpy()
            
            if valid_neighbors is not None:
                mask = np.zeros(self.num_nodes)
                for n in valid_neighbors:
                    # Filter out already visited nodes to prevent loops
                    if visited_nodes is not None and n in visited_nodes:
                        continue
                    mask[n] = 1.0
                
                # If all neighbors are visited, report a trap/dead-end explicitly by returning None
                if np.sum(mask) == 0:
                    return None
                        
                q_values[mask == 0.0] = -1e9
                
            next_hop = int(np.argmax(q_values))
            
        return next_hop

if __name__ == "__main__":
    engine = RoutingInference()
    mock_delays = np.random.uniform(5.0, 50.0, size=(6, 6))
    next_node = engine.get_next_hop(current_node=0, target_node=5, link_delay_matrix=mock_delays, valid_neighbors=[1, 2], visited_nodes={0})
    print(f"Predicted next hop from Node 0 to Node 5 is: Node {next_node}")