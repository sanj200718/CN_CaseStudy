"""Smoke-test that the saved 44-feature candidate loads and selects a valid hop."""
from pathlib import Path
import numpy as np
import graph_config as cfg
from inference_visited_aware import VisitedAwareRoutingInference

model_path = Path(__file__).resolve().parent / "routing_dqn_visited_double_candidate.pt"
if not model_path.exists():
    raise FileNotFoundError(
        f"Missing {model_path.name}. Copy the trained candidate checkpoint into the project root."
    )

agent = VisitedAwareRoutingInference(model_path=str(model_path), num_nodes=cfg.NUM_NODES)
matrix = np.full((cfg.NUM_NODES, cfg.NUM_NODES), 50.0, dtype=np.float32)
np.fill_diagonal(matrix, 0.0)
for (u, v), delay in cfg.BASE_DELAYS_MS.items():
    matrix[u, v] = matrix[v, u] = float(delay)

next_hop = agent.get_next_hop(
    current_node=0,
    target_node=5,
    link_delay_matrix=matrix,
    valid_neighbors=[2, 3, 4],
    visited_nodes={0},
)
if next_hop not in {2, 3, 4}:
    raise AssertionError(f"Candidate returned invalid next hop: {next_hop!r}")

print("Double DQN checkpoint loaded successfully.")
print(f"Example next hop from node 0 to node 5: {next_hop}")
print("Smoke test passed.")
