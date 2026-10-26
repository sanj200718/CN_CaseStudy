"""Fair, deterministic-snapshot demo comparing the DQN to weighted Dijkstra."""
from pathlib import Path
import networkx as nx
import numpy as np
from env import DynamicRoutingEnv
from inference import RoutingInference


def main():
    env = DynamicRoutingEnv(num_nodes=6, src=0, dst=5)
    # Lock a single topology and delay matrix for both methods.
    graph = env.G.copy()
    delays = env.link_delays.copy()
    for u, v in graph.edges:
        graph[u][v]["weight"] = float(delays[u][v])
        graph[u][v]["delay"] = float(delays[u][v])

    dijkstra_path = nx.shortest_path(graph, source=env.src, target=env.dst, weight="weight")
    dijkstra_cost = sum(graph[u][v]["delay"] for u, v in zip(dijkstra_path, dijkstra_path[1:]))

    agent = RoutingInference(model_path=Path(__file__).with_name("routing_dqn.pt"), num_nodes=6)
    path = [env.src]
    visited = {env.src}
    current = env.src
    while current != env.dst and len(path) <= env.num_nodes:
        neighbors = list(graph.neighbors(current))
        nxt = agent.get_next_hop(current, env.dst, delays,
                                 valid_neighbors=neighbors, visited_nodes=visited)
        if nxt is None:
            break
        if nxt not in graph.neighbors(current) or nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    dqn_success = current == env.dst
    dqn_cost = sum(float(delays[u][v]) for u, v in zip(path, path[1:])) if dqn_success else float("nan")

    print("=== Same-snapshot routing comparison ===")
    print(f"Topology edges: {graph.number_of_edges()}")
    print(f"Dijkstra: path={dijkstra_path}, simulated route cost={dijkstra_cost:.2f}")
    print(f"DQN:      path={path}, success={dqn_success}, simulated route cost={dqn_cost:.2f}")
    print("Note: these are simulated route costs, not packet-level network measurements.")


if __name__ == "__main__":
    main()
