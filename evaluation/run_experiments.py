"""Repeated, snapshot-matched DQN vs. weighted-Dijkstra route evaluation.

This evaluates route choice in the repository's six-node *simulated* graph.
It does not measure actual packet latency, throughput, or packet loss.

Run from the repository root:
    python evaluation/run_experiments.py --runs 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import RoutingInference  # noqa: E402

NUM_NODES = 6
SOURCE = 0
DESTINATION = 5
DEFAULT_RUNS = 30
BASE_SEED = 20261013
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.csv"
MODEL_PATH = ROOT / "routing_dqn.pt"


def build_base_graph() -> nx.Graph:
    """Mirror the connected six-node topology produced by the current env.py."""
    graph = nx.erdos_renyi_graph(n=NUM_NODES, p=0.6, seed=42)
    # Keep the same fallback strategy as env.py in case the selected graph changes.
    fallback_seed = 43
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(n=NUM_NODES, p=0.6, seed=fallback_seed)
        fallback_seed += 1
    return graph


def make_delay_matrix(rng: np.random.Generator) -> np.ndarray:
    """Create directional costs in ms, matching env.py's input scale."""
    return rng.uniform(5.0, 50.0, size=(NUM_NODES, NUM_NODES)).astype(np.float32)


def choose_removable_edge(graph: nx.Graph) -> tuple[int, int]:
    """Choose a deterministic edge whose removal keeps the test graph connected."""
    for u, v in sorted((min(a, b), max(a, b)) for a, b in graph.edges()):
        candidate = graph.copy()
        candidate.remove_edge(u, v)
        if nx.is_connected(candidate):
            return u, v
    raise RuntimeError("No single edge can be removed while keeping the graph connected.")


def apply_scenario(
    base_graph: nx.Graph,
    base_delays: np.ndarray,
    scenario: str,
) -> tuple[nx.Graph, np.ndarray, str]:
    """Apply a controlled condition to a copy of a delay snapshot/topology.

    The selected penalties are simulation settings, not measured physical values.
    A bandwidth-reduction case is represented by a delay penalty on one link;
    it must not be described as a real throughput test.
    """
    graph = base_graph.copy()
    delays = base_delays.copy()
    detail = "none"

    if scenario == "normal":
        pass
    elif scenario == "congested_link":
        u, v = 2, 5
        if graph.has_edge(u, v):
            delays[u, v] = min(float(delays[u, v]) + 25.0, 100.0)
            delays[v, u] = min(float(delays[v, u]) + 25.0, 100.0)
            detail = f"delay_penalty_on_link_{u}-{v}_plus_25ms"
        else:
            raise RuntimeError(f"Expected test edge {u}-{v} is missing from base topology.")
    elif scenario == "high_delay_alternate_path":
        penalized = [(2, 3), (3, 5)]
        for u, v in penalized:
            if graph.has_edge(u, v):
                delays[u, v] = min(float(delays[u, v]) + 20.0, 100.0)
                delays[v, u] = min(float(delays[v, u]) + 20.0, 100.0)
        detail = "delay_penalty_on_links_2-3_and_3-5_plus_20ms"
    elif scenario == "reduced_bandwidth_proxy":
        # In this simulator, bandwidth is not an explicit state variable.
        # This scenario is only a delay-cost proxy for a degraded link.
        u, v = 0, 4
        if graph.has_edge(u, v):
            delays[u, v] = min(float(delays[u, v]) + 30.0, 100.0)
            delays[v, u] = min(float(delays[v, u]) + 30.0, 100.0)
            detail = f"simulated_degraded_link_{u}-{v}_delay_proxy_plus_30ms"
        else:
            raise RuntimeError(f"Expected test edge {u}-{v} is missing from base topology.")
    elif scenario == "link_failure":
        u, v = choose_removable_edge(graph)
        graph.remove_edge(u, v)
        detail = f"failed_link_{u}-{v}"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return graph, delays, detail


def weighted_dijkstra_path(
    graph: nx.Graph,
    delays: np.ndarray,
    source: int,
    target: int,
) -> list[int]:
    """Find the minimum-cost route using the same directional delay snapshot."""
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        directed.add_edge(u, v, weight=float(delays[u, v]))
        directed.add_edge(v, u, weight=float(delays[v, u]))
    return nx.shortest_path(directed, source=source, target=target, weight="weight")


def measure_path(path: list[int], graph: nx.Graph, delays: np.ndarray) -> tuple[bool, float, int]:
    """Measure path cost and verify every hop against the scenario topology."""
    if not path or path[0] != SOURCE or path[-1] != DESTINATION:
        return False, float("nan"), max(0, len(path) - 1)
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return False, float("nan"), len(path) - 1
        total += float(delays[u, v])
    return True, total, len(path) - 1


def dqn_path(
    engine: RoutingInference,
    graph: nx.Graph,
    delays: np.ndarray,
    source: int,
    target: int,
) -> tuple[list[int], bool, str]:
    """Roll out the trained policy on a fixed snapshot, detecting loops safely."""
    path = [source]
    current = source
    visited_edges: set[tuple[int, int]] = set()

    for _ in range(NUM_NODES * 3):
        if current == target:
            return path, True, "success"

        neighbors = sorted(graph.neighbors(current))
        if not neighbors:
            return path, False, "no_valid_neighbor"

        proposed_next = engine.get_next_hop(
            current_node=current,
            target_node=target,
            link_delay_matrix=delays,
            valid_neighbors=neighbors,
            visited_nodes=path,
        )
        if proposed_next is None:
            return path, False, "no_valid_next_hop"
        next_node = int(proposed_next)
        if next_node in path:
            path.append(next_node)
            return path, False, "routing_loop_detected"
        if next_node not in neighbors:
            return path, False, "invalid_hop"

        edge = (current, next_node)
        if edge in visited_edges or next_node in path:
            path.append(next_node)
            return path, False, "loop_detected"

        visited_edges.add(edge)
        path.append(next_node)
        current = next_node

    return path, current == target, "step_limit_reached"


def evaluate_one(
    engine: RoutingInference,
    scenario: str,
    run_number: int,
    seed: int,
    base_graph: nx.Graph,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base_delays = make_delay_matrix(rng)
    graph, delays, scenario_detail = apply_scenario(base_graph, base_delays, scenario)

    records: list[dict[str, Any]] = []

    try:
        path = weighted_dijkstra_path(graph, delays, SOURCE, DESTINATION)
        valid, cost, hops = measure_path(path, graph, delays)
        records.append(
            {
                "scenario": scenario,
                "scenario_detail": scenario_detail,
                "run": run_number,
                "seed": seed,
                "algorithm": "Dijkstra",
                "source": SOURCE,
                "destination": DESTINATION,
                "route": "->".join(map(str, path)),
                "route_valid": valid,
                "route_delay_ms": round(cost, 6) if np.isfinite(cost) else "",
                "hop_count": hops,
                "failure_reason": "" if valid else "invalid_dijkstra_path",
            }
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        records.append(
            {
                "scenario": scenario,
                "scenario_detail": scenario_detail,
                "run": run_number,
                "seed": seed,
                "algorithm": "Dijkstra",
                "source": SOURCE,
                "destination": DESTINATION,
                "route": "",
                "route_valid": False,
                "route_delay_ms": "",
                "hop_count": 0,
                "failure_reason": type(exc).__name__,
            }
        )

    path, reached, reason = dqn_path(engine, graph, delays, SOURCE, DESTINATION)
    valid, cost, hops = measure_path(path, graph, delays)
    valid = bool(valid and reached)
    records.append(
        {
            "scenario": scenario,
            "scenario_detail": scenario_detail,
            "run": run_number,
            "seed": seed,
            "algorithm": "DQN",
            "source": SOURCE,
            "destination": DESTINATION,
            "route": "->".join(map(str, path)),
            "route_valid": valid,
            "route_delay_ms": round(cost, 6) if np.isfinite(cost) else "",
            "hop_count": hops,
            "failure_reason": "" if valid else reason,
        }
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Runs per scenario (default: 30).")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["normal", "congested_link", "high_delay_alternate_path", "reduced_bandwidth_proxy", "link_failure"],
        choices=["normal", "congested_link", "high_delay_alternate_path", "reduced_bandwidth_proxy", "link_failure"],
        help="Scenarios to run.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = RoutingInference(model_path=str(MODEL_PATH), num_nodes=NUM_NODES)
    engine.q_net.eval()
    base_graph = build_base_graph()
    all_records: list[dict[str, Any]] = []

    print(f"Loaded trained model: {MODEL_PATH.name}")
    print(f"Topology: {NUM_NODES} nodes, {base_graph.number_of_edges()} undirected links")
    print(f"Runs per scenario: {args.runs}")
    print("Note: results are simulated route-delay/cost measurements, not real network traffic metrics.\n")

    for scenario_index, scenario in enumerate(args.scenarios):
        for run_number in range(1, args.runs + 1):
            seed = args.seed + scenario_index * 100_000 + run_number
            all_records.extend(evaluate_one(engine, scenario, run_number, seed, base_graph))
        print(f"Completed: {scenario} ({args.runs} runs per routing method)")

    fieldnames = [
        "scenario", "scenario_detail", "run", "seed", "algorithm", "source", "destination",
        "route", "route_valid", "route_delay_ms", "hop_count", "failure_reason",
    ]
    with RAW_RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nSaved {len(all_records)} result rows to: {RAW_RESULTS_PATH}")
    print("Next: python evaluation/analyze_results.py")


if __name__ == "__main__":
    main()
