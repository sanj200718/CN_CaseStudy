"""Paired dynamic-routing evaluation with state-aware cycle detection.

This is a SIMULATED route-cost experiment, not a packet-level Mininet test.
Both algorithms receive the same precomputed network-state timeline. A revisit
to a node after network conditions change is allowed; a cycle is flagged only
when the same current node and the same network state repeat. DQN also receives
visited-node history as expected by its inference policy.

Run from the repository root:
    python evaluation/run_dynamic_experiments_state_cycle.py --runs 2
    python evaluation/run_dynamic_experiments_state_cycle.py --runs 30

Outputs are written to evaluation/results_dynamic_state_cycle/.
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
BASE_SEED = 20261016
MAX_HOPS = NUM_NODES * 3
RESULTS_DIR = Path(__file__).resolve().parent / "results_dynamic_state_cycle"
MODEL_PATH = ROOT / "routing_dqn.pt"
SCENARIOS = ["dynamic_delay_spike", "dynamic_link_failure", "gradual_delay_drift"]


def build_base_graph() -> nx.Graph:
    """Use the same fixed six-node topology as the existing evaluator/env."""
    graph = nx.erdos_renyi_graph(n=NUM_NODES, p=0.6, seed=42)
    fallback_seed = 43
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(n=NUM_NODES, p=0.6, seed=fallback_seed)
        fallback_seed += 1
    return graph


def make_delay_matrix(rng: np.random.Generator) -> np.ndarray:
    """Create simulated directional per-link delay costs in milliseconds."""
    return rng.uniform(5.0, 50.0, size=(NUM_NODES, NUM_NODES)).astype(np.float32)


def weighted_dijkstra_path(
    graph: nx.Graph, delays: np.ndarray, source: int, target: int
) -> list[int]:
    """Return a minimum-cost path for the current network state."""
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        directed.add_edge(u, v, weight=float(delays[u, v]))
        directed.add_edge(v, u, weight=float(delays[v, u]))
    return nx.shortest_path(directed, source=source, target=target, weight="weight")


def removable_future_edge(graph: nx.Graph, path: list[int]) -> tuple[int, int]:
    """Choose a future-path edge whose removal leaves the graph connected."""
    future_edges = list(zip(path[1:-1], path[2:]))
    for u, v in future_edges:
        candidate = graph.copy()
        candidate.remove_edge(u, v)
        if nx.is_connected(candidate):
            return min(u, v), max(u, v)

    # Fallback: any non-bridge edge that does not correspond to the first hop.
    first_hop = (min(path[0], path[1]), max(path[0], path[1])) if len(path) > 1 else None
    for u, v in sorted((min(a, b), max(a, b)) for a, b in graph.edges()):
        if (u, v) == first_hop:
            continue
        candidate = graph.copy()
        candidate.remove_edge(u, v)
        if nx.is_connected(candidate):
            return u, v
    raise RuntimeError("Could not find a removable edge that preserves connectivity.")


def precompute_timeline(
    base_graph: nx.Graph,
    base_delays: np.ndarray,
    scenario: str,
    seed: int,
) -> tuple[list[tuple[nx.Graph, np.ndarray]], str]:
    """Precompute state snapshots; the timeline is independent of the algorithm."""
    # Identify an edge on the initial weighted-Dijkstra route that is still ahead
    # after the first hop. Both algorithms receive the same network change.
    initial_path = weighted_dijkstra_path(base_graph, base_delays, SOURCE, DESTINATION)
    timeline: list[tuple[nx.Graph, np.ndarray]] = []
    detail = "initial_state_unchanged_at_step_0"
    event_edge: tuple[int, int] | None = None

    if scenario in {"dynamic_delay_spike", "dynamic_link_failure"}:
        event_edge = removable_future_edge(base_graph, initial_path)
        u, v = event_edge
        if scenario == "dynamic_delay_spike":
            detail = f"after_first_hop_add_{25}ms_to_link_{u}-{v}"
        else:
            detail = f"after_first_hop_fail_link_{u}-{v}"

    drift_rng = np.random.default_rng(seed + 900_000)
    current_delays = base_delays.copy()
    for step in range(MAX_HOPS):
        graph = base_graph.copy()
        delays = base_delays.copy()

        if step >= 1 and scenario == "dynamic_delay_spike":
            assert event_edge is not None
            u, v = event_edge
            delays[u, v] = min(float(delays[u, v]) + 25.0, 100.0)
            delays[v, u] = min(float(delays[v, u]) + 25.0, 100.0)
        elif step >= 1 and scenario == "dynamic_link_failure":
            assert event_edge is not None
            graph.remove_edge(*event_edge)
        elif step >= 1 and scenario == "gradual_delay_drift":
            # Apply a seeded, shared drift process only to links that exist.
            for u, v in graph.edges():
                current_delays[u, v] = np.clip(
                    current_delays[u, v] + drift_rng.uniform(-4.0, 4.0), 1.0, 100.0
                )
                current_delays[v, u] = np.clip(
                    current_delays[v, u] + drift_rng.uniform(-4.0, 4.0), 1.0, 100.0
                )
            delays = current_delays.copy()
        elif scenario not in SCENARIOS:
            raise ValueError(f"Unknown dynamic scenario: {scenario}")

        timeline.append((graph, delays))

    if scenario == "gradual_delay_drift":
        detail = "after_first_hop_seeded_per_link_delay_drift_each_decision"
    return timeline, detail


def simulate_route(
    engine: RoutingInference,
    algorithm: str,
    timeline: list[tuple[nx.Graph, np.ndarray]],
) -> dict[str, Any]:
    """Run a next-hop policy over the shared timeline; DQN receives route history.

    A None action is recorded as no_valid_next_hop. A repeated node is recorded
    as routing_loop_detected. Results are simulated costs, not packet measurements.
    """
    current = SOURCE
    path = [SOURCE]
    total_cost = 0.0
    route_valid = True
    reached = False
    failure_reason = ""
    seen_network_states: set[tuple[int, tuple[tuple[int, int], ...], bytes]] = set()

    for step, (graph, delays) in enumerate(timeline):
        if current == DESTINATION:
            reached = True
            break

        # A node revisit is not automatically a routing loop in a dynamic
        # network: the topology or delay matrix may have changed since the
        # previous visit. Flag a cycle only when the same node and same network
        # state recur. This avoids counting a legitimate post-failure backtrack
        # as an immediate loop.
        edge_signature = tuple(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()))
        delay_signature = np.asarray(delays, dtype=np.float32).tobytes()
        state_key = (int(current), edge_signature, delay_signature)
        if state_key in seen_network_states:
            route_valid = False
            failure_reason = "routing_state_cycle_detected"
            break
        seen_network_states.add(state_key)

        neighbors = sorted(int(node) for node in graph.neighbors(current))
        if not neighbors:
            route_valid = False
            failure_reason = "no_valid_neighbor"
            break

        try:
            if algorithm == "Dijkstra":
                best_path = weighted_dijkstra_path(graph, delays, current, DESTINATION)
                if len(best_path) < 2:
                    reached = current == DESTINATION
                    break
                next_node = int(best_path[1])
            elif algorithm == "DQN":
                # Pass the route history so inference can filter already visited nodes.
                proposed_next = engine.get_next_hop(
                    current_node=current,
                    target_node=DESTINATION,
                    link_delay_matrix=delays,
                    valid_neighbors=neighbors,
                    visited_nodes=path,
                )
                if proposed_next is None:
                    failure_reason = (
                        "routing_loop_prevented_no_unvisited_neighbor"
                        if neighbors and all(node in path for node in neighbors)
                        else "no_valid_next_hop"
                    )
                    break
                next_node = int(proposed_next)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            failure_reason = "no_path"
            break

        # DQN's inference policy is specifically configured to avoid revisiting
        # path nodes. Keep a defensive check for DQN. Dijkstra may legitimately
        # backtrack to a previously visited node once the network changes; its
        # cycles are detected using repeated (node, network-state) signatures.
        if algorithm == "DQN" and next_node in path:
            route_valid = False
            failure_reason = "routing_loop_detected"
            break

        # Guard the experiment against invalid action implementations.
        if next_node not in neighbors:
            route_valid = False
            failure_reason = "invalid_hop"
            break

        total_cost += float(delays[current, next_node])
        path.append(next_node)
        current = next_node

        if current == DESTINATION:
            reached = True
            break
    else:
        failure_reason = "step_limit_reached"

    if not reached and not failure_reason:
        failure_reason = "destination_not_reached"

    return {
        "route": "->".join(map(str, path)),
        "route_valid": route_valid,
        "destination_reached": reached,
        "route_delay_cost_ms": round(total_cost, 6),
        "hop_count": len(path) - 1,
        "failure_reason": "" if reached else failure_reason,
    }


def evaluate_one(
    engine: RoutingInference,
    scenario: str,
    run_number: int,
    seed: int,
    base_graph: nx.Graph,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base_delays = make_delay_matrix(rng)
    timeline, detail = precompute_timeline(base_graph, base_delays, scenario, seed)
    records = []

    for algorithm in ("Dijkstra", "DQN"):
        outcome = simulate_route(engine, algorithm, timeline)
        records.append(
            {
                "scenario": scenario,
                "scenario_detail": detail,
                "run": run_number,
                "seed": seed,
                "algorithm": algorithm,
                "source": SOURCE,
                "destination": DESTINATION,
                **outcome,
            }
        )
    return records


def summarize(raw_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        by_alg: dict[str, list[dict[str, Any]]] = {
            alg: [r for r in raw_records if r["scenario"] == scenario and r["algorithm"] == alg]
            for alg in ("Dijkstra", "DQN")
        }
        means: dict[str, float] = {}
        success_rates: dict[str, float] = {}
        for algorithm, rows in by_alg.items():
            successes = [r for r in rows if r["destination_reached"]]
            costs = [float(r["route_delay_cost_ms"]) for r in successes]
            success_rate = 100.0 * len(successes) / len(rows) if rows else 0.0
            success_rates[algorithm] = success_rate
            means[algorithm] = float(np.mean(costs)) if costs else float("nan")
            summary.append(
                {
                    "scenario": scenario,
                    "algorithm": algorithm,
                    "total_runs": len(rows),
                    "destinations_reached": len(successes),
                    "success_rate_pct": round(success_rate, 4),
                    "mean_route_delay_cost_ms_successes_only": round(means[algorithm], 6) if costs else "",
                    "std_route_delay_cost_ms_successes_only": round(float(np.std(costs, ddof=1)), 6) if len(costs) > 1 else (0.0 if costs else ""),
                    "mean_hops_successes_only": round(float(np.mean([r["hop_count"] for r in successes])), 4) if successes else "",
                }
            )

        base, dqn = means["Dijkstra"], means["DQN"]
        delay_change = ((dqn - base) / base * 100.0) if np.isfinite(base) and base != 0 and np.isfinite(dqn) else float("nan")
        improvements.append(
            {
                "scenario": scenario,
                "dijkstra_mean_route_delay_cost_ms": round(base, 6) if np.isfinite(base) else "",
                "dqn_mean_route_delay_cost_ms": round(dqn, 6) if np.isfinite(dqn) else "",
                "dqn_delay_change_pct_vs_dijkstra": round(delay_change, 4) if np.isfinite(delay_change) else "",
                "dqn_success_rate_pct": round(success_rates["DQN"], 4),
                "dijkstra_success_rate_pct": round(success_rates["Dijkstra"], 4),
                "success_rate_difference_dqn_minus_dijkstra_pp": round(success_rates["DQN"] - success_rates["Dijkstra"], 4),
            }
        )
    return summary, improvements


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = SCENARIOS
    algorithms = ["Dijkstra", "DQN"]
    x = np.arange(len(scenarios))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, algorithm in enumerate(algorithms):
        rows = {r["scenario"]: r for r in summary if r["algorithm"] == algorithm}
        vals = [float(rows[s]["mean_route_delay_cost_ms_successes_only"]) if rows[s]["mean_route_delay_cost_ms_successes_only"] != "" else np.nan for s in scenarios]
        ax.bar(x + (i - 0.5) * width, vals, width, label=algorithm)
    ax.set_title("Mean Accumulated Simulated Route Cost by Dynamic Scenario")
    ax.set_ylabel("Accumulated route cost (simulated ms; successful runs only)")
    ax.set_xlabel("Dynamic scenario")
    ax.set_xticks(x, [s.replace("_", "\n") for s in scenarios])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Routing method")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_dynamic_route_cost.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, algorithm in enumerate(algorithms):
        rows = {r["scenario"]: r for r in summary if r["algorithm"] == algorithm}
        vals = [float(rows[s]["success_rate_pct"]) for s in scenarios]
        ax.bar(x + (i - 0.5) * width, vals, width, label=algorithm)
    ax.set_title("Destination Success Rate by Dynamic Scenario")
    ax.set_ylabel("Runs reaching destination (%)")
    ax.set_xlabel("Dynamic scenario")
    ax.set_ylim(0, 105)
    ax.set_xticks(x, [s.replace("_", "\n") for s in scenarios])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Routing method")
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_success_rate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Runs per scenario (default: 30).")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = RoutingInference(model_path=str(MODEL_PATH), num_nodes=NUM_NODES)
    engine.q_net.eval()
    graph = build_base_graph()
    all_records: list[dict[str, Any]] = []

    print(f"Loaded trained model: {MODEL_PATH.name}")
    print(f"Topology: {NUM_NODES} nodes, {graph.number_of_edges()} undirected links")
    print(f"Runs per scenario: {args.runs}; max hops per route: {MAX_HOPS}")
    print("Both algorithms use the same precomputed network-state timeline for each paired run.")
    print("DQN inference is given visited-node history.")
    print("A revisit is only called a loop when the same node and identical network state repeat.")
    print("These are simulated route-cost results, not real packet-level measurements.\n")

    for scenario_index, scenario in enumerate(SCENARIOS):
        for run_number in range(1, args.runs + 1):
            seed = args.seed + scenario_index * 100_000 + run_number
            all_records.extend(evaluate_one(engine, scenario, run_number, seed, graph))
        print(f"Completed: {scenario} ({args.runs} paired runs)")

    summary, improvements = summarize(all_records)
    write_csv(RESULTS_DIR / "raw_dynamic_results.csv", all_records)
    write_csv(RESULTS_DIR / "summary_dynamic.csv", summary)
    write_csv(RESULTS_DIR / "improvement_dynamic.csv", improvements)
    make_plots(summary, RESULTS_DIR)

    print(f"\nSaved raw records, summary, comparisons, and graphs under: {RESULTS_DIR}")
    print("Files: raw_dynamic_results.csv, summary_dynamic.csv, improvement_dynamic.csv,")
    print("       mean_dynamic_route_cost.png, dynamic_success_rate.png")


if __name__ == "__main__":
    main()
