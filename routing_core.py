"""Controller-independent routing helpers, unit-testable without Ryu/Mininet."""
from __future__ import annotations
import math
import networkx as nx
import numpy as np

VALID_MODES = {"ospf", "dijkstra", "dqn"}


def normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {"hop_count": "ospf", "ospf_style": "ospf", "shortest_path": "dijkstra"}
    value = aliases.get(value, value)
    if value not in VALID_MODES:
        raise ValueError(f"Unsupported routing mode {mode!r}; choose one of {sorted(VALID_MODES)}")
    return value


def build_delay_matrix(num_nodes, edge_delays, active_edges, default_delay=50.0):
    """Create an NxN symmetric delay matrix; non-edges remain a safe default."""
    matrix = np.full((num_nodes, num_nodes), float(default_delay), dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    for edge, delay in edge_delays.items():
        u, v = edge
        if tuple(sorted((u, v))) in active_edges:
            matrix[u, v] = matrix[v, u] = float(delay)
    return matrix


def make_graph(num_nodes, edge_delays, active_edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for edge, delay in edge_delays.items():
        normalized = tuple(sorted(edge))
        if normalized in active_edges:
            u, v = normalized
            if math.isfinite(float(delay)) and float(delay) > 0:
                graph.add_edge(u, v, delay=float(delay), weight=float(delay))
    return graph


def compute_route(graph, source, target, mode="dijkstra", inference_engine=None,
                  delay_matrix=None, dqn_fallback=True, max_hops=None):
    """Return a transparent route result.

    Result keys: path, success, failure_reason, used_fallback, fallback_reason,
    route_cost_ms, route_mode. In DQN mode, inference is asked for a next hop
    with valid neighbours and visited-node history. A fallback is explicit.
    """
    mode = normalize_mode(mode)
    source, target = int(source), int(target)
    result = {
        "path": [], "success": False, "failure_reason": None,
        "used_fallback": False, "fallback_reason": None,
        "route_cost_ms": float("nan"), "route_mode": mode,
    }
    if source not in graph or target not in graph:
        result["failure_reason"] = "unknown_endpoint"
        return result
    if source == target:
        result.update(path=[source], success=True, route_cost_ms=0.0)
        return result
    if not nx.has_path(graph, source, target):
        result["failure_reason"] = "no_path_available"
        return result

    if mode in ("dijkstra", "ospf"):
        weight = "delay" if mode == "dijkstra" else None
        try:
            path = nx.shortest_path(graph, source=source, target=target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            result["failure_reason"] = "no_path_available"
            return result
        result.update(path=path, success=True,
                      route_cost_ms=sum(float(graph[u][v]["delay"]) for u, v in zip(path, path[1:])))
        return result

    if inference_engine is None:
        result["failure_reason"] = "dqn_model_unavailable"
        return result
    if delay_matrix is None:
        delay_matrix = np.full((len(graph), len(graph)), 50.0, dtype=np.float32)
        np.fill_diagonal(delay_matrix, 0.0)
        for u, v, data in graph.edges(data=True):
            delay_matrix[u, v] = delay_matrix[v, u] = float(data["delay"])

    limit = max_hops or max(1, len(graph) * 2)
    path = [source]
    visited = {source}
    current = source
    dead_end_reason = None
    while current != target and len(path) <= limit:
        neighbors = sorted(graph.neighbors(current))
        try:
            candidate = inference_engine.get_next_hop(
                current_node=current,
                target_node=target,
                link_delay_matrix=delay_matrix,
                valid_neighbors=neighbors,
                visited_nodes=visited,
            )
        except TypeError as exc:
            result["failure_reason"] = f"inference_interface_error: {exc}"
            result["path"] = path
            return result
        if candidate is None:
            dead_end_reason = "no_valid_next_hop"
            break
        candidate = int(candidate)
        if candidate not in graph.neighbors(current):
            dead_end_reason = "invalid_next_hop"
            break
        if candidate in visited:
            dead_end_reason = "routing_loop_detected"
            break
        path.append(candidate)
        visited.add(candidate)
        current = candidate

    if current == target:
        result.update(path=path, success=True,
                      route_cost_ms=sum(float(graph[u][v]["delay"]) for u, v in zip(path, path[1:])))
        return result
    if dead_end_reason is None:
        dead_end_reason = "step_limit_reached"

    result["path"] = path
    if dqn_fallback:
        try:
            # A fallback replaces the failed DQN prefix. Appending a path from
            # the dead-end to the destination could revisit an earlier node and
            # create a route that looks connected but contains a loop.
            fallback_path = nx.shortest_path(graph, source=source, target=target, weight="delay")
            result.update(
                path=fallback_path, success=True, used_fallback=True,
                fallback_reason=dead_end_reason, failure_reason=None,
                route_cost_ms=sum(float(graph[u][v]["delay"]) for u, v in zip(fallback_path, fallback_path[1:])),
            )
            return result
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    result["failure_reason"] = dead_end_reason
    return result
