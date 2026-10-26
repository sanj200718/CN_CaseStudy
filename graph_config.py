"""Shared graph/topology definition for training, controller, and experiments.

Node IDs used by the DQN are 0..5. The Mininet switch for node n is s(n+1),
with OpenFlow datapath ID n+1. The graph is undirected.
"""
from __future__ import annotations

NUM_NODES = 6
SRC_NODE = 0
DST_NODE = 5

# Keep insertion order stable: topo.py uses explicit switch-port assignments
# derived from this order, so controller forwarding rules match Mininet ports.
BASE_DELAYS_MS = {
    (0, 2): 20,
    (0, 3): 10,
    (0, 4): 15,
    (1, 4): 10,
    (1, 5): 25,
    (2, 3): 10,
    (2, 4): 15,
    (2, 5): 30,
    (3, 4): 10,
    (3, 5): 15,
}
EDGES = list(BASE_DELAYS_MS.keys())

LINK_BW_MBPS = 10
HOST_LINK_BW = 100
HOST_PORT = 1

H1_IP, H1_MAC = "10.0.0.1", "00:00:00:00:00:01"
H2_IP, H2_MAC = "10.0.0.2", "00:00:00:00:00:02"
HOST_IP_TO_NODE = {H1_IP: SRC_NODE, H2_IP: DST_NODE}
HOST_NODE_TO_IP = {SRC_NODE: H1_IP, DST_NODE: H2_IP}
HOST_IP_TO_MAC = {H1_IP: H1_MAC, H2_IP: H2_MAC}
CONTROLLER_IP, CONTROLLER_PORT = "127.0.0.1", 6653
REST_HOST, REST_PORT = "127.0.0.1", 8080


def node_to_dpid(node: int) -> int:
    return int(node) + 1


def dpid_to_node(dpid: int) -> int:
    return int(dpid) - 1


def switch_name(node: int) -> str:
    return f"s{int(node) + 1}"


def normalized_edge(u: int, v: int) -> tuple[int, int]:
    return (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))


def _build_port_maps():
    """Derive deterministic switch-port mappings from topology creation order.

    Switches 0 and 5 have a host attached on port 1. Other switches begin at
    port 1. Inter-switch links then consume ports in BASE_DELAYS_MS insertion
    order. topo.py uses the same map explicitly when constructing each link.
    """
    next_port = {node: (2 if node in (SRC_NODE, DST_NODE) else 1)
                 for node in range(NUM_NODES)}
    edge_ports = {}
    port_to_neighbor = {node: {} for node in range(NUM_NODES)}
    for (u, v) in EDGES:
        pu, pv = next_port[u], next_port[v]
        edge = normalized_edge(u, v)
        # Keep ports oriented to the sorted edge tuple for lookup simplicity.
        if edge == (u, v):
            edge_ports[edge] = (pu, pv)
        else:
            edge_ports[edge] = (pv, pu)
        port_to_neighbor[u][pu] = v
        port_to_neighbor[v][pv] = u
        next_port[u] += 1
        next_port[v] += 1
    return edge_ports, port_to_neighbor


EDGE_PORTS, PORT_TO_NEIGHBOR = _build_port_maps()


def port_for_neighbor(node: int, neighbor: int) -> int:
    node, neighbor = int(node), int(neighbor)
    for port, other in PORT_TO_NEIGHBOR[node].items():
        if other == neighbor:
            return port
    raise KeyError(f"No configured port from node {node} to neighbor {neighbor}")


def ports_for_edge(u: int, v: int) -> tuple[int, int]:
    """Return (port_on_u, port_on_v) for the undirected edge u-v."""
    edge = normalized_edge(u, v)
    pu, pv = EDGE_PORTS[edge]
    return (pu, pv) if edge == (int(u), int(v)) else (pv, pu)


def edge_for_port(node: int, port: int):
    """Return the neighbor for an inter-switch port, else None."""
    return PORT_TO_NEIGHBOR.get(int(node), {}).get(int(port))
