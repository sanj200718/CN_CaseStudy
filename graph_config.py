"""
Shared network definition. topo.py AND the Ryu controller import this,
so the Mininet network always matches Sanjana's training graph.

Graph = nx.erdos_renyi_graph(n=6, p=0.6, seed=42) from env.py.
Model node i  <->  Mininet switch s(i+1)  <->  OpenFlow dpid i+1
Source host h1 is on node 0, destination host h2 is on node 5.
"""

NUM_NODES = 6
SRC_NODE = 0
DST_NODE = 5

# (node_a, node_b): one-way link delay in ms (kept inside the 5-50 ms range the DQN trained on)
BASE_DELAYS_MS = {
    (0, 2): 20, (0, 3): 10, (0, 4): 15,
    (1, 4): 10, (1, 5): 25,
    (2, 3): 10, (2, 4): 15, (2, 5): 30,
    (3, 4): 10, (3, 5): 15,
}
EDGES = sorted(BASE_DELAYS_MS)

LINK_BW_MBPS = 10      # inter-switch link bandwidth
HOST_LINK_BW = 100     # host-switch link bandwidth

H1_IP, H1_MAC = "10.0.0.1", "00:00:00:00:00:01"
H2_IP, H2_MAC = "10.0.0.2", "00:00:00:00:00:02"

CONTROLLER_IP, CONTROLLER_PORT = "127.0.0.1", 6653


def node_to_dpid(node):
    return node + 1


def dpid_to_node(dpid):
    return dpid - 1
