import unittest
import networkx as nx
import numpy as np
import graph_config as cfg
from experiments.metric_parsers import parse_ping, parse_iperf_udp
from routing_core import make_graph, build_delay_matrix, compute_route, normalize_mode


class GreedyTestAgent:
    """Small deterministic stand-in for the trained model in routing-core tests."""
    def __init__(self, preference=None):
        self.preference = preference or {}

    def get_next_hop(self, current_node, target_node, link_delay_matrix,
                     valid_neighbors=None, visited_nodes=None):
        candidates = list(valid_neighbors or [])
        visited = set(visited_nodes or [])
        candidates = [n for n in candidates if n not in visited]
        if not candidates:
            return None
        wanted = self.preference.get(current_node)
        return wanted if wanted in candidates else candidates[0]


class NoHopAgent:
    def get_next_hop(self, current_node, target_node, link_delay_matrix,
                     valid_neighbors=None, visited_nodes=None):
        return None


class RoutingCoreTests(unittest.TestCase):
    def setUp(self):
        self.delays = {cfg.normalized_edge(u, v): float(d) for (u, v), d in cfg.BASE_DELAYS_MS.items()}
        self.active = set(self.delays)
        self.graph = make_graph(cfg.NUM_NODES, self.delays, self.active)
        self.matrix = build_delay_matrix(cfg.NUM_NODES, self.delays, self.active)

    def test_dijkstra_uses_delay_weight(self):
        result = compute_route(self.graph, 0, 5, mode="dijkstra")
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], [0, 3, 5])
        self.assertAlmostEqual(result["route_cost_ms"], 25.0)

    def test_ospf_style_uses_hop_count(self):
        result = compute_route(self.graph, 0, 5, mode="ospf")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["path"]) - 1, 2)

    def test_double_dqn_mode_alias_normalizes(self):
        self.assertEqual(normalize_mode("dqn_double"), "dqn_double")
        self.assertEqual(normalize_mode("double_dqn"), "dqn_double")
        self.assertEqual(normalize_mode("visited_aware_double_dqn"), "dqn_double")

    def test_double_dqn_mode_uses_next_hop_interface(self):
        agent = GreedyTestAgent({0: 2, 2: 4, 4: 1, 1: 5})
        result = compute_route(self.graph, 0, 5, mode="dqn_double", inference_engine=agent,
                               delay_matrix=self.matrix, dqn_fallback=False)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], [0, 2, 4, 1, 5])
        self.assertEqual(result["route_mode"], "dqn_double")

    def test_dqn_passes_visited_nodes_and_reaches_destination(self):
        agent = GreedyTestAgent({0: 2, 2: 4, 4: 1, 1: 5})
        result = compute_route(self.graph, 0, 5, mode="dqn", inference_engine=agent,
                               delay_matrix=self.matrix, dqn_fallback=False)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], [0, 2, 4, 1, 5])
        self.assertEqual(len(result["path"]), len(set(result["path"])))

    def test_failed_link_is_not_used(self):
        active = set(self.active)
        active.remove(cfg.normalized_edge(0, 3))
        graph = make_graph(cfg.NUM_NODES, self.delays, active)
        result = compute_route(graph, 0, 5, mode="dijkstra")
        self.assertTrue(result["success"])
        self.assertNotIn((0, 3), [cfg.normalized_edge(a, b) for a, b in zip(result["path"], result["path"][1:])])

    def test_dqn_dead_end_is_reported_or_fallback_is_explicit(self):
        agent = NoHopAgent()
        result = compute_route(self.graph, 0, 5, mode="dqn", inference_engine=agent,
                               delay_matrix=self.matrix, dqn_fallback=True)
        self.assertTrue(result["success"])
        self.assertTrue(result["used_fallback"])
        self.assertIsNotNone(result["fallback_reason"])
        no_fallback = compute_route(self.graph, 0, 5, mode="dqn", inference_engine=agent,
                                    delay_matrix=self.matrix, dqn_fallback=False)
        self.assertFalse(no_fallback["success"])
        self.assertEqual(no_fallback["failure_reason"], "no_valid_next_hop")

    def test_ping_parser_reads_loss_and_rtt(self):
        output = """8 packets transmitted, 7 received, 12.5% packet loss, time 1400ms
rtt min/avg/max/mdev = 1.000/2.500/5.000/1.200 ms"""
        values = parse_ping(output)
        self.assertEqual(values["ping_sent"], 8)
        self.assertEqual(values["ping_received"], 7)
        self.assertAlmostEqual(values["packet_loss_pct"], 12.5)
        self.assertAlmostEqual(values["rtt_avg_ms"], 2.5)

    def test_iperf2_udp_parser_reads_throughput_jitter_and_loss(self):
        output = "[  3] 0.0-3.0 sec 3.11 MBytes 8.70 Mbits/sec 0.037 ms 7/2219 (0.32%)"
        values = parse_iperf_udp(output)
        self.assertAlmostEqual(values["udp_rx_mbps"], 8.70)
        self.assertAlmostEqual(values["udp_jitter_ms"], 0.037)
        self.assertAlmostEqual(values["udp_loss_pct"], 100 * 7 / 2219)

    def test_every_edge_port_is_unique_per_switch(self):
        for node, mapping in cfg.PORT_TO_NEIGHBOR.items():
            ports = list(mapping)
            self.assertEqual(len(ports), len(set(ports)))
            self.assertNotIn(cfg.HOST_PORT, ports if node in (cfg.SRC_NODE, cfg.DST_NODE) else [])
        for (u, v) in cfg.EDGES:
            pu, pv = cfg.ports_for_edge(u, v)
            self.assertEqual(cfg.edge_for_port(u, pu), v)
            self.assertEqual(cfg.edge_for_port(v, pv), u)


if __name__ == "__main__":
    unittest.main(verbosity=2)
