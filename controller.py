#!/usr/bin/env python3
"""Ryu/OpenFlow 1.3 controller for the Group 4 intelligent-routing project.

Modes:
  ospf     OSPF-style hop-count shortest path (not a full OSPF protocol)
  dijkstra Live configured-delay-weighted shortest path
  dqn      DQN next-hop routing with explicit Dijkstra fallback on dead ends

REST API (default http://127.0.0.1:8080):
  GET  /routing/status
  GET  /routing/metrics
  GET  /routing/decisions?limit=50
  POST /routing/mode          {"mode":"dijkstra|ospf|dqn"}
  POST /routing/link-state    {"u":0,"v":3,"up":false}
  POST /routing/link-delay    {"u":0,"v":3,"delay_ms":35}
  POST /routing/reset         {"mode":"dijkstra"}  (mode optional)
  POST /routing/metrics/reset {}

Link-delay values are controller telemetry/configuration values. They are not
measured automatically from packets; the experiment runner updates them when
it changes a link's configured Mininet delay.
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import networkx as nx
import numpy as np
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib.packet import packet, ethernet, ipv4, ether_types
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import ControllerBase, WSGIApplication, route, Response

import graph_config as cfg
from inference import RoutingInference
from routing_core import build_delay_matrix, make_graph, compute_route, normalize_mode


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "routing_dqn.pt"
DECISION_LOG_PATH = ROOT / "results" / "controller_decisions.csv"
CSV_FIELDS = [
    "timestamp", "mode", "current_node", "target_node", "src_ip", "dst_ip",
    "path", "route_cost_ms", "success", "failure_reason", "used_fallback",
    "fallback_reason", "packet_in_port",
]


class RoutingRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.routing_app = data["routing_app"]

    @staticmethod
    def _response(payload, status=200):
        body = json.dumps(payload, allow_nan=False, default=str).encode("utf-8")
        return Response(status=status, content_type="application/json", body=body)

    @staticmethod
    def _body(req):
        try:
            raw = req.body.decode("utf-8") if req.body else "{}"
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value
        except Exception as exc:
            raise ValueError(f"Invalid JSON request body: {exc}") from exc

    @route("routing", "/routing/status", methods=["GET"])
    def status(self, req, **kwargs):
        return self._response(self.routing_app.status_payload())

    @route("routing", "/routing/metrics", methods=["GET"])
    def metrics(self, req, **kwargs):
        return self._response(self.routing_app.metrics_payload())

    @route("routing", "/routing/decisions", methods=["GET"])
    def decisions(self, req, **kwargs):
        try:
            limit = max(1, min(200, int(req.GET.get("limit", "50"))))
        except ValueError:
            return self._response({"error": "limit must be an integer"}, 400)
        return self._response({"decisions": list(self.routing_app.recent_decisions)[-limit:]})

    @route("routing", "/routing/mode", methods=["POST"])
    def set_mode(self, req, **kwargs):
        try:
            body = self._body(req)
            mode = self.routing_app.set_mode(body.get("mode", ""))
            return self._response({"ok": True, "mode": mode})
        except Exception as exc:
            return self._response({"error": str(exc)}, 400)

    @route("routing", "/routing/link-state", methods=["POST"])
    def set_link_state(self, req, **kwargs):
        try:
            body = self._body(req)
            u, v = int(body["u"]), int(body["v"])
            up = body["up"]
            if not isinstance(up, bool):
                raise ValueError("'up' must be true or false")
            result = self.routing_app.set_link_state(u, v, up)
            return self._response({"ok": True, **result})
        except Exception as exc:
            return self._response({"error": str(exc)}, 400)

    @route("routing", "/routing/link-delay", methods=["POST"])
    def set_link_delay(self, req, **kwargs):
        try:
            body = self._body(req)
            result = self.routing_app.set_link_delay(int(body["u"]), int(body["v"]), float(body["delay_ms"]))
            return self._response({"ok": True, **result})
        except Exception as exc:
            return self._response({"error": str(exc)}, 400)

    @route("routing", "/routing/reset", methods=["POST"])
    def reset(self, req, **kwargs):
        try:
            body = self._body(req)
            payload = self.routing_app.reset_network_state(body.get("mode"))
            return self._response({"ok": True, **payload})
        except Exception as exc:
            return self._response({"error": str(exc)}, 400)

    @route("routing", "/routing/metrics/reset", methods=["POST"])
    def reset_metrics(self, req, **kwargs):
        self.routing_app.reset_metrics()
        return self._response({"ok": True, "metrics": self.routing_app.metrics_payload()})

    @route("routing", "/routing/clear-flows", methods=["POST"])
    def clear_flows(self, req, **kwargs):
        self.routing_app.flush_ipv4_flows()
        return self._response({"ok": True, "message": "IPv4 flows flushed"})


class IntelligentRoutingController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs["wsgi"]
        wsgi.register(RoutingRestController, {"routing_app": self})

        self.mode = "dijkstra"
        self.datapaths = {}
        self.edge_delays = {cfg.normalized_edge(u, v): float(delay)
                            for (u, v), delay in cfg.BASE_DELAYS_MS.items()}
        self.active_edges = set(self.edge_delays)
        self.inference = None
        self._load_model()
        self.metrics = Counter()
        self.mode_counts = Counter()
        self.recent_decisions = deque(maxlen=500)
        self.last_route = None
        self._log_lock = threading.Lock()
        self._prepare_log()
        self.logger.info("IntelligentRoutingController initialized mode=%s model_loaded=%s", self.mode, self.inference is not None)

    def _load_model(self):
        try:
            if DEFAULT_MODEL_PATH.exists():
                self.inference = RoutingInference(model_path=str(DEFAULT_MODEL_PATH), num_nodes=cfg.NUM_NODES)
                self.logger.info("Loaded DQN model from %s", DEFAULT_MODEL_PATH)
            else:
                self.logger.warning("Model file %s not found; DQN mode will report unavailable", DEFAULT_MODEL_PATH)
        except Exception:
            self.inference = None
            self.logger.exception("Could not load DQN model; DQN mode will be unavailable")

    def _prepare_log(self):
        DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not DECISION_LOG_PATH.exists() or DECISION_LOG_PATH.stat().st_size == 0:
            with DECISION_LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()

    def set_mode(self, mode):
        mode = normalize_mode(mode)
        if mode == "dqn" and self.inference is None:
            raise RuntimeError("DQN mode requested but the trained model could not be loaded")
        self.mode = mode
        self.flush_ipv4_flows()
        self.logger.info("Routing mode changed to %s; IPv4 flows flushed", mode)
        return self.mode

    def reset_network_state(self, mode=None):
        self.active_edges = set(self.edge_delays)
        self.edge_delays = {cfg.normalized_edge(u, v): float(delay)
                            for (u, v), delay in cfg.BASE_DELAYS_MS.items()}
        if mode is not None:
            self.mode = normalize_mode(mode)
        self.flush_ipv4_flows()
        return {"mode": self.mode, "active_edges": [list(e) for e in sorted(self.active_edges)],
                "edge_delays_ms": {f"{u}-{v}": d for (u, v), d in self.edge_delays.items()}}

    def set_link_state(self, u, v, up):
        edge = cfg.normalized_edge(u, v)
        if edge not in self.edge_delays:
            raise ValueError(f"Edge {edge} is not present in the configured topology")
        if up:
            self.active_edges.add(edge)
        else:
            self.active_edges.discard(edge)
        self.flush_ipv4_flows()
        return {"edge": list(edge), "up": bool(up), "active_edges": [list(e) for e in sorted(self.active_edges)]}

    def set_link_delay(self, u, v, delay_ms):
        edge = cfg.normalized_edge(u, v)
        if edge not in self.edge_delays:
            raise ValueError(f"Edge {edge} is not present in the configured topology")
        if not math.isfinite(delay_ms) or not (1.0 <= delay_ms <= 1000.0):
            raise ValueError("delay_ms must be finite and between 1 and 1000")
        self.edge_delays[edge] = float(delay_ms)
        self.flush_ipv4_flows()
        return {"edge": list(edge), "delay_ms": float(delay_ms)}

    def reset_metrics(self):
        self.metrics.clear()
        self.mode_counts.clear()
        self.last_route = None

    def _graph(self):
        return make_graph(cfg.NUM_NODES, self.edge_delays, self.active_edges)

    def _delay_matrix(self):
        return build_delay_matrix(cfg.NUM_NODES, self.edge_delays, self.active_edges, default_delay=50.0)

    def _choose_route(self, source, target):
        graph = self._graph()
        matrix = self._delay_matrix()
        return compute_route(graph, source, target, mode=self.mode,
                             inference_engine=self.inference, delay_matrix=matrix,
                             dqn_fallback=True, max_hops=cfg.NUM_NODES * 2)

    def _record_decision(self, record):
        record = dict(record)
        record["timestamp"] = time.time()
        if not isinstance(record.get("route_cost_ms"), (int, float)) or not math.isfinite(float(record["route_cost_ms"])):
            record["route_cost_ms"] = ""
        else:
            record["route_cost_ms"] = round(float(record["route_cost_ms"]), 4)
        record["path"] = "->".join(str(n) for n in record.get("path", []))
        for field in CSV_FIELDS:
            record.setdefault(field, "")
        clean = {field: record.get(field, "") for field in CSV_FIELDS}
        with self._log_lock:
            with DECISION_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(clean)
        self.recent_decisions.append(clean)
        self.last_route = clean
        self.metrics["routing_decisions"] += 1
        self.mode_counts[clean.get("mode") or self.mode] += 1
        if clean.get("used_fallback") in (True, "True", "true", 1, "1"):
            self.metrics["dqn_fallbacks"] += 1
        if not (clean.get("success") in (True, "True", "true", 1, "1")):
            self.metrics["failed_routes"] += 1

    def status_payload(self):
        return {
            "controller": "ready",
            "mode": self.mode,
            "model_loaded": self.inference is not None,
            "connected_switches": sorted(cfg.dpid_to_node(dpid) for dpid in self.datapaths),
            "expected_switches": cfg.NUM_NODES,
            "active_edges": [list(e) for e in sorted(self.active_edges)],
            "edge_delays_ms": {f"{u}-{v}": d for (u, v), d in sorted(self.edge_delays.items())},
            "last_route": self.last_route,
            "metrics": self.metrics_payload(),
        }

    def metrics_payload(self):
        return {"counters": dict(self.metrics), "mode_counts": dict(self.mode_counts),
                "last_route": self.last_route}

    def flush_ipv4_flows(self):
        for dp in list(self.datapaths.values()):
            ofp, parser = dp.ofproto, dp.ofproto_parser
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP)
            mod = parser.OFPFlowMod(
                datapath=dp, command=ofp.OFPFC_DELETE,
                out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                match=match,
            )
            dp.send_msg(mod)

    def _add_flow(self, datapath, priority, match, actions, idle_timeout=15, hard_timeout=90):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        instructions = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=instructions, idle_timeout=idle_timeout,
                                hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    def _install_path(self, current_node, path, in_port, src_ip, dst_ip):
        if not path:
            return None
        dst_node = path[-1]
        packet_output_port = None
        for index, node in enumerate(path):
            dpid = cfg.node_to_dpid(node)
            dp = self.datapaths.get(dpid)
            if dp is None:
                continue
            ofp, parser = dp.ofproto, dp.ofproto_parser
            if index == 0:
                in_port_here = int(in_port)
            else:
                in_port_here = cfg.port_for_neighbor(node, path[index - 1])
            if index < len(path) - 1:
                out_port = cfg.port_for_neighbor(node, path[index + 1])
            else:
                out_port = cfg.HOST_PORT
            match = parser.OFPMatch(in_port=in_port_here,
                                    eth_type=ether_types.ETH_TYPE_IP,
                                    ipv4_src=src_ip, ipv4_dst=dst_ip)
            self._add_flow(dp, 100, match, [parser.OFPActionOutput(out_port)])
            if node == current_node:
                packet_output_port = out_port
        return packet_output_port

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self._add_flow(dp, 0, match, actions, idle_timeout=0, hard_timeout=0)
        self.logger.info("Installed table-miss rule for switch node=%s dpid=%s", cfg.dpid_to_node(dp.id), dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        msg = ev.msg
        node = cfg.dpid_to_node(msg.datapath.id)
        if node < 0 or node >= cfg.NUM_NODES:
            return
        neighbor = cfg.edge_for_port(node, msg.desc.port_no)
        if neighbor is None:
            return
        edge = cfg.normalized_edge(node, neighbor)
        ofp = msg.datapath.ofproto
        if msg.reason == ofp.OFPPR_DELETE or (msg.desc.state & ofp.OFPPS_LINK_DOWN):
            self.active_edges.discard(edge)
            self.flush_ipv4_flows()
            self.logger.warning("OpenFlow port down: edge=%s", edge)
        else:
            # An up notification marks the edge up; the scenario runner also
            # updates the REST state so the physical and controller state agree.
            self.active_edges.add(edge)
            self.flush_ipv4_flows()
            self.logger.info("OpenFlow port up: edge=%s", edge)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        node = cfg.dpid_to_node(dp.id)
        if node < 0 or node >= cfg.NUM_NODES:
            return
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        ip = pkt.get_protocol(ipv4.ipv4)
        # The topology preconfigures ARP to prevent broadcast storms. We route IPv4 only.
        if ip is None:
            return
        if ip.dst not in cfg.HOST_IP_TO_NODE:
            return
        target = cfg.HOST_IP_TO_NODE[ip.dst]
        in_port = msg.match.get("in_port")
        if in_port is None:
            self.metrics["failed_routes"] += 1
            return

        route_result = self._choose_route(node, target)
        rec = {
            "mode": self.mode, "current_node": node, "target_node": target,
            "src_ip": ip.src, "dst_ip": ip.dst, "path": route_result.get("path", []),
            "route_cost_ms": route_result.get("route_cost_ms"),
            "success": route_result.get("success", False),
            "failure_reason": route_result.get("failure_reason") or "",
            "used_fallback": route_result.get("used_fallback", False),
            "fallback_reason": route_result.get("fallback_reason") or "",
            "packet_in_port": in_port,
        }
        self._record_decision(rec)
        if not route_result.get("success"):
            self.logger.warning("No route for %s -> %s mode=%s reason=%s", ip.src, ip.dst, self.mode,
                                route_result.get("failure_reason"))
            return

        path = route_result["path"]
        output_port = self._install_path(node, path, in_port, ip.src, ip.dst)
        if output_port is None:
            self.logger.warning("Path calculated but no output port available at node %s; switches_connected=%s", node,
                                sorted(cfg.dpid_to_node(dpid) for dpid in self.datapaths))
            return
        actions = [dp.ofproto_parser.OFPActionOutput(output_port)]
        data = None if msg.buffer_id != dp.ofproto.OFP_NO_BUFFER else msg.data
        out = dp.ofproto_parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
            actions=actions, data=data,
        )
        dp.send_msg(out)
        self.metrics["packet_ins_forwarded"] += 1
        self.logger.info("route mode=%s %s->%s path=%s cost=%s fallback=%s", self.mode, ip.src, ip.dst,
                         path, route_result.get("route_cost_ms"), route_result.get("used_fallback"))


if __name__ == "__main__":
    print("Load this module with: ryu-manager controller.py")
