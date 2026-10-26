#!/usr/bin/env python3
"""Run packet-level Mininet experiments against a separately running Ryu app.

Run this script inside the same WSL distro as Mininet, with sudo. Ryu should
already be running in another terminal. Results are written to a timestamped
CSV under results/ and never overwrite previous runs.

Examples:
  sudo python3 experiments/run_network_experiments.py --scenario baseline --mode dijkstra --runs 1
  sudo python3 experiments/run_network_experiments.py --scenario all --mode all --runs 3 --duration 4
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import select
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import graph_config as cfg
from topo import build, configure_static_arp
from experiments.metric_parsers import parse_ping, parse_iperf_udp

API_BASE = f"http://{cfg.REST_HOST}:{cfg.REST_PORT}"
MODES = ("dijkstra", "ospf", "dqn")
SCENARIOS = ("baseline", "delay_increase", "bandwidth_drop", "congestion",
             "link_failure_recovery", "changing_traffic")
CSV_FIELDS = [
    "timestamp", "scenario", "mode", "run", "traffic_phase", "source_ip", "destination_ip",
    "offered_udp_mbps", "changed_link", "configured_delay_ms", "configured_bandwidth_mbps",
    "ping_sent", "ping_received", "packet_loss_pct", "rtt_min_ms", "rtt_avg_ms",
    "rtt_max_ms", "rtt_stddev_ms", "udp_rx_mbps", "udp_jitter_ms", "udp_loss_pct",
    "recovery_ms", "recovery_detected", "controller_fallbacks", "controller_failures",
    "forwarding_decisions", "last_route", "notes",
]


def api_request(path, payload=None, timeout=3.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            result = json.loads(body) if body else {}
            if response.status >= 400:
                raise RuntimeError(f"Controller API returned HTTP {response.status}: {result}")
            return result
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Controller API returned HTTP {exc.code} for {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ryu REST API at {API_BASE}. Start the controller first. Details: {exc}"
        ) from exc


def ensure_controller():
    status = api_request("/routing/status")
    if status.get("controller") != "ready":
        raise RuntimeError(f"Controller did not report ready: {status}")
    return status


def set_link_tc(link, bandwidth_mbps, delay_ms):
    """Apply symmetric TCLink configuration to both egress interfaces."""
    for intf in (link.intf1, link.intf2):
        intf.config(bw=float(bandwidth_mbps), delay=f"{float(delay_ms):g}ms", loss=0)


def reset_network(net):
    # Raise links before resetting qdisc configuration.
    for u, v in cfg.EDGES:
        net.configLinkStatus(cfg.switch_name(u), cfg.switch_name(v), "up")
    for edge, link in net._routing_inter_switch_links.items():
        set_link_tc(link, cfg.LINK_BW_MBPS, cfg.BASE_DELAYS_MS[edge])
    api_request("/routing/reset", {"mode": "dijkstra"})
    api_request("/routing/metrics/reset", {})


def configure_scenario(net, scenario, run_number):
    """Return fields describing the scenario and the offered UDP rate."""
    details = {"changed_link": "", "configured_delay_ms": "", "configured_bandwidth_mbps": "",
               "offered_udp_mbps": 4.0, "notes": ""}
    if scenario == "baseline":
        details["notes"] = "Configured baseline topology; UDP probe uses a 4 Mbps offered rate."
    elif scenario == "delay_increase":
        edge = cfg.normalized_edge(0, 3)
        delay = float(cfg.BASE_DELAYS_MS[edge] + 25)
        set_link_tc(net._routing_inter_switch_links[edge], cfg.LINK_BW_MBPS, delay)
        api_request("/routing/link-delay", {"u": edge[0], "v": edge[1], "delay_ms": delay})
        details.update(changed_link=f"{edge[0]}-{edge[1]}", configured_delay_ms=delay,
                       configured_bandwidth_mbps=cfg.LINK_BW_MBPS, offered_udp_mbps=4.0,
                       notes="Delay raised symmetrically in Mininet and controller telemetry.")
    elif scenario == "bandwidth_drop":
        edge = cfg.normalized_edge(3, 5)
        bw = 2.0
        set_link_tc(net._routing_inter_switch_links[edge], bw, cfg.BASE_DELAYS_MS[edge])
        details.update(changed_link=f"{edge[0]}-{edge[1]}", configured_delay_ms=cfg.BASE_DELAYS_MS[edge],
                       configured_bandwidth_mbps=bw, offered_udp_mbps=8.0,
                       notes="Physical capacity reduced; routing costs remain delay-based because DQN has no bandwidth input.")
    elif scenario == "congestion":
        details.update(offered_udp_mbps=9.0,
                       notes="Near-capacity UDP offered load over the 10 Mbps inter-switch links; ping runs concurrently.")
    elif scenario == "changing_traffic":
        details.update(offered_udp_mbps=2.0,
                       notes="Offered UDP rate changes within each run across 2, 6, 9 and 12 Mbps phases.")
    elif scenario == "link_failure_recovery":
        details.update(offered_udp_mbps=4.0,
                       notes="Failure is applied to the first inter-switch edge on the current selected route; link is restored after failover measurement.")
    return details


def get_forward_route():
    payload = api_request("/routing/decisions?limit=100")
    decisions = payload.get("decisions", [])
    for item in reversed(decisions):
        if item.get("src_ip") == cfg.H1_IP and item.get("dst_ip") == cfg.H2_IP and item.get("success") in (True, "True", "true", "1"):
            value = item.get("path", "")
            try:
                return [int(part) for part in str(value).split("->") if part != ""]
            except ValueError:
                return []
    return []


def _drain_ready(pipe):
    while True:
        ready, _, _ = select.select([pipe], [], [], 0)
        if not ready:
            return
        line = pipe.readline()
        if not line:
            return


def measure_link_failure_recovery(net, h1, h2, mode, timeout_s=6.0):
    # Warm up forwarding state and obtain the actual forward route in this mode.
    h1.cmd(f"ping -n -c 3 -W 1 {h2.IP()}")
    route = get_forward_route()
    route_edge = None
    for u, v in zip(route, route[1:]):
        edge = cfg.normalized_edge(u, v)
        if edge in cfg.BASE_DELAYS_MS:
            route_edge = edge
            break
    if route_edge is None:
        # The route may not be logged yet; choose a well-known edge on the default route.
        route_edge = cfg.normalized_edge(0, 3)

    ping_proc = h1.popen(["ping", "-D", "-n", "-i", "0.2", "-w", str(int(timeout_s + 4)), h2.IP()],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    recovery_ms = float("nan")
    detected = False
    try:
        time.sleep(0.6)
        _drain_ready(ping_proc.stdout)
        down_wall = time.time()
        down_mono = time.monotonic()
        net.configLinkStatus(cfg.switch_name(route_edge[0]), cfg.switch_name(route_edge[1]), "down")
        api_request("/routing/link-state", {"u": route_edge[0], "v": route_edge[1], "up": False})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready, _, _ = select.select([ping_proc.stdout], [], [], 0.1)
            if not ready:
                if ping_proc.poll() is not None:
                    break
                continue
            line = ping_proc.stdout.readline()
            if not line:
                continue
            if "bytes from" in line and "Destination Host Unreachable" not in line:
                # Timestamp printed by ping is the receive time; timing resolution is bounded by ping interval.
                recovery_ms = (time.monotonic() - down_mono) * 1000.0
                detected = True
                break
    finally:
        try:
            net.configLinkStatus(cfg.switch_name(route_edge[0]), cfg.switch_name(route_edge[1]), "up")
            api_request("/routing/link-state", {"u": route_edge[0], "v": route_edge[1], "up": True})
            edge_link = net._routing_inter_switch_links[route_edge]
            set_link_tc(edge_link, cfg.LINK_BW_MBPS, cfg.BASE_DELAYS_MS[route_edge])
            # Reset controller weight too in case this edge had prior scenario changes.
            api_request("/routing/link-delay", {"u": route_edge[0], "v": route_edge[1],
                                                 "delay_ms": cfg.BASE_DELAYS_MS[route_edge]})
        finally:
            try:
                ping_proc.terminate()
                ping_proc.communicate(timeout=2)
            except Exception:
                try:
                    ping_proc.kill()
                except Exception:
                    pass
    return recovery_ms, detected, route_edge, down_wall


def run_udp_with_ping(h1, h2, offered_mbps, duration_s, port=5201, ping_count=8):
    if not h1.cmd("command -v iperf").strip() or not h2.cmd("command -v iperf").strip():
        raise RuntimeError("iperf (iperf2) is missing inside Mininet hosts. Install it in WSL with: sudo apt install iperf")
    server = h2.popen(["iperf", "-s", "-u", "-p", str(port)], stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1)
    time.sleep(0.4)
    client = h1.popen(["iperf", "-c", h2.IP(), "-u", "-b", f"{offered_mbps:g}M",
                       "-t", str(int(duration_s)), "-p", str(port), "-i", "1"],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        ping_output = h1.cmd(f"ping -n -c {int(ping_count)} -i 0.2 -W 1 {h2.IP()}")
        try:
            client_output, _ = client.communicate(timeout=duration_s + 8)
        except subprocess.TimeoutExpired:
            client.kill()
            client_output, _ = client.communicate(timeout=2)
    finally:
        try:
            server.send_signal(signal.SIGINT)
            server_output, _ = server.communicate(timeout=3)
        except Exception:
            server.kill()
            try:
                server_output, _ = server.communicate(timeout=2)
            except Exception:
                server_output = ""
    # Prefer receiver output for actual received rate/jitter/loss.
    udp = parse_iperf_udp(server_output or "")
    if math.isnan(udp["udp_rx_mbps"]):
        udp = parse_iperf_udp(client_output or "")
    result = parse_ping(ping_output)
    result.update(udp)
    if not (server_output or client_output).strip():
        result["notes"] = "iperf returned no output; check iperf2 installation."
    return result


def run_one(net, h1, h2, scenario, mode, run_number, duration_s):
    reset_network(net)
    api_request("/routing/mode", {"mode": mode})
    api_request("/routing/metrics/reset", {})
    details = configure_scenario(net, scenario, run_number)
    recovery_ms, recovery_detected, failed_edge = float("nan"), False, None

    if scenario == "link_failure_recovery":
        recovery_ms, recovery_detected, failed_edge, _ = measure_link_failure_recovery(
            net, h1, h2, mode, timeout_s=max(5.0, duration_s + 2.0))
        details["changed_link"] = f"{failed_edge[0]}-{failed_edge[1]}"
        details["configured_delay_ms"] = cfg.BASE_DELAYS_MS[failed_edge]
        details["configured_bandwidth_mbps"] = cfg.LINK_BW_MBPS
        details["notes"] += " Recovery time is approximate, bounded by 200 ms ping interval and controller/OS scheduling."

    # A changing-traffic experiment changes offered load within each run, rather
    # than only changing it between run numbers. Each phase is saved separately.
    traffic_phases = [
        ("", details["offered_udp_mbps"]),
    ]
    if scenario == "changing_traffic":
        traffic_phases = [(f"phase_{index + 1}", rate) for index, rate in enumerate((2.0, 6.0, 9.0, 12.0))]

    output_rows = []
    for phase, offered_rate in traffic_phases:
        if phase:
            # Force a fresh PacketIn/routing decision for this load phase.
            api_request("/routing/mode", {"mode": mode})
            api_request("/routing/metrics/reset", {})
        measured = run_udp_with_ping(h1, h2, offered_rate, duration_s,
                                     port=5201 + ((run_number + len(output_rows)) % 100), ping_count=8)
        metrics = api_request("/routing/metrics")
        status = api_request("/routing/status")
        counters = metrics.get("counters", {})
        last_route = (status.get("last_route") or {}).get("path", "")
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "scenario": scenario,
            "mode": mode, "run": run_number, "traffic_phase": phase,
            "source_ip": cfg.H1_IP, "destination_ip": cfg.H2_IP,
            "offered_udp_mbps": offered_rate,
            "changed_link": details["changed_link"],
            "configured_delay_ms": details["configured_delay_ms"],
            "configured_bandwidth_mbps": details["configured_bandwidth_mbps"],
            **measured, "recovery_ms": recovery_ms, "recovery_detected": recovery_detected,
            "controller_fallbacks": counters.get("dqn_fallbacks", 0),
            "controller_failures": counters.get("failed_routes", 0),
            "forwarding_decisions": counters.get("routing_decisions", 0),
            "last_route": last_route, "notes": details["notes"],
        }
        output_rows.append(row)
    return output_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--mode", choices=["all", *MODES], default="all")
    parser.add_argument("--runs", type=int, default=1, help="repetitions per scenario and mode (default: 1)")
    parser.add_argument("--duration", type=int, default=4, help="UDP probe duration in seconds (default: 4)")
    parser.add_argument("--output", default="", help="optional output CSV path")
    args = parser.parse_args()
    if args.runs < 1 or args.duration < 1:
        parser.error("--runs and --duration must be >= 1")

    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    modes = MODES if args.mode == "all" else (args.mode,)
    # Validate controller/API before starting Mininet so failures are clear.
    ensure_controller()
    out_path = Path(args.output) if args.output else ROOT / "results" / f"network_experiments_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    net = build()
    try:
        net.start()
        configure_static_arp(net)
        h1, h2 = net.get("h1", "h2")
        # Wait briefly for every switch to establish its OpenFlow connection.
        deadline = time.time() + 12
        while time.time() < deadline:
            status = api_request("/routing/status")
            if len(status.get("connected_switches", [])) == cfg.NUM_NODES:
                break
            time.sleep(0.25)
        status = api_request("/routing/status")
        if len(status.get("connected_switches", [])) != cfg.NUM_NODES:
            raise RuntimeError(f"Only {len(status.get('connected_switches', []))}/{cfg.NUM_NODES} switches connected to controller. Status: {status}")
        if not h1.cmd("command -v iperf").strip():
            raise RuntimeError("iperf2 is not installed inside Mininet hosts. Install it in WSL: sudo apt install iperf")

        total = len(scenarios) * len(modes) * args.runs
        index = 0
        for scenario in scenarios:
            for mode in modes:
                for run_number in range(1, args.runs + 1):
                    index += 1
                    print(f"[{index}/{total}] scenario={scenario} mode={mode} run={run_number}", flush=True)
                    experiment_rows = run_one(net, h1, h2, scenario, mode, run_number, args.duration)
                    rows.extend(experiment_rows)
                    for row in experiment_rows:
                        phase_label = f" {row['traffic_phase']}" if row.get("traffic_phase") else ""
                        print(f"  {row['mode']}{phase_label} offered={row.get('offered_udp_mbps')} Mbps "
                              f"ping_loss={row.get('packet_loss_pct')}% udp_rx={row.get('udp_rx_mbps')} Mbps "
                              f"fallbacks={row.get('controller_fallbacks')} recovery_ms={row.get('recovery_ms')}", flush=True)
    finally:
        try:
            net.stop()
        finally:
            # Preserve completed measurements even if a later experiment fails.
            if rows:
                with out_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"\nWrote {len(rows)} rows to {out_path}")
                print("Run evaluation/analyze_network_results.py with the analysis environment to summarize and graph the results.")


if __name__ == "__main__":
    main()
