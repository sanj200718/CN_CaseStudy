"""Small parser helpers for Linux ping and iperf2 UDP text output.

Kept separate from Mininet imports so these parsing helpers can be unit-tested
on macOS and other environments where Mininet is unavailable.
"""
import math
import re


def parse_ping(output):
    result = {"ping_sent": 0, "ping_received": 0, "packet_loss_pct": float("nan"),
              "rtt_min_ms": float("nan"), "rtt_avg_ms": float("nan"),
              "rtt_max_ms": float("nan"), "rtt_stddev_ms": float("nan")}
    match = re.search(r"(\d+)\s+packets transmitted,\s*(\d+)\s+(?:packets )?received", output)
    if match:
        result["ping_sent"], result["ping_received"] = int(match.group(1)), int(match.group(2))
    loss = re.search(r"([\d.]+)%\s*packet loss", output)
    if loss:
        result["packet_loss_pct"] = float(loss.group(1))
    rtt = re.search(r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev)\s*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", output)
    if rtt:
        (result["rtt_min_ms"], result["rtt_avg_ms"], result["rtt_max_ms"],
         result["rtt_stddev_ms"]) = map(float, rtt.groups())
    return result


def parse_iperf_udp(output):
    result = {"udp_rx_mbps": float("nan"), "udp_jitter_ms": float("nan"), "udp_loss_pct": float("nan")}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # iperf2 receiver summary contains throughput, jitter and lost/total datagrams.
    for line in reversed(lines):
        if "ms" not in line or "/" not in line:
            continue
        speed = re.search(r"([\d.]+)\s+(bits/sec|Kbits/sec|Mbits/sec|Gbits/sec)", line)
        jitter = re.search(r"([\d.]+)\s+ms\s+(\d+)\s*/\s*(\d+)", line)
        if not speed or not jitter:
            continue
        value, unit = float(speed.group(1)), speed.group(2)
        factor = {"bits/sec": 1e-6, "Kbits/sec": 1e-3, "Mbits/sec": 1.0, "Gbits/sec": 1000.0}[unit]
        result["udp_rx_mbps"] = value * factor
        result["udp_jitter_ms"] = float(jitter.group(1))
        lost, total = int(jitter.group(2)), int(jitter.group(3))
        result["udp_loss_pct"] = (100.0 * lost / total) if total else float("nan")
        return result
    return result
