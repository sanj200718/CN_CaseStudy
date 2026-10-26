# Machine Learning-Based Intelligent Routing Framework for Dynamic Networks

**Computer Networks — Group 4**  
**Members:** 25BCE2187 Karthik R Nair · 25BCE2169 Iraban Bhattacharya · 25BAI0149 Cherukuri Sanjana

This repository combines a DQN routing agent, a Mininet topology, a Ryu/OpenFlow 1.3 controller, REST hooks for controlled dynamic scenarios, and an experiment runner for packet-level measurements.

## Architecture

```text
Mininet hosts h1 / h2
        │  6 switches, 10 inter-switch links
        ▼
Open vSwitch (OpenFlow 1.3)
        │
        ▼
controller.py (Ryu)
  ├─ ospf     = OSPF-style hop-count baseline (not a full OSPF implementation)
  ├─ dijkstra = configured-delay-weighted shortest path
  └─ dqn      = DQN next-hop policy with explicit Dijkstra fallback on dead ends
        │
        ├─ controller decision CSV: results/controller_decisions.csv
        └─ REST API: http://127.0.0.1:8080/routing/...

experiments/run_network_experiments.py
        │
        ├─ delay changes / bandwidth constraints / link down-up
        ├─ ping RTT and loss
        ├─ iperf2 UDP throughput, jitter and loss
        └─ timestamped CSV → evaluation/analyze_network_results.py → graphs
```

## Important interpretation limits

- The existing `evaluation/results*` simulator experiments measure *simulated route costs* and route completion, not real packet latency, throughput or packet loss.
- The Ryu controller's delay weights come from `graph_config.py` and scenario REST updates. They are configured values, not continuously measured per-link delay telemetry.
- DQN receives a delay matrix and valid-neighbour list. The current trained model does **not** receive bandwidth or utilization as explicit features. The bandwidth-drop experiment measures packet-level impact but is not evidence that DQN adapts to bandwidth.
- `ospf` is an OSPF-style hop-count shortest-path baseline. Do not describe it as a full OSPF protocol unless one is implemented.
- In `dqn` mode, a Dijkstra fallback is enabled when the model returns no valid next hop. Fallback events are logged and counted; report results as **DQN with Dijkstra fallback**, not pure DQN.
- The controller's flow rules are installed in OpenFlow switches. The REST endpoint flushes IPv4 rules when the routing mode or link/delay state changes.

## 1. Required runtime

The Mininet/Ryu experiment must run in the same Linux/WSL distribution where Mininet and Open vSwitch are installed. Do not try to run Mininet directly on macOS.

Required WSL/Ubuntu packages include:

```bash
sudo apt update
sudo apt install mininet openvswitch-switch iperf curl python3-pip python3-venv
```

Ryu must already be installed in a Python environment compatible with the version in your WSL setup. Karthik reported that Ryu, Mininet and Open vSwitch are installed in his WSL environment. Since Ryu packaging can be version-sensitive, keep using the working environment rather than installing a second incompatible copy.

Install analysis/ML dependencies in a virtual environment (or the environment already used by the project):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install gymnasium
```

The Mininet runner is intentionally launched with the system Python (which can import Mininet), while the Ryu controller is launched in the environment where Ryu and PyTorch are installed.

## 2. Start Ryu controller

**WSL Terminal 1** — from the project root:

```bash
ryu-manager --ofp-tcp-listen-port 6653 controller.py
```

Ryu should expose the controller REST API on port 8080. In a second terminal, verify it:

```bash
curl -s http://127.0.0.1:8080/routing/status
```

The status payload should include `"controller": "ready"`. It may report zero connected switches until Mininet is started.

## 3. Smoke-test Mininet connectivity

**WSL Terminal 2** — stop any other Mininet session first, then run:

```bash
sudo /usr/bin/python3 topo.py --test
```

The controller must already be running. The test starts the topology and tries to ping between h1 and h2. If it fails, inspect the Ryu terminal for OpenFlow/flow-installation errors and check the status endpoint's `connected_switches` field.

To use Mininet CLI interactively instead:

```bash
sudo /usr/bin/python3 topo.py
```

Then use `nodes`, `net`, `pingall`, and `exit`. Do not run this interactive topology at the same time as `experiments/run_network_experiments.py`.

## 4. Run packet-level experiments

Keep Ryu running in Terminal 1. Run experiments from Terminal 2 (same WSL distro and project folder). Start with one short baseline run:

```bash
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario baseline --mode dijkstra --runs 1 --duration 3
```

Then test each routing mode on one scenario:

```bash
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario baseline --mode all --runs 1 --duration 3
```

Run the full scenario matrix after smoke tests pass:

```bash
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario all --mode all --runs 3 --duration 4
```

Available modes: `dijkstra`, `ospf`, `dqn`, `all`.  
Available scenarios:

| Scenario | Physical/controller change | Measurement focus |
|---|---|---|
| `baseline` | Original topology and delay settings | Reference RTT, UDP throughput/loss/jitter |
| `delay_increase` | Adds 25 ms to link 0–3 in Mininet and controller delay weights | Route choice and RTT change |
| `bandwidth_drop` | Reduces link 3–5 capacity to 2 Mbps | Actual throughput under a bottleneck; route metrics still use delay |
| `congestion` | Sends UDP traffic at 9 Mbps over 10 Mbps inter-switch links while ping runs | RTT/loss under near-capacity load |
| `link_failure_recovery` | Finds the logged h1→h2 route, disables its first inter-switch edge, observes first successful ping, then restores the edge | Approximate failover/recovery time and continuity |
| `changing_traffic` | Changes offered UDP load through 2, 6, 9 and 12 Mbps within each run | Throughput, UDP loss/jitter, RTT vs offered rate |

The runner checks for a live Ryu API, all six OpenFlow switch connections, and `iperf` inside Mininet hosts. It writes one timestamped CSV to `results/network_experiments_YYYYMMDD_HHMMSS.csv` and does not overwrite previous experiment files.

### Recovery-time definition

Recovery time is measured from the moment a route edge is administratively disabled until the first successful reply observed by a continuous ping. Ping runs at 200 ms intervals, so the value is approximate and includes controller/OS scheduling and probe granularity. The runner records `recovery_detected` separately to distinguish a measured value from a timeout.

## 5. Analyze real-network CSV and generate graphs

Activate the environment containing pandas and matplotlib, then run:

```bash
python evaluation/analyze_network_results.py results/network_experiments_YYYYMMDD_HHMMSS.csv --outdir evaluation/network_analysis
```

Replace the example CSV filename with the exact filename printed by the experiment runner. This produces `network_summary.csv` and available metric graphs for RTT, throughput, packet loss, jitter and recovery time. NaN values mean that the metric was not measured for a particular run or was not parsed from `iperf` output; inspect raw logs before making claims.

## 6. Inspect controller logs and change modes via REST

```bash
curl -s http://127.0.0.1:8080/routing/status
curl -s http://127.0.0.1:8080/routing/metrics
curl -s -X POST -H 'Content-Type: application/json' -d '{"mode":"dqn"}' http://127.0.0.1:8080/routing/mode
```

Other API requests:

```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"u":0,"v":3,"up":false}' http://127.0.0.1:8080/routing/link-state
curl -s -X POST -H 'Content-Type: application/json' -d '{"u":0,"v":3,"delay_ms":35}' http://127.0.0.1:8080/routing/link-delay
curl -s -X POST -H 'Content-Type: application/json' -d '{"mode":"dijkstra"}' http://127.0.0.1:8080/routing/reset
curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8080/routing/clear-flows
```

Controller decisions are appended to `results/controller_decisions.csv`, including selected path, route cost, success/failure reason, and whether a DQN fallback occurred.

## 7. Run local tests and simulator experiments

On a Python environment with `networkx`, `numpy`, `torch`, `pandas`, and `matplotlib` installed:

```bash
python -m unittest discover -s tests -v
python demo_compare.py
python evaluation/run_experiments.py --runs 30
python evaluation/analyze_results.py
```

The last two commands operate on the simulator, not Mininet. The Mininet packet-level experiment commands in section 4 must be run inside WSL/Ubuntu.

## Troubleshooting

- **Ryu REST API refused connection:** start `ryu-manager` first and check it is listening on port 8080.
- **Only some switches connected:** stop old Mininet sessions, restart Ryu, then start one topology. Confirm OpenFlow 1.3 and matching controller IP/port.
- **Ping has 100% loss:** inspect the controller log for flow-installation errors; ensure static ARP matches the host IP/MAC settings; verify all switches connected.
- **No UDP metrics parsed:** install iperf2 (`sudo apt install iperf`) in WSL; the current parser uses iperf2 receiver output.
- **Permission problems in `results/`:** avoid mixing root- and user-owned result directories; the runner is launched as root for Mininet.
- **Model file not found:** `routing_dqn.pt` must sit next to `controller.py`.

## Latest simulator summary and team handover

- `evaluation/results_latest_console_run/` preserves the latest five-scenario aggregate summary provided from the Mac terminal output, with a comparison table and provenance notes. The corresponding per-run raw CSV was not supplied with that terminal output; do not treat the older `evaluation/results/raw_results.csv` as if it generated the latest aggregate values.
- `HANDOFF_TO_KARTHIK.md` contains the WSL/Mininet/Ryu validation checklist and acceptance criteria for handing the real-network evaluation to Iraban.

## Team work split

- **Karthik:** Mininet topology, Open vSwitch/Ryu setup, traffic/scenario runtime.
- **Sanjana:** DQN training/model and inference policy.
- **Iraban:** experiment design, repeated evaluation, metric collection, analysis and graphs.
