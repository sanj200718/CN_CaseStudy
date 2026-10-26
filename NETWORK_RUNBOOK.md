# Network runbook: Mininet/Ryu setup (tested commands)

Everything below was run in **WSL2, Ubuntu 26.04** (system Python 3.14). Mininet runs with the system Python; Ryu and PyTorch run in a separate **Python 3.9 virtual environment** (`~/ryu-env`), because Ryu does not work on newer Python versions.

## 1. One-time setup

**System packages**
```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch iperf iproute2 curl git
```

**Python 3.9 environment for Ryu** (using `uv`, which downloads Python 3.9 without touching the system Python)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv --python 3.9 --seed ~/ryu-env
source ~/ryu-env/bin/activate
pip install "setuptools<58" wheel
pip install eventlet==0.30.2 networkx pbr
PBR_VERSION=4.34 pip install ryu==4.34 --no-build-isolation
ryu-manager --version
```
Ryu 4.34 needs the `PBR_VERSION` workaround, otherwise pip reports its version as 0.0.0 and refuses to install it.

**ML and analysis packages (same environment)**
```bash
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib
```
Tested with torch 2.8.0+cpu.

## 2. Start order matters

Always clean up **before** starting Ryu. `sudo mn -c` kills any running `ryu-manager`.

```bash
sudo mn -c
sudo service openvswitch-switch start
```

**Terminal 1: controller** (from the project root)
```bash
source ~/ryu-env/bin/activate
ryu-manager --ofp-tcp-listen-port 6653 controller.py
```
Expect `Loaded DQN model` and `model_loaded=True`.

**Terminal 2: check the API**
```bash
curl -s http://127.0.0.1:8080/routing/status
```
Expect `"controller": "ready"`.

## 3. Connectivity test
```bash
sudo /usr/bin/python3 topo.py --test
```
Expected: all 6 switches connect (Ryu logs a table-miss rule for each), `0% dropped`, and a route such as `path=[0, 3, 5]` in the Ryu log.

When Mininet exits, Ryu logs `OpenFlow port down` for every link. This is normal. Before running experiments after a manual `topo.py` session, reset the controller state:
```bash
curl -s -X POST http://127.0.0.1:8080/routing/reset -d '{}'
```

## 4. Switch routing modes through REST
```bash
for m in dijkstra ospf dqn; do
  curl -s -X POST -H 'Content-Type: application/json' -d "{\"mode\":\"$m\"}" http://127.0.0.1:8080/routing/mode; echo
done
curl -s http://127.0.0.1:8080/routing/metrics
```
Every decision is appended to `results/controller_decisions.csv`, including whether DQN fell back to Dijkstra and why.

## 5. Run the experiments

Keep Ryu running. Do not leave an interactive `topo.py` session open, because the runner builds its own network.

```bash
# first check: one baseline run
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario baseline --mode dijkstra --runs 1 --duration 3

# baseline, all three modes
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario baseline --mode all --runs 1 --duration 3

# short full matrix (1 run each)
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario all --mode all --runs 1 --duration 3

# full matrix used for the results below (6 scenarios x 3 modes x 3 runs)
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario all --mode all --runs 3 --duration 4
```
Each run writes a new timestamped file, `results/network_experiments_YYYYMMDD_HHMMSS.csv`. Raw files are never overwritten.

## 6. Analyze
Use the exact filename the runner printed:
```bash
python evaluation/analyze_network_results.py results/network_experiments_20261025_171812.csv --outdir evaluation/network_analysis
```
This writes `network_summary.csv` and six graphs (RTT, throughput, ping loss, UDP loss, jitter, recovery time) to `evaluation/network_analysis/`.

## 7. First results (full matrix, 3 runs per cell)

Mean RTT in ms:

| Scenario | Dijkstra | DQN (with fallback) | OSPF-style |
|---|---|---|---|
| baseline | 55.7 | 91.1 | 105.6 |
| delay_increase | 85.6 | 106.2 | 105.7 |
| congestion | 55.4 | 90.6 | 105.3 |
| changing_traffic | 107.7 | 143.6 | 157.1 |
| link_failure_recovery | 55.5 | 90.9 | 105.3 |
| bandwidth_drop | 1058.1 | 90.5 | 105.2 |

Other measured values:
- **Bandwidth drop:** Dijkstra had 50% ping loss and 1.95 Mbps UDP throughput. DQN and OSPF-style had 0% ping loss and 8.4 Mbps.
- **Recovery time after link failure:** Dijkstra 302 ms, DQN 296 ms, OSPF-style 271 ms. The differences are within about one ping interval.
- **UDP loss** was 0% and jitter about 0.1 ms in all scenarios except bandwidth drop for Dijkstra.
- **Congestion and changing traffic** gave nearly identical throughput for all modes.

## 8. Limitations (report these)

- `ospf` is an **OSPF-style hop-count baseline**, not the OSPF protocol.
- **DQN results are "DQN with Dijkstra fallback".** The model was trained only for source node 0 to destination node 5. The return direction (5 to 0) always fails with `no_valid_next_hop` and falls back to Dijkstra, so measured round-trip values mix a DQN forward path with a Dijkstra return path.
- In the baseline condition the DQN forward path (0-4-2-5, 60 ms) is slower than the Dijkstra path (0-3-5, 25 ms).
- The DQN input has no bandwidth or utilisation features. Its good bandwidth-drop result is incidental: its path happened to avoid the throttled link (3-5).
- Delay-weighted Dijkstra uses the configured link delays, not continuously measured delays.
- Recovery time is approximate (periodic ping probes plus controller scheduling).
- Only 3 runs per cell, so no statistical significance is claimed.
- Reset of link state, qdisc settings and controller state between runs was not independently verified.
- The simulator results in `evaluation/results*` are simulated route costs, not packet-level measurements.

## 9. Known harmless messages
- `sch_htb: quantum of class 50001 is big` appears many times when links are created. It is a Linux traffic-control warning and does not affect results.
