# Evaluation guide

There are two distinct evaluation levels in this project. Keep their claims separate.

## A. Simulator experiments (available without Mininet)

These scripts operate on the six-node NetworkX graph and the trained DQN checkpoint. They measure simulated route costs, hop counts, route validity, and destination reachability; they do not measure actual packets.

```bash
python evaluation/run_experiments.py --runs 30
python evaluation/analyze_results.py
python evaluation/run_dynamic_experiments_state_cycle.py --runs 30
```

`run_experiments.py` writes `results/raw_results.csv`, `summary.csv`, `improvement.csv`, and two graphs. The dynamic script writes to `results_dynamic_state_cycle/` and does not overwrite the static CSVs.

Scenarios named `reduced_bandwidth_proxy` are delay penalties, not physical bandwidth tests. Dynamic delay/link changes are simulated snapshots; they are not Mininet measurements.

## B. Mininet packet-level experiments (requires WSL/Ubuntu)

Follow the root README. Start `ryu-manager --ofp-tcp-listen-port 6653 controller.py` in one terminal, then run the Mininet runner in a second terminal:

```bash
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario baseline --mode dijkstra --runs 1 --duration 3
sudo /usr/bin/python3 experiments/run_network_experiments.py --scenario all --mode all --runs 3 --duration 4
```

The runner writes timestamped files to `results/network_experiments_*.csv`. Analyze one file with:

```bash
python evaluation/analyze_network_results.py results/network_experiments_YYYYMMDD_HHMMSS.csv --outdir evaluation/network_analysis
```

These experiments report ping RTT/loss and iperf2 UDP throughput/jitter/loss, plus approximate link-failure recovery. The controller uses configured link-delay values, not continuous active probing. DQN fallback-to-Dijkstra is logged and must be disclosed in analysis.
