# Updated Simulator Benchmark (Preliminary, Not Final Project Claims)

These figures were regenerated after the evaluator was updated to pass `visited_nodes` to DQN inference. The current checked-in `routing_dqn.pt` checkpoint was **not retrained**. Each method uses the same per-run graph and delay snapshot; 30 runs per scenario were completed.

| Scenario | Dijkstra mean route cost (ms) | DQN mean route cost (ms) | DQN route success | DQN cost change vs. Dijkstra |
|---|---:|---:|---:|---:|
| Normal | 41.7298 | 52.4212 | 100% | +25.62% |
| Congested-link delay penalty | 50.8289 | 62.2165 | 100% | +22.40% |
| High-delay alternative path | 49.8065 | 59.1963 | 100% | +18.85% |
| Reduced-bandwidth proxy | 39.0894 | 44.5903 | 100% | +14.07% |
| Link failure | 41.7234 | 52.0865 | 100% | +24.84% |

Positive cost change means DQN's measured simulator route cost was higher (worse) than Dijkstra's. Both algorithms reached the destination in all runs, but Dijkstra had lower mean route cost in every scenario.

## Interpretation

These are a **static-snapshot simulator benchmark**, not measured packet latency or evidence that ML improves a real network. Weighted Dijkstra is expected to minimize additive link-delay cost for a fixed graph and delay snapshot. The `reduced_bandwidth_proxy` and `congested_link` scenarios alter simulated delay costs; they are not physical bandwidth/congestion experiments.

Run `python evaluation/run_experiments.py --runs 30` followed by `python evaluation/analyze_results.py` to regenerate. This document should be updated if topology, model checkpoint, seeds, action masking, or scenario settings change.
