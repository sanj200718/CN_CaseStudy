# Latest simulator results captured from terminal output

These files preserve the latest five-scenario summary pasted by Iraban on 2026-10-22. The summary reported 30 runs per algorithm per scenario and 100% route success for both algorithms in all scenarios.

## Important provenance note

The latest run's *per-run raw CSV was not attached or copied into this archive*. Therefore, `summary.csv` preserves the exact aggregate values provided in the terminal output, and `improvement.csv` computes the DQN cost percentage change from those aggregate values. The existing `evaluation/results/raw_results.csv` belongs to an earlier run and must not be represented as raw data for this latest summary.

## Interpretation

These values are simulated route-delay/cost measurements, not actual packet latency, throughput, jitter, or packet loss. `reduced_bandwidth_proxy` is a simulated delay-penalty proxy, not a physical bandwidth-reduction experiment. In these provided results, DQN's mean route cost is higher than Dijkstra's in all five scenarios, while both have 100% route success.

## Exact aggregate values

| Scenario | Algorithm | Runs | Successful | Success rate (%) | Mean route cost | Std. dev. | Mean hops |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Dijkstra | 30 | 30 | 100.0 | 41.7298 | 15.2338 | 2.2333 |
| normal | DQN | 30 | 30 | 100.0 | 52.4212 | 19.9641 | 2.4000 |
| congested_link | Dijkstra | 30 | 30 | 100.0 | 50.8289 | 13.1701 | 2.2333 |
| congested_link | DQN | 30 | 30 | 100.0 | 62.2165 | 19.6455 | 2.4000 |
| high_delay_alternate_path | Dijkstra | 30 | 30 | 100.0 | 49.8065 | 11.0752 | 2.2667 |
| high_delay_alternate_path | DQN | 30 | 30 | 100.0 | 59.1963 | 19.4301 | 2.4333 |
| reduced_bandwidth_proxy | Dijkstra | 30 | 30 | 100.0 | 39.0894 | 17.3760 | 2.0667 |
| reduced_bandwidth_proxy | DQN | 30 | 30 | 100.0 | 44.5903 | 19.7636 | 2.0333 |
| link_failure | Dijkstra | 30 | 30 | 100.0 | 41.7234 | 13.9735 | 2.3333 |
| link_failure | DQN | 30 | 30 | 100.0 | 52.0865 | 20.6142 | 2.4667 |

Do not claim these results prove real-network performance or that DQN outperforms Dijkstra. Use the packet-level experiment runner to measure the real Mininet/Ryu network after live testing succeeds.
