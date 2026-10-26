"""Summarize simulated routing runs and generate comparison graphs.

Run from the repository root after run_experiments.py:
    python evaluation/analyze_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVALUATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALUATION_DIR / "results"
RAW_PATH = RESULTS_DIR / "raw_results.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
IMPROVEMENT_PATH = RESULTS_DIR / "improvement.csv"
DELAY_GRAPH_PATH = RESULTS_DIR / "mean_route_delay.png"
SUCCESS_GRAPH_PATH = RESULTS_DIR / "route_success_rate.png"
SCENARIO_ORDER = [
    "normal",
    "congested_link",
    "high_delay_alternate_path",
    "reduced_bandwidth_proxy",
    "link_failure",
]
SCENARIO_LABELS = {
    "normal": "Normal",
    "congested_link": "Congested link",
    "high_delay_alternate_path": "High-delay\nalternative path",
    "reduced_bandwidth_proxy": "Reduced-bandwidth\nproxy",
    "link_failure": "Link failure",
}


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_PATH}. Run `python evaluation/run_experiments.py` first."
        )

    df = pd.read_csv(RAW_PATH)
    if df.empty:
        raise ValueError("raw_results.csv is empty; no experiments were recorded.")

    df["route_valid"] = df["route_valid"].astype(str).str.lower().eq("true")
    df["route_delay_ms"] = pd.to_numeric(df["route_delay_ms"], errors="coerce")
    df["hop_count"] = pd.to_numeric(df["hop_count"], errors="coerce")

    rows = []
    for (scenario, algorithm), group in df.groupby(["scenario", "algorithm"], sort=False):
        successful = group[group["route_valid"]]
        delays = successful["route_delay_ms"].dropna()
        hops = successful["hop_count"].dropna()
        rows.append(
            {
                "scenario": scenario,
                "algorithm": algorithm,
                "total_runs": int(len(group)),
                "successful_routes": int(len(successful)),
                "success_rate_pct": round(100 * len(successful) / len(group), 2) if len(group) else np.nan,
                "mean_route_delay_ms_success_only": round(float(delays.mean()), 4) if len(delays) else np.nan,
                "std_route_delay_ms_success_only": round(float(delays.std(ddof=1)), 4) if len(delays) > 1 else (0.0 if len(delays) == 1 else np.nan),
                "mean_hop_count_success_only": round(float(hops.mean()), 4) if len(hops) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    # Positive delay_reduction_pct means DQN's mean route delay was lower than Dijkstra's.
    improvement_rows = []
    for scenario in df["scenario"].drop_duplicates():
        chunk = summary[summary["scenario"] == scenario].set_index("algorithm")
        dijkstra_delay = chunk.at["Dijkstra", "mean_route_delay_ms_success_only"] if "Dijkstra" in chunk.index else np.nan
        dqn_delay = chunk.at["DQN", "mean_route_delay_ms_success_only"] if "DQN" in chunk.index else np.nan
        dqn_success = chunk.at["DQN", "success_rate_pct"] if "DQN" in chunk.index else np.nan
        reduction = np.nan
        if pd.notna(dijkstra_delay) and dijkstra_delay != 0 and pd.notna(dqn_delay):
            reduction = (dijkstra_delay - dqn_delay) / dijkstra_delay * 100.0
        improvement_rows.append(
            {
                "scenario": scenario,
                "dijkstra_mean_route_delay_ms": dijkstra_delay,
                "dqn_mean_route_delay_ms": dqn_delay,
                "dqn_success_rate_pct": dqn_success,
                "dqn_delay_reduction_pct_vs_dijkstra": round(float(reduction), 4) if pd.notna(reduction) else np.nan,
            }
        )
    improvement = pd.DataFrame(improvement_rows)
    improvement.to_csv(IMPROVEMENT_PATH, index=False)

    # Graph 1: mean route-delay/cost among successful simulated routes.
    plotting = summary.pivot(index="scenario", columns="algorithm", values="mean_route_delay_ms_success_only")
    plotting = plotting.reindex([s for s in SCENARIO_ORDER if s in plotting.index])
    plotting.index = [SCENARIO_LABELS.get(s, s) for s in plotting.index]
    ax = plotting.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Mean Simulated Route Cost by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Route delay/cost (ms; successful routes only)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Routing method")
    plt.tight_layout()
    plt.savefig(DELAY_GRAPH_PATH, dpi=180)
    plt.close()

    # Graph 2: proportion of paths that reach the destination without a loop/invalid hop.
    success_plot = summary.pivot(index="scenario", columns="algorithm", values="success_rate_pct")
    success_plot = success_plot.reindex([s for s in SCENARIO_ORDER if s in success_plot.index])
    success_plot.index = [SCENARIO_LABELS.get(s, s) for s in success_plot.index]
    ax = success_plot.plot(kind="bar", figsize=(10, 5), ylim=(0, 100))
    ax.set_title("Successful Route Rate by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Successful routes (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Routing method")
    plt.tight_layout()
    plt.savefig(SUCCESS_GRAPH_PATH, dpi=180)
    plt.close()

    print("Summary saved:", SUMMARY_PATH)
    print("Improvement table saved:", IMPROVEMENT_PATH)
    print("Graph saved:", DELAY_GRAPH_PATH)
    print("Graph saved:", SUCCESS_GRAPH_PATH)
    print("\nNote: these are simulated route costs. They are not actual packet latency, throughput, jitter, or packet loss.")
    print("\nSummary:\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
