#!/usr/bin/env python3
"""Summarize packet-level network experiment CSVs and create comparison plots."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV emitted by experiments/run_network_experiments.py")
    parser.add_argument("--outdir", default="evaluation/network_analysis", help="output directory")
    args = parser.parse_args()
    source = Path(args.csv)
    if not source.exists():
        parser.error(f"CSV not found: {source}")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    if df.empty:
        raise SystemExit("No experiment records found in input CSV")

    metrics = ["rtt_avg_ms", "packet_loss_pct", "udp_rx_mbps", "udp_jitter_ms", "udp_loss_pct", "recovery_ms"]
    if "recovery_detected" in df.columns:
        recovery_flags = df["recovery_detected"].astype(str).str.lower().eq("true")
    else:
        recovery_flags = pd.Series(False, index=df.index)
    # Recovery is only tested in link_failure_recovery. Mark it N/A elsewhere
    # instead of misleadingly reporting 0% recovery on scenarios that never
    # attempted a recovery measurement.
    df["recovery_detected_pct_row"] = recovery_flags.astype(float).where(
        df["scenario"].astype(str).eq("link_failure_recovery")
    )
    summary = df.groupby(["scenario", "mode"], dropna=False).agg(
        measurement_rows=("run", "count"),
        runs_with_any_ping_reply_pct=("ping_received", lambda s: 100.0 * (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
        recovery_detected_pct=("recovery_detected_pct_row", lambda s: 100.0 * s.mean() if s.notna().any() else float("nan")),
        **{f"mean_{metric}": (metric, "mean") for metric in metrics if metric in df.columns},
        **{f"std_{metric}": (metric, "std") for metric in metrics if metric in df.columns},
        dqn_fallbacks=("controller_fallbacks", "sum"),
        controller_failures=("controller_failures", "sum"),
    ).reset_index()
    summary.to_csv(out / "network_summary.csv", index=False)

    plots = [
        ("rtt_avg_ms", "Mean RTT (ms)", "mean_rtt_by_mode.png"),
        ("udp_rx_mbps", "UDP received throughput (Mbps)", "mean_throughput_by_mode.png"),
        ("packet_loss_pct", "Ping packet loss (%)", "ping_loss_by_mode.png"),
        ("udp_loss_pct", "UDP datagram loss (%)", "udp_loss_by_mode.png"),
        ("udp_jitter_ms", "UDP jitter (ms)", "udp_jitter_by_mode.png"),
        ("recovery_ms", "Recovery time (ms; approximate)", "recovery_time_by_mode.png"),
    ]
    for metric, ylabel, filename in plots:
        if metric not in df.columns or df[metric].notna().sum() == 0:
            continue
        agg = df.groupby(["scenario", "mode"])[metric].mean().unstack("mode")
        ax = agg.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scenario")
        ax.set_title(f"{ylabel} by routing mode and scenario")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out / filename, dpi=160)
        plt.close()
    print(f"Summary: {out / 'network_summary.csv'}")
    for path in sorted(out.glob("*.png")):
        print(f"Graph: {path}")
    print("Interpretation note: these metrics are from Mininet packet traffic, not the earlier simulator route-cost estimates.")


if __name__ == "__main__":
    main()
