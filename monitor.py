"""
monitor.py — SRE Observability, Four Golden Signals & PSI Drift Monitor
======================================================================
Observability engine providing:
1. Four Golden Signals: Latency, Traffic, Errors (SLO Breaches), and Saturation.
2. SRE Error Budget & Burn Rate calculation based on latency SLO target (99.0%).
3. Population Stability Index (PSI) to detect workload traffic drift:
   - PSI < 0.10: Nominal (No drift)
   - 0.10 <= PSI < 0.25: Moderate Drift (Warning alert)
   - PSI >= 0.25: Significant Drift (Critical alert / Retrain trigger)
4. Hierarchical Alerting to prevent alert fatigue.
5. Automated generation of `logs/monitoring_report.json` and `charts/psi_drift_report.png`.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from schema import DEFAULT_CONFIG_PATH, load_config

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_psi(
    baseline: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Population Stability Index (PSI) between baseline and target distributions:
        PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))

    Returns:
        psi_value: Total PSI scalar
        bin_details: Dict containing bin edges, expected %, actual %, and per-bin contributions
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Remove any NaNs or Infs
    baseline = baseline[np.isfinite(baseline)]
    target = target[np.isfinite(target)]

    if len(baseline) == 0 or len(target) == 0:
        return 0.0, {"error": "Empty or non-finite distribution"}

    # Define quantile-based or uniform bins on the baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, percentiles)
    # Ensure strictly increasing edges to prevent duplicate bin errors
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.linspace(baseline.min() - 1e-3, baseline.max() + 1e-3, num_bins + 1)

    # Bin counts
    base_counts, _ = np.histogram(baseline, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)

    # Convert to proportions with epsilon smoothing to prevent div-by-zero / log(0)
    base_pct = (base_counts + epsilon) / (len(baseline) + epsilon * len(base_counts))
    target_pct = (target_counts + epsilon) / (len(target) + epsilon * len(target_counts))

    # Normalize proportions to strictly sum to 1.0
    base_pct = base_pct / np.sum(base_pct)
    target_pct = target_pct / np.sum(target_pct)

    # Per-bin PSI contribution: (Actual - Expected) * ln(Actual / Expected)
    psi_bins = (target_pct - base_pct) * np.log(target_pct / base_pct)
    psi_total = float(np.sum(psi_bins))

    return psi_total, {
        "num_bins": len(psi_bins),
        "bin_edges": [float(b) for b in bin_edges],
        "expected_pct": [float(p) for p in base_pct],
        "actual_pct": [float(p) for p in target_pct],
        "bin_contributions": [float(c) for c in psi_bins],
    }


def classify_drift(psi: float, moderate_thresh: float = 0.10, significant_thresh: float = 0.25) -> str:
    """Classify drift according to standard SRE thresholds."""
    if psi < moderate_thresh:
        return "NOMINAL (No Drift)"
    elif psi < significant_thresh:
        return "WARNING (Moderate Drift)"
    else:
        return "CRITICAL (Significant Drift - Action Required)"


class SREMonitor:
    """
    SRE & Observability Engine tracking Golden Signals, Error Budgets, and Workload Drift.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.mon_cfg = self.cfg.get("monitoring", {})
        self.env_cfg = self.cfg.get("environment", {})

        self.slo_latency_ms = float(self.env_cfg.get("slo_latency_ms", 150.0))
        self.slo_target_ratio = float(self.mon_cfg.get("slo_target_ratio", 0.99))
        self.psi_bins = int(self.mon_cfg.get("psi_bins", 10))
        self.moderate_threshold = float(self.mon_cfg.get("psi_moderate_threshold", 0.10))
        self.significant_threshold = float(self.mon_cfg.get("psi_significant_threshold", 0.25))

        self.reports_dir = os.path.join(ROOT_DIR, "logs")
        self.charts_dir = os.path.join(ROOT_DIR, "charts")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

    def extract_golden_signals(self) -> Dict[str, Any]:
        """
        Aggregate the Four Golden Signals from `logs/swap_comparison.json`:
        1. Latency (P50/P99 decision latency)
        2. Traffic (Incoming arrival rate)
        3. Errors (SLO Breaches & Error Budget Burn)
        4. Saturation (CPU/RAM Utilization)
        """
        swap_json_path = os.path.join(self.reports_dir, "swap_comparison.json")
        if not os.path.exists(swap_json_path):
            raise FileNotFoundError(
                f"Benchmark results missing at {swap_json_path}. Run evaluate_swap.py first!"
            )

        with open(swap_json_path, "r", encoding="utf-8") as f:
            swap_data = json.load(f)

        policies_summary = {}
        # Allowed breach ratio = 1.0 - 0.99 = 0.01 (1%)
        allowed_breach_rate = 1.0 - self.slo_target_ratio

        for item in swap_data.get("results", []):
            name = item["policy"]
            breach_rate = float(item["slo_breach_rate"])
            # Error budget burn rate: actual / allowed
            burn_rate = breach_rate / allowed_breach_rate if allowed_breach_rate > 0 else 0.0

            policies_summary[name] = {
                "name": name,
                "family": item.get("metadata", {}).get("family", "Unknown"),
                # Signal 1: Latency
                "mean_infer_latency_ms": float(item["mean_infer_latency_ms"]),
                "p99_infer_latency_ms": float(item["p99_infer_latency_ms"]),
                # Signal 2: Traffic
                "total_steps": int(item["total_steps"]),
                "episodes": int(item["episodes"]),
                # Signal 3: Errors & Error Budget
                "slo_breaches": int(item["slo_breaches"]),
                "slo_breach_rate": breach_rate,
                "error_budget_allowed_rate": allowed_breach_rate,
                "error_budget_burn_rate": float(burn_rate),
                "budget_overruns": int(item["budget_overruns"]),
                # Signal 4: Saturation
                "mean_utilization": float(item["mean_utilization"]),
                "average_ep_ret": float(item["average_ep_ret"]),
                "release_gate_passed": bool(item.get("release_gate_passed", False)),
            }

        return policies_summary

    def run_workload_drift_analysis(self) -> Dict[str, Any]:
        """
        Compute Population Stability Index (PSI) on real telemetry from `offline_store.parquet`.
        Compares baseline traffic distribution (first 50% steps) against current traffic (last 50% steps)
        plus an explicit sudden spike distribution to demonstrate drift detection.
        """
        parquet_path = os.path.join(ROOT_DIR, "data", "feature_store", "offline_store.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Feature store missing at {parquet_path}. Run feature_store.py first!"
            )

        df = pd.read_parquet(parquet_path)
        half = len(df) // 2

        features_to_check = ["arrival_rate", "fw_util", "ws_util", "db_util"]
        drift_results = {}

        for feat in features_to_check:
            if feat not in df.columns:
                continue
            baseline_vals = df[feat].iloc[:half].to_numpy()
            target_vals = df[feat].iloc[half:].to_numpy()

            psi_val, details = calculate_psi(
                baseline_vals, target_vals, num_bins=self.psi_bins
            )
            status = classify_drift(psi_val, self.moderate_threshold, self.significant_threshold)

            drift_results[feat] = {
                "feature": feat,
                "psi_score": float(psi_val),
                "status": status,
                "baseline_mean": float(np.mean(baseline_vals)),
                "target_mean": float(np.mean(target_vals)),
                "baseline_std": float(np.std(baseline_vals)),
                "target_std": float(np.std(target_vals)),
                "bin_details": details,
            }

        # Simulated Bursty Workload Drift Scenario (Demonstrating Red Alert for viva)
        base_arr = df["arrival_rate"].iloc[:half].to_numpy()
        burst_arrival = base_arr * 2.8 + np.random.normal(50, 10, len(base_arr))
        # Ensure bin edges cover both distributions
        min_val = min(base_arr.min(), burst_arrival.min())
        max_val = max(base_arr.max(), burst_arrival.max())
        custom_edges = np.linspace(min_val, max_val, self.psi_bins + 1)
        base_counts, _ = np.histogram(base_arr, bins=custom_edges)
        burst_counts, _ = np.histogram(burst_arrival, bins=custom_edges)
        eps = 1e-4
        b_pct = (base_counts + eps) / np.sum(base_counts + eps)
        t_pct = (burst_counts + eps) / np.sum(burst_counts + eps)
        burst_psi = float(np.sum((t_pct - b_pct) * np.log(t_pct / b_pct)))

        drift_results["simulated_traffic_surge"] = {
            "feature": "arrival_rate_surge_stress",
            "psi_score": float(burst_psi),
            "status": classify_drift(burst_psi, self.moderate_threshold, self.significant_threshold),
            "baseline_mean": float(np.mean(base_arr)),
            "target_mean": float(np.mean(burst_arrival)),
            "note": "Simulated sudden surge demonstrating alert trigger (PSI >= 0.25)",
        }

        return drift_results

    def generate_report(self) -> Dict[str, Any]:
        """Compile and write `logs/monitoring_report.json` and render `charts/psi_drift_report.png`."""
        golden_signals = self.extract_golden_signals()
        drift_metrics = self.run_workload_drift_analysis()

        # Determine overall system health
        max_drift_psi = max([m["psi_score"] for k, m in drift_metrics.items() if "surge" not in k])
        system_alert = "GREEN (Healthy)"
        if max_drift_psi >= self.significant_threshold:
            system_alert = "RED (Critical Workload Drift)"
        elif max_drift_psi >= self.moderate_threshold:
            system_alert = "YELLOW (Moderate Drift Warning)"

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "system_health": system_alert,
            "slo_latency_target_ms": self.slo_latency_ms,
            "slo_availability_target_ratio": self.slo_target_ratio,
            "golden_signals_by_policy": golden_signals,
            "workload_drift_psi": drift_metrics,
        }

        report_path = os.path.join(self.reports_dir, "monitoring_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Saved SRE monitoring report to {report_path}")
        self._plot_dashboard(golden_signals, drift_metrics)
        return report

    def _plot_dashboard(self, golden: Dict[str, Any], drift: Dict[str, Any]) -> None:
        """Render a publication-quality 3-panel SRE dashboard."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plt.subplots_adjust(wspace=0.35, top=0.88, bottom=0.15)

        # Panel 1: Four Golden Signals Tradeoff (Breach Rate vs Decision Latency)
        ax1 = axes[0]
        policies = list(golden.keys())
        breach_rates = [golden[p]["slo_breach_rate"] * 100.0 for p in policies]
        p99_latencies = [golden[p]["p99_infer_latency_ms"] for p in policies]
        colors_list = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]

        bars = ax1.bar(policies, breach_rates, color=colors_list[: len(policies)], alpha=0.85)
        ax1.set_xticks(range(len(policies)))
        ax1.set_xticklabels(policies, rotation=25, ha="right")
        ax1.set_ylabel("SLO Breach Rate (%)", color="#1E293B", fontweight="bold")
        ax1.set_title("Signal 3: SLO Breach Rate (%)", fontsize=10, fontweight="bold", color="#1E3A8A")
        ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

        # Annotate values
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

        # Panel 2: P99 Decision Latency (ms)
        ax2 = axes[1]
        bars2 = ax2.bar(policies, p99_latencies, color=colors_list[: len(policies)], alpha=0.85)
        ax2.set_xticks(range(len(policies)))
        ax2.set_xticklabels(policies, rotation=25, ha="right")
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{h:.2f}ms", ha="center", va="bottom", fontsize=8)

        # Panel 3: Workload Drift PSI Scores
        ax3 = axes[2]
        drift_feats = [k for k in drift.keys() if "surge" not in k] + ["simulated_traffic_surge"]
        drift_labels = [f.replace("_", " ").title() for f in drift_feats]
        psi_scores = [drift[k]["psi_score"] for k in drift_feats]
        bar_colors = ["#059669" if p < 0.10 else ("#D97706" if p < 0.25 else "#DC2626") for p in psi_scores]

        bars3 = ax3.barh(drift_labels, psi_scores, color=bar_colors, alpha=0.85)
        ax3.axvline(0.10, color="#D97706", linestyle="--", label="Moderate (0.10)")
        ax3.axvline(0.25, color="#DC2626", linestyle="--", label="Critical (0.25)")
        ax3.set_xlabel("Population Stability Index (PSI)", fontweight="bold")
        ax3.set_title("Workload Drift Detection (PSI)", fontsize=10, fontweight="bold", color="#1E3A8A")
        ax3.grid(True, linestyle="--", alpha=0.4, axis="x")
        ax3.legend(loc="lower right", fontsize=8)

        for bar in bars3:
            w = bar.get_width()
            ax3.text(w + 0.01, bar.get_y() + bar.get_height() / 2.0, f"PSI={w:.3f}", ha="left", va="center", fontsize=8)

        chart_path = os.path.join(self.charts_dir, "psi_drift_report.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved SRE Observability & Drift Dashboard to {chart_path}")


def main():
    print("=" * 70)
    print("COUNSEL SRE Observability & PSI Drift Monitor")
    print("=" * 70)
    monitor = SREMonitor()
    report = monitor.generate_report()

    print("\n--- Summary Leaderboard ---")
    for name, data in report["golden_signals_by_policy"].items():
        print(
            f"• {name.upper():<22} | Return: {data['average_ep_ret']:>6.1f} | "
            f"Breaches: {data['slo_breach_rate']*100:>5.1f}% | "
            f"P99 Latency: {data['p99_infer_latency_ms']:>5.2f}ms | "
            f"Gate: {'PASS' if data['release_gate_passed'] else 'FAIL'}"
        )

    print("\n--- Workload Drift Detection (PSI) ---")
    for feat, data in report["workload_drift_psi"].items():
        print(f"• {feat:<28} | PSI: {data['psi_score']:>6.4f} | Status: {data['status']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
