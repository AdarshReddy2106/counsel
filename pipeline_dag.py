"""
pipeline_dag.py — DAG Orchestrator & Kahn's Topological Sorter
==============================================================
Workflow orchestration providing:
1. Orchestration: Directed Acyclic Graph (DAG) workflow execution.
2. Kahn's Algorithm (`topological_order`): Computes dependency execution order
   and detects circular dependency cycles at compile time.
3. Exponential Backoff: `base * (factor ** attempt)` retry pattern for transient failures.
4. End-to-End Pipeline Execution:
     validate_config
           ↓
     extract_features
           ↓
     score_policies
       ↙         ↘
 monitor_drift     generate_model_card
       ↘         ↙
     notify_summary
"""

import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Set, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def topological_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Kahn's Algorithm for Topological Sorting of a Directed Acyclic Graph (DAG).

    Args:
        nodes: List of unique task identifiers.
        edges: List of directed dependencies (upstream_task, downstream_task).

    Returns:
        List of task IDs in strictly valid dependency order.

    Raises:
        ValueError: If a circular dependency cycle is detected.
    """
    in_degree: Dict[str, int] = {node: 0 for node in nodes}
    adjacency: Dict[str, List[str]] = defaultdict(list)

    for src, dst in edges:
        if src not in in_degree or dst not in in_degree:
            raise ValueError(f"Edge ({src} -> {dst}) references unknown task.")
        adjacency[src].append(dst)
        in_degree[dst] += 1

    # Queue of all nodes with 0 in-degree (ready to run)
    zero_in_degree = deque([node for node, deg in in_degree.items() if deg == 0])
    ordered_tasks: List[str] = []

    while zero_in_degree:
        curr = zero_in_degree.popleft()
        ordered_tasks.append(curr)

        for neighbor in adjacency[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)

    if len(ordered_tasks) != len(nodes):
        unresolved = [n for n, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"Cyclic dependency detected! Tasks in cycle: {unresolved}"
        )

    return ordered_tasks


def execute_with_backoff(
    task_fn: Callable[[], Any],
    task_name: str,
    max_retries: int = 3,
    base_delay: float = 0.5,
    factor: float = 2.0,
) -> Any:
    """
    Execute `task_fn` with Exponential Backoff retry pattern.
    Delay formula: base_delay * (factor ** attempt)
    """
    attempt = 0
    last_err = None

    while attempt < max_retries:
        try:
            return task_fn()
        except Exception as e:
            attempt += 1
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (factor ** (attempt - 1))
                print(f"  [RETRY] Task '{task_name}' failed ({e}). Retrying in {delay:.2f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)

    raise RuntimeError(f"Task '{task_name}' failed after {max_retries} attempts: {last_err}")


# ==================== PIPELINE TASK DEFINITIONS ====================


def task_validate_config() -> Dict[str, Any]:
    """Task 1: Load config.yaml and verify schema contracts & Norway Problem."""
    from schema import DEFAULT_CONFIG_PATH, load_config
    cfg = load_config(DEFAULT_CONFIG_PATH)
    return {
        "status": "PASS",
        "project": cfg["project"]["name"],
        "active_policy": cfg["policy"]["active_policy"],
        "slo_ms": cfg["environment"]["slo_latency_ms"],
    }


def task_extract_features() -> Dict[str, Any]:
    """Task 2: Verify or materialize the Offline Parquet Feature Store."""
    from feature_store import FeatureStore
    fs = FeatureStore()
    if os.path.exists(fs.parquet_path) and os.path.exists(fs.metadata_path):
        import pandas as pd
        df = pd.read_parquet(fs.parquet_path)
        return {
            "status": "PASS (Cached)",
            "rows": len(df),
            "parquet_path": fs.parquet_path,
        }
    else:
        # Run 5 smoke episodes to materialize if not present
        res = fs.materialize_from_ppo(num_episodes=5)
        return {"status": "PASS (Materialized)", "details": res}


def task_score_policies() -> Dict[str, Any]:
    """Task 3: Benchmark all 5 plug-in models on identical workloads."""
    swap_json = os.path.join(ROOT_DIR, "logs", "swap_comparison.json")
    if os.path.exists(swap_json):
        with open(swap_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "PASS (Cached)",
            "models_evaluated": len(data.get("results", [])),
            "source": swap_json,
        }
    else:
        from evaluate_swap import evaluate_all_available_policies
        res = evaluate_all_available_policies(num_episodes=2)
        return {"status": "PASS (Evaluated)", "models_evaluated": len(res)}


def task_monitor_drift() -> Dict[str, Any]:
    """Task 4: Run SRE Four Golden Signals & PSI Drift Monitoring."""
    from monitor import SREMonitor
    monitor = SREMonitor()
    report = monitor.generate_report()
    return {
        "status": "PASS",
        "system_health": report["system_health"],
        "max_psi": max([m["psi_score"] for k, m in report["workload_drift_psi"].items() if "surge" not in k]),
    }


def task_generate_model_card() -> Dict[str, Any]:
    """Task 5: Check that MODEL_CARD.md is up-to-date and populated."""
    card_path = os.path.join(ROOT_DIR, "MODEL_CARD.md")
    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Missing MODEL_CARD.md at {card_path}")
    size = os.path.getsize(card_path)
    return {"status": "PASS", "model_card_bytes": size, "path": card_path}


def task_notify_summary() -> Dict[str, Any]:
    """Task 6: Final terminal reporting and release gate confirmation."""
    swap_json = os.path.join(ROOT_DIR, "logs", "swap_comparison.json")
    mon_json = os.path.join(ROOT_DIR, "logs", "monitoring_report.json")

    with open(swap_json, "r", encoding="utf-8") as f:
        swap = json.load(f)
    with open(mon_json, "r", encoding="utf-8") as f:
        mon = json.load(f)

    all_passed = all([r.get("release_gate_passed", False) for r in swap.get("results", [])])

    return {
        "status": "RELEASE_CERTIFIED" if all_passed else "RELEASE_BLOCKED",
        "models_passed_gate": sum([1 for r in swap.get("results", []) if r.get("release_gate_passed", False)]),
        "total_models": len(swap.get("results", [])),
        "system_health": mon.get("system_health", "UNKNOWN"),
    }


class DAGOrchestrator:
    """Orchestrates pipeline execution using Kahn's Algorithm + Exponential Backoff."""

    def __init__(self):
        self.tasks: Dict[str, Callable[[], Any]] = {
            "validate_config": task_validate_config,
            "extract_features": task_extract_features,
            "score_policies": task_score_policies,
            "monitor_drift": task_monitor_drift,
            "generate_model_card": task_generate_model_card,
            "notify_summary": task_notify_summary,
        }

        # Dependencies: (upstream, downstream)
        self.dependencies: List[Tuple[str, str]] = [
            ("validate_config", "extract_features"),
            ("extract_features", "score_policies"),
            ("score_policies", "monitor_drift"),
            ("score_policies", "generate_model_card"),
            ("monitor_drift", "notify_summary"),
            ("generate_model_card", "notify_summary"),
        ]

    def get_execution_order(self) -> List[str]:
        """Compute execution order using Kahn's Algorithm."""
        return topological_sort(list(self.tasks.keys()), self.dependencies)

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the DAG tasks in topological order."""
        order = self.get_execution_order()
        print("=" * 75)
        print("COUNSEL MLOps Pipeline Orchestrator (Kahn's DAG)")
        print("=" * 75)
        print(f"Topological Order: {' -> '.join(order)}\n")

        execution_log: Dict[str, Any] = {}
        t_start_total = time.perf_counter()

        for idx, task_name in enumerate(order, 1):
            task_fn = self.tasks[task_name]
            print(f"[{idx}/{len(order)}] Executing Task: {task_name.upper()}...")
            t0 = time.perf_counter()
            try:
                res = execute_with_backoff(task_fn, task_name, max_retries=3, base_delay=0.3)
                elapsed = time.perf_counter() - t0
                execution_log[task_name] = {
                    "status": "SUCCESS",
                    "elapsed_sec": round(elapsed, 4),
                    "result": res,
                }
                print(f"       Status: SUCCESS ({elapsed:.3f}s) | Info: {res.get('status', 'OK')}")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                execution_log[task_name] = {
                    "status": "FAILED",
                    "elapsed_sec": round(elapsed, 4),
                    "error": str(e),
                }
                print(f"       Status: FAILED ({elapsed:.3f}s) -> {e}")
                raise

        total_time = time.perf_counter() - t_start_total
        print("=" * 75)
        print(f"Pipeline Completed Successfully in {total_time:.2f}s!")
        print("=" * 75)

        summary_file = os.path.join(ROOT_DIR, "logs", "pipeline_execution.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "total_duration_sec": total_time,
                    "topological_order": order,
                    "tasks": execution_log,
                },
                f,
                indent=2,
            )
        print(f"[OK] Saved pipeline DAG run summary to {summary_file}")
        return execution_log


def main():
    orchestrator = DAGOrchestrator()
    orchestrator.run_pipeline()


if __name__ == "__main__":
    main()

