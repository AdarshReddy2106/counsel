import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from adapters import AVAILABLE_POLICIES, get_policy_adapter
from model.env import CloudEnv
from model.synthetic import set_slo
from schema import DEFAULT_CONFIG_PATH, load_config

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_single_policy(
    policy_name: str,
    cfg: Dict[str, Any],
    num_episodes: int = 20,
) -> Dict[str, Any]:
    """Run `policy_name` on `num_episodes` with a fixed workload seed."""
    env_cfg = cfg["environment"]
    pol_cfg = cfg["policy"]
    mon_cfg = cfg["monitoring"]

    eval_seed = int(pol_cfg.get("eval_seed", 42))
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    adapter = get_policy_adapter(policy_name)

    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    port = int(env_cfg["load_server_port"])
    slo = int(env_cfg["slo_latency_ms"])
    freq = int(1e6 / slo)
    knob = float(env_cfg["knob"])
    set_slo(slo, freq, knob, port=port)

    log_dir = os.path.join(ROOT_DIR, "infer_logs", f"swap_{policy_name}")
    os.makedirs(log_dir, exist_ok=True)

    env = CloudEnv(
        log_dir=log_dir,
        steps_per_epoch=int(env_cfg["steps_per_epoch"]),
        budget=list(env_cfg["budget"]),
        slo_latency=float(slo),
        overrun_lim=float(env_cfg["overrun_lim"]),
        mode=str(env_cfg["mode"]),
        nconf=int(env_cfg["nconf"]),
        ncomp=int(env_cfg["ncomp"]),
        port=port,
    )

    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    infer_latencies_ms: List[float] = []
    total_steps = 0
    slo_breaches = 0
    budget_overruns = 0
    utilizations: List[float] = []

    obs, mask = env.reset()
    adapter.reset_episode()
    ep_id = 0
    ep_ret = 0.0
    ep_len = 0

    while ep_id < num_episodes:
        t0 = time.perf_counter()
        action = adapter.predict(obs, mask)
        t1 = time.perf_counter()
        infer_latencies_ms.append((t1 - t0) * 1000.0)

        utilizations.append(float(np.mean(obs[:, 5])))
        obs, mask, reward, done, _info, _util_hist, breach = env.step(action)

        ep_ret += float(reward)
        ep_len += 1
        total_steps += 1
        if breach:
            slo_breaches += 1
        if env.chain.get_budget_overrun() > float(env.overrun_lim):
            budget_overruns += 1

        if done:
            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)
            obs, mask = env.reset()
            adapter.reset_episode()
            ep_ret = 0.0
            ep_len = 0
            ep_id += 1

    avg_ret = float(np.mean(ep_returns))
    min_gate_reward = float(mon_cfg["release_gate_min_reward"])
    max_gate_overruns = int(mon_cfg["release_gate_max_budget_overruns"])
    gate_passed = (avg_ret >= min_gate_reward) and (budget_overruns <= max_gate_overruns)

    return {
        "policy": policy_name,
        "metadata": adapter.get_metadata(),
        "episodes": num_episodes,
        "total_steps": total_steps,
        "average_ep_ret": avg_ret,
        "std_ep_ret": float(np.std(ep_returns)),
        "max_ep_ret": float(np.max(ep_returns)),
        "min_ep_ret": float(np.min(ep_returns)),
        "average_ep_len": float(np.mean(ep_lengths)),
        "mean_infer_latency_ms": float(np.mean(infer_latencies_ms)),
        "p99_infer_latency_ms": float(np.percentile(infer_latencies_ms, 99)),
        "slo_breaches": slo_breaches,
        "slo_breach_rate": float(slo_breaches / max(1, total_steps)),
        "budget_overruns": budget_overruns,
        "mean_utilization": float(np.mean(utilizations)),
        "release_gate_passed": gate_passed,
    }


def run_swap_benchmark(
    policies: Optional[List[str]] = None,
    num_episodes: Optional[int] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """Evaluate all selected policies and write `logs/swap_comparison.json`."""
    cfg = load_config(config_path)
    selected = policies or list(AVAILABLE_POLICIES)
    episodes = int(
        num_episodes if num_episodes is not None else cfg["feature_store"]["num_episodes"]
    )

    results: List[Dict[str, Any]] = []
    for pol in selected:
        print(f"\n>>> Plugging IN policy: [{pol}] ({episodes} episodes)...")
        res = evaluate_single_policy(pol, cfg, num_episodes=episodes)
        results.append(res)
        print(
            f"    Done [{pol}] -> AvgEpRet={res['average_ep_ret']:.2f}, "
            f"InferP99={res['p99_infer_latency_ms']:.3f}ms, "
            f"Overruns={res['budget_overruns']}, Gate={'PASS' if res['release_gate_passed'] else 'FAIL'}"
        )

    out_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "swap_comparison.json")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_config_default": cfg["policy"]["active_policy"],
        "episodes_per_policy": episodes,
        "release_gate_criteria": {
            "min_average_ep_ret": cfg["monitoring"]["release_gate_min_reward"],
            "max_budget_overruns": cfg["monitoring"]["release_gate_max_budget_overruns"],
        },
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 92)
    print(
        f"{'Policy':<15} | {'AvgEpRet':>9} | {'InferMean(ms)':>13} | {'InferP99(ms)':>12} | "
        f"{'SLOBreach%':>10} | {'Overruns':>8} | {'Gate':>6}"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{r['policy']:<15} | {r['average_ep_ret']:>9.2f} | "
            f"{r['mean_infer_latency_ms']:>13.3f} | {r['p99_infer_latency_ms']:>12.3f} | "
            f"{r['slo_breach_rate'] * 100:>9.2f}% | {r['budget_overruns']:>8d} | "
            f"{'PASS' if r['release_gate_passed'] else 'FAIL':>6}"
        )
    print("=" * 92)
    print(f"Saved full swap benchmark report to: {os.path.relpath(out_path, ROOT_DIR)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark COUNSEL Plug-In Policies")
    parser.add_argument(
        "--episodes", "-e", type=int, default=None, help="Episodes per policy (default: 20)"
    )
    parser.add_argument(
        "--policies",
        "-p",
        nargs="+",
        default=None,
        help=f"Subset of policies from {AVAILABLE_POLICIES}",
    )
    args = parser.parse_args()
    run_swap_benchmark(policies=args.policies, num_episodes=args.episodes)

