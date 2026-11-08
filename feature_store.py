import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from model.synthetic import set_slo
from model.utils.test_policy import load_policy_and_env
from schema import (
    DEFAULT_CONFIG_PATH,
    load_config,
    validate_action,
    validate_observation,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FeatureStore:
    OBS_COLUMNS = [f"obs_{i}" for i in range(24)]
    MASK_COLUMNS = [f"mask_{i}" for i in range(6)]
    RESOURCE_COLUMNS = [
        "fw_cpu",
        "ws_cpu",
        "db_cpu",
        "fw_mem",
        "ws_mem",
        "db_mem",
        "fw_util",
        "ws_util",
        "db_util",
        "arrival_rate",
        "act_type",
        "act_comp",
        "budget_overrun_ratio",
    ]
    TIME_COLUMNS = ["event_time", "ingestion_time", "processing_time"]

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        fs_cfg = self.cfg["feature_store"]
        self.parquet_path = os.path.join(ROOT_DIR, fs_cfg["offline_parquet_path"])
        self.metadata_path = os.path.join(ROOT_DIR, fs_cfg["metadata_json_path"])

    def materialize_from_ppo(self, num_episodes: Optional[int] = None) -> Dict:
        env_cfg = self.cfg["environment"]
        pol_cfg = self.cfg["policy"]
        fs_cfg = self.cfg["feature_store"]

        episodes = int(num_episodes if num_episodes is not None else fs_cfg["num_episodes"])
        seed = int(fs_cfg.get("seed", 42))
        step_interval = int(fs_cfg.get("step_interval_seconds", 5))

        np.random.seed(seed)
        torch.manual_seed(seed)

        port = int(env_cfg["load_server_port"])
        slo = int(env_cfg["slo_latency_ms"])
        freq = int(1e6 / slo)
        knob = float(env_cfg["knob"])
        set_slo(slo, freq, knob, port=port)

        model_dir = os.path.join(ROOT_DIR, pol_cfg["ppo_model_dir"])
        infer_log_dir = os.path.join(ROOT_DIR, fs_cfg["infer_log_dir"])
        os.makedirs(infer_log_dir, exist_ok=True)

        params = {
            "log_dir": infer_log_dir,
            "steps_per_epoch": int(env_cfg["steps_per_epoch"]),
            "budget": list(env_cfg["budget"]),
            "slo_latency": float(slo),
            "overrun_lim": float(env_cfg["overrun_lim"]),
            "mode": str(env_cfg["mode"]),
            "nconf": int(env_cfg["nconf"]),
            "ncomp": int(env_cfg["ncomp"]),
            "port": port,
        }

        env, get_action = load_policy_and_env(
            model_dir, pol_cfg["ppo_checkpoint"], params
        )
        flavors = list(env_cfg["flavors"])
        schema_ver = str(self.cfg["project"]["schema_version"])

        rows = []
        episode_returns = []
        episode_lengths = []
        total_slo_breaches = 0
        total_budget_overruns = 0
        contract_checks_passed = 0

        base_event_dt = datetime(2026, 11, 6, 0, 0, 0, tzinfo=timezone.utc)
        global_step = 0

        obs, mask = env.reset()
        ep_id = 0
        ep_ret = 0.0
        ep_len = 0

        while ep_id < episodes:
            ingestion_dt = datetime.now(timezone.utc)

            v_obs, v_mask = validate_observation(
                obs, mask, ncomp=env.ncomp, nactions=env.action_space.n
            )

            raw_act = get_action(v_obs, v_mask)
            act = validate_action(int(raw_act), v_mask, nactions=env.action_space.n)
            contract_checks_passed += 1

            for c in env.components:
                c.compute_resources()
            fw, ws, db = env.components[0], env.components[1], env.components[2]
            cur_arrival = float(v_obs[0, 6])
            cur_act_type = int(env.act_type)
            cur_act_comp = int(env.act_comp)

            next_obs, next_mask, reward, done, _info, _util_hist, breach = env.step(act)
            overrun_ratio = float(env.chain.get_budget_overrun())
            is_overrun = overrun_ratio > float(env.overrun_lim)

            event_dt = base_event_dt + timedelta(seconds=global_step * step_interval)
            processing_dt = datetime.now(timezone.utc)

            row = {
                "episode_id": ep_id,
                "step_in_episode": ep_len,
                "global_step": global_step,
                "event_time": event_dt.isoformat(),
                "ingestion_time": ingestion_dt.isoformat(),
                "processing_time": processing_dt.isoformat(),
                "schema_version": schema_ver,
            }

            flat_obs = v_obs.reshape(-1)
            for i, val in enumerate(flat_obs):
                row[f"obs_{i}"] = float(val)
            for i, val in enumerate(v_mask):
                row[f"mask_{i}"] = float(val)

            row.update(
                {
                    "fw_cpu": int(fw.cpu),
                    "ws_cpu": int(ws.cpu),
                    "db_cpu": int(db.cpu),
                    "fw_mem": int(fw.mem),
                    "ws_mem": int(ws.mem),
                    "db_mem": int(db.mem),
                    "fw_util": float(v_obs[0, 5]),
                    "ws_util": float(v_obs[1, 5]),
                    "db_util": float(v_obs[2, 5]),
                    "arrival_rate": cur_arrival,
                    "act_type": cur_act_type,
                    "act_comp": cur_act_comp,
                    "budget_overrun_ratio": overrun_ratio,
                    "action": act,
                    "action_flavor": flavors[act],
                    "reward": float(reward),
                    "slo_breach": bool(breach),
                    "budget_overrun_flag": bool(is_overrun),
                }
            )
            rows.append(row)

            ep_ret += float(reward)
            ep_len += 1
            global_step += 1
            if breach:
                total_slo_breaches += 1
            if is_overrun:
                total_budget_overruns += 1

            obs, mask = next_obs, next_mask

            if done:
                episode_returns.append(ep_ret)
                episode_lengths.append(ep_len)
                print(
                    f"Episode {ep_id:02d} | Steps: {ep_len:4d} | EpRet: {ep_ret:8.3f}"
                )
                obs, mask = env.reset()
                ep_ret = 0.0
                ep_len = 0
                ep_id += 1

        df = pd.DataFrame(rows)
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
        df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], utc=True)
        df["processing_time"] = pd.to_datetime(df["processing_time"], utc=True)

        os.makedirs(os.path.dirname(self.parquet_path), exist_ok=True)
        df.to_parquet(self.parquet_path, engine="pyarrow", index=False)

        model_pt_path = os.path.join(model_dir, "pyt_save", "model.pt")
        metadata = {
            "schema_version": schema_ver,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "lineage": {
                "config_path": os.path.relpath(self.config_path, ROOT_DIR),
                "config_sha256": _sha256_file(self.config_path),
                "source_model_path": os.path.relpath(model_pt_path, ROOT_DIR),
                "source_model_sha256": _sha256_file(model_pt_path),
                "parquet_path": os.path.relpath(self.parquet_path, ROOT_DIR),
                "parquet_sha256": _sha256_file(self.parquet_path),
            },
            "feature_groups": {
                "graph_and_zscore_obs": self.OBS_COLUMNS,
                "action_mask": self.MASK_COLUMNS,
                "readable_resources": self.RESOURCE_COLUMNS,
                "timestamps": self.TIME_COLUMNS,
                "target": ["action", "action_flavor", "reward", "slo_breach"],
            },
            "summary": {
                "num_episodes": episodes,
                "total_rows": int(len(df)),
                "contract_checks_passed": contract_checks_passed,
                "average_ep_ret": float(np.mean(episode_returns)),
                "std_ep_ret": float(np.std(episode_returns)),
                "max_ep_ret": float(np.max(episode_returns)),
                "min_ep_ret": float(np.min(episode_returns)),
                "average_ep_len": float(np.mean(episode_lengths)),
                "total_slo_breaches": int(total_slo_breaches),
                "total_budget_overruns": int(total_budget_overruns),
                "action_distribution": {
                    str(k): int(v)
                    for k, v in df["action"].value_counts().sort_index().items()
                },
            },
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def load_dataframe(self) -> pd.DataFrame:
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(
                f"Parquet dataset not found at {self.parquet_path}. Run feature_store.py first."
            )
        return pd.read_parquet(self.parquet_path, engine="pyarrow")

    def get_training_view(
        self, as_of_time: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        df = self.load_dataframe()
        if as_of_time is not None:
            cutoff = pd.to_datetime(as_of_time, utc=True)
            df = df[df["event_time"] <= cutoff].copy()

        x_obs = df[self.OBS_COLUMNS].to_numpy(dtype=np.float32)
        x_mask = df[self.MASK_COLUMNS].to_numpy(dtype=np.float32)
        y_act = df["action"].to_numpy(dtype=np.int64)
        return x_obs, x_mask, y_act, df


if __name__ == "__main__":
    store = FeatureStore()
    meta = store.materialize_from_ppo()

    x_all, m_all, y_all, df_all = store.get_training_view()
    midpoint_ts = df_all["event_time"].iloc[len(df_all) // 2].isoformat()
    x_pit, _, _, _ = store.get_training_view(as_of_time=midpoint_ts)

    print(f"Saved {meta['summary']['total_rows']} rows to {meta['lineage']['parquet_path']}")
    print(f"Average return: {meta['summary']['average_ep_ret']:.2f}, overruns: {meta['summary']['total_budget_overruns']}")
    print(f"Point-in-time rows <= {midpoint_ts}: {len(x_pit)} / {len(x_all)}")
