import os
import sys
import warnings

# Suppress noisy external gym deprecation warnings from transformers
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Ensure repository root is on sys.path for direct execution and package imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
from collections import deque  # noqa: E402
from typing import Any, Deque, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import load_model, save_model  # noqa: E402
from transformers import (  # noqa: E402
    DecisionTransformerConfig,
    DecisionTransformerModel,
    PatchTSTConfig,
    PatchTSTForClassification,
    PatchTSTModel,
)

from feature_store import FeatureStore  # noqa: E402

try:
    from .base import PolicyAdapter, apply_action_mask_to_scores  # noqa: E402
except ImportError:
    from adapters.base import PolicyAdapter, apply_action_mask_to_scores  # noqa: E402


def _derive_outcome_aware_targets(
    obs_3d: np.ndarray,
    mask_2d: np.ndarray,
    ppo_actions: np.ndarray,
    slo_breaches: np.ndarray,
    mode: str = "dt",
) -> np.ndarray:
    """Adjust training targets using logged `slo_breach` outcomes from `offline_store.parquet`."""
    targets = ppo_actions.copy()
    for i in range(len(targets)):
        is_scale_up = bool(np.all(mask_2d[i] > 0.5))
        max_u = float(np.max(obs_3d[i, :, 5]))
        if slo_breaches[i] and is_scale_up:
            if mode == "dt":
                targets[i] = 5 if max_u >= 0.95 else (4 if max_u >= 0.82 else 3)
            else:
                targets[i] = 4 if max_u >= 0.90 else 3
        elif (not slo_breaches[i]) and is_scale_up and max_u < 0.48 and mode == "patchtst":
            targets[i] = 1
    return targets


def _extract_10ch_telemetry(obs_2d: np.ndarray) -> np.ndarray:
    """Extract 10 telemetry channels (utilization, CPU/Mem z-scores, and arrival rate)."""
    utils = obs_2d[:, 5]
    cpu_z = obs_2d[:, 3]
    mem_z = obs_2d[:, 4]
    rate = np.array([obs_2d[0, 6] / 100.0], dtype=np.float32)
    return np.concatenate([utils, cpu_z, mem_z, rate]).astype(np.float32)


class DecisionTransformerPolicyAdapter(PolicyAdapter):
    """Offline-RL Decision Transformer policy."""

    name = "decision_transformer"
    family = "Hugging Face Offline-RL Transformer (DecisionTransformerModel)"
    artifact_format = ".safetensors + .json"
    hf_repo_id = "edbeeching/decision-transformer-gym-hopper-medium"
    context_len = 8
    rtg_scale = 600.0
    target_rtg = 1.25

    def __init__(self, force_retrain: bool = False):
        registry_dir = os.path.join(ROOT_DIR, "data", "registry")
        os.makedirs(registry_dir, exist_ok=True)
        self.weights_path = os.path.join(
            registry_dir, "decision_transformer.safetensors"
        )
        self.meta_path = os.path.join(registry_dir, "decision_transformer_meta.json")
        self.meta: Dict[str, Any] = {}

        self.dt_config = DecisionTransformerConfig(
            state_dim=30,
            act_dim=6,
            hidden_size=128,
            n_layer=3,
            n_head=1,
            max_ep_len=2048,
            action_tanh=False,
        )
        self.model: Any = DecisionTransformerModel(self.dt_config)
        self.loaded_from_hub = False

        if force_retrain or not (
            os.path.exists(self.weights_path) and os.path.exists(self.meta_path)
        ):
            self._init_and_train()
        else:
            load_model(self.model, self.weights_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        self.model.eval()
        self._states_buf: Deque[np.ndarray] = deque(maxlen=self.context_len)
        self._actions_buf: Deque[np.ndarray] = deque(maxlen=self.context_len)
        self._rtg_buf: Deque[float] = deque(maxlen=self.context_len)
        self._t_step: int = 0
        self.reset_episode()

    def reset_episode(self) -> None:
        self._states_buf.clear()
        self._actions_buf.clear()
        self._rtg_buf.clear()
        self._t_step = 0

    def _try_load_hub_encoder(self) -> bool:
        try:
            pre = DecisionTransformerModel.from_pretrained(
                self.hf_repo_id, local_files_only=True
            )
            self.model.encoder.load_state_dict(pre.encoder.state_dict(), strict=False)
            return True
        except Exception:
            return False

    def _build_sequences(
        self, df_train: pd.DataFrame, store: FeatureStore
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = self.context_len
        states_list: List[np.ndarray] = []
        actions_in_list: List[np.ndarray] = []
        rtgs_list: List[np.ndarray] = []
        timesteps_list: List[np.ndarray] = []
        attn_list: List[np.ndarray] = []
        targets_list: List[int] = []

        for _ep_id, grp in df_train.groupby("episode_id", sort=False):
            obs_arr = grp[store.OBS_COLUMNS].to_numpy(dtype=np.float32)
            mask_arr = grp[store.MASK_COLUMNS].to_numpy(dtype=np.float32)
            feat_arr = np.hstack([obs_arr, mask_arr])  # (T, 30)
            ppo_act = grp["action"].to_numpy(dtype=np.int64)
            breach_arr = grp["slo_breach"].to_numpy(dtype=bool)
            act_arr = _derive_outcome_aware_targets(
                obs_arr.reshape(-1, 3, 8), mask_arr, ppo_act, breach_arr, mode="dt"
            )
            rew_arr = grp["reward"].to_numpy(dtype=np.float32)

            # Reverse cumulative sum for Return-to-Go
            rtg_arr = np.cumsum(rew_arr[::-1])[::-1] / self.rtg_scale
            ep_len = len(grp)

            # Sample stride=2 across episode steps for fast, diverse trajectory training
            for t in range(0, ep_len, 2):
                start = max(0, t - k + 1)
                win_len = t - start + 1
                pad_len = k - win_len

                s_win = np.zeros((k, 30), dtype=np.float32)
                a_win = np.zeros((k, 6), dtype=np.float32)
                r_win = np.zeros((k, 1), dtype=np.float32)
                ts_win = np.zeros((k,), dtype=np.int64)
                am_win = np.zeros((k,), dtype=np.float32)

                s_win[pad_len:] = feat_arr[start:t + 1]
                r_win[pad_len:, 0] = rtg_arr[start:t + 1]
                ts_win[pad_len:] = np.arange(start, t + 1, dtype=np.int64)
                am_win[pad_len:] = 1.0

                # Previous actions placed in slots before final slot; 0 vector for current query
                if win_len > 1:
                    prev_acts = act_arr[start:t]
                    a_win[pad_len:k - 1, :] = np.eye(6, dtype=np.float32)[prev_acts]

                states_list.append(s_win)
                actions_in_list.append(a_win)
                rtgs_list.append(r_win)
                timesteps_list.append(ts_win)
                attn_list.append(am_win)
                targets_list.append(int(act_arr[t]))

        return (
            np.stack(states_list),
            np.stack(actions_in_list),
            np.stack(rtgs_list),
            np.stack(timesteps_list),
            np.stack(attn_list),
            np.array(targets_list, dtype=np.int64),
        )

    def _init_and_train(self) -> None:
        torch.manual_seed(42)
        self.loaded_from_hub = self._try_load_hub_encoder()

        store = FeatureStore()
        df_full = store.load_dataframe()
        cutoff_idx = int(len(df_full) * 0.80)
        cutoff_ts = df_full["event_time"].iloc[cutoff_idx].isoformat()
        _, _, _, df_train = store.get_training_view(as_of_time=cutoff_ts)

        s_np, a_np, r_np, ts_np, am_np, y_np = self._build_sequences(df_train, store)
        s_t = torch.as_tensor(s_np, dtype=torch.float32)
        a_t = torch.as_tensor(a_np, dtype=torch.float32)
        r_t = torch.as_tensor(r_np, dtype=torch.float32)
        ts_t = torch.as_tensor(ts_np, dtype=torch.long)
        am_t = torch.as_tensor(am_np, dtype=torch.float32)
        y_t = torch.as_tensor(y_np, dtype=torch.long)

        # Weight trajectory windows by normalized Return-to-Go so high-return behavior dominates
        seq_weights = torch.clamp(r_t[:, -1, 0], min=0.15)

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.08)
        batch_sz = 128

        for _epoch in range(18):
            perm = torch.randperm(len(s_t))
            for i in range(0, len(s_t), batch_sz):
                idx = perm[i:i + batch_sz]
                opt.zero_grad()
                out = self.model(
                    states=s_t[idx],
                    actions=a_t[idx],
                    returns_to_go=r_t[idx],
                    timesteps=ts_t[idx],
                    attention_mask=am_t[idx],
                )
                assert out.action_preds is not None
                logits = out.action_preds[:, -1, :]
                loss = (loss_fn(logits, y_t[idx]) * seq_weights[idx]).mean()
                loss.backward()
                opt.step()

        self.model.eval()
        with torch.inference_mode():
            out_all = self.model(
                states=s_t,
                actions=a_t,
                returns_to_go=r_t,
                timesteps=ts_t,
                attention_mask=am_t,
            )
            assert out_all.action_preds is not None
            preds = torch.argmax(out_all.action_preds[:, -1, :], dim=-1)
            train_acc = float((preds == y_t).float().mean().item())

        save_model(self.model, self.weights_path)
        self.meta = {
            "model_name": self.name,
            "hf_repo_id": self.hf_repo_id,
            "loaded_from_hub": self.loaded_from_hub,
            "context_length": self.context_len,
            "target_return_conditioning": self.target_rtg * self.rtg_scale,
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "train_sequences": int(len(s_t)),
            "train_accuracy": train_acc,
            "safetensors_artifact": os.path.relpath(self.weights_path, ROOT_DIR),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def _predict_impl(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        k = self.context_len
        feat = np.concatenate([obs_2d.reshape(-1), mask_1d]).astype(np.float32)
        self._states_buf.append(feat)
        # Decay conditioned RTG slightly along episode while keeping optimistic floor
        rem_rtg = max(0.45, self.target_rtg - (self._t_step / 1200.0))
        self._rtg_buf.append(rem_rtg)

        win_len = len(self._states_buf)
        pad_len = k - win_len

        s_win = np.zeros((1, k, 30), dtype=np.float32)
        a_win = np.zeros((1, k, 6), dtype=np.float32)
        r_win = np.zeros((1, k, 1), dtype=np.float32)
        ts_win = np.zeros((1, k), dtype=np.int64)
        am_win = np.zeros((1, k), dtype=np.float32)

        s_win[0, pad_len:] = np.stack(list(self._states_buf))
        r_win[0, pad_len:, 0] = np.array(list(self._rtg_buf), dtype=np.float32)
        start_t = max(0, self._t_step - win_len + 1)
        ts_win[0, pad_len:] = np.arange(
            start_t, start_t + win_len, dtype=np.int64
        ).clip(0, 2047)
        am_win[0, pad_len:] = 1.0

        if len(self._actions_buf) > 0:
            prev_a = list(self._actions_buf)[-(win_len - 1):] if win_len > 1 else []
            if prev_a:
                a_win[0, pad_len:k - 1] = np.stack(prev_a)

        with torch.inference_mode():
            out = self.model(
                states=torch.from_numpy(s_win),
                actions=torch.from_numpy(a_win),
                returns_to_go=torch.from_numpy(r_win),
                timesteps=torch.from_numpy(ts_win),
                attention_mask=torch.from_numpy(am_win),
            )
            assert out.action_preds is not None
            logits = out.action_preds[0, -1].cpu().numpy()

        # Proactive high-RTG headroom boost when utilization nears SLO saturation
        max_util = float(np.max(obs_2d[:, 5]))
        if max_util > 0.78:
            logits[5] += 0.65
            logits[4] += 0.45

        chosen = apply_action_mask_to_scores(logits, mask_1d)
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[chosen] = 1.0
        self._actions_buf.append(one_hot)
        self._t_step += 1
        return chosen

    def get_metadata(self) -> Dict[str, Any]:
        out = super().get_metadata()
        out.update(self.meta)
        return out


class _PatchTSTWithContext(nn.Module):
    """Combines Hugging Face PatchTST temporal encoder with instantaneous mask/target context."""

    def __init__(self, pt_config: PatchTSTConfig):
        super().__init__()
        self.patchtst: Any = PatchTSTForClassification(pt_config)
        self.ctx_proj = nn.Sequential(
            nn.Linear(6 + 9, 32),  # 6-way PatchTST logits + (3 target flags + 6 mask bits)
            nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, past_values: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        pt_out = self.patchtst(past_values=past_values)
        base_logits = pt_out.prediction_logits
        assert base_logits is not None
        return base_logits + self.ctx_proj(torch.cat([base_logits, ctx], dim=-1))


class PatchTSTPolicyAdapter(PolicyAdapter):
    """Hugging Face Multivariate Time-Series Transformer (`PatchTSTForClassification`)."""

    name = "patchtst"
    family = "Hugging Face Time-Series Transformer (PatchTSTForClassification)"
    artifact_format = ".safetensors + .json"
    hf_repo_id = "ibm-granite/granite-timeseries-patchtst"
    context_len = 16
    num_channels = 10

    def __init__(self, force_retrain: bool = False):
        registry_dir = os.path.join(ROOT_DIR, "data", "registry")
        os.makedirs(registry_dir, exist_ok=True)
        self.weights_path = os.path.join(registry_dir, "patchtst.safetensors")
        self.meta_path = os.path.join(registry_dir, "patchtst_meta.json")
        self.meta: Dict[str, Any] = {}

        self.pt_config = PatchTSTConfig(
            num_input_channels=self.num_channels,
            context_length=self.context_len,
            patch_length=4,
            patch_stride=2,
            d_model=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            ffn_dim=256,
            num_targets=6,
        )
        self.net: Any = _PatchTSTWithContext(self.pt_config)
        self.loaded_from_hub = False

        if force_retrain or not (
            os.path.exists(self.weights_path) and os.path.exists(self.meta_path)
        ):
            self._init_and_train()
        else:
            load_model(self.net, self.weights_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        self.net.eval()
        self._ts_buf: Deque[np.ndarray] = deque(maxlen=self.context_len)
        self.reset_episode()

    def reset_episode(self) -> None:
        self._ts_buf.clear()

    def _try_load_hub_encoder(self) -> bool:
        try:
            pre = PatchTSTModel.from_pretrained(self.hf_repo_id, local_files_only=True)
            # Copy matching attention & LayerNorm tensors from IBM Granite PatchTST
            own_sd = self.net.patchtst.model.encoder.state_dict()
            pre_sd = pre.encoder.state_dict()
            matched = {
                k: v
                for k, v in pre_sd.items()
                if k in own_sd and own_sd[k].shape == v.shape
            }
            if matched:
                own_sd.update(matched)
                self.net.patchtst.model.encoder.load_state_dict(own_sd)
                return True
        except Exception:
            pass
        return False

    def _build_windows(
        self, df_train: pd.DataFrame, store: FeatureStore
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        win = self.context_len
        past_list: List[np.ndarray] = []
        ctx_list: List[np.ndarray] = []
        y_list: List[int] = []
        w_list: List[float] = []

        for _ep_id, grp in df_train.groupby("episode_id", sort=False):
            obs_arr = grp[store.OBS_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 3, 8)
            mask_arr = grp[store.MASK_COLUMNS].to_numpy(dtype=np.float32)
            ppo_act = grp["action"].to_numpy(dtype=np.int64)
            breach_arr = grp["slo_breach"].to_numpy(dtype=bool)
            act_arr = _derive_outcome_aware_targets(
                obs_arr, mask_arr, ppo_act, breach_arr, mode="patchtst"
            )
            rew_arr = grp["reward"].to_numpy(dtype=np.float32)

            ch_arr = np.stack([_extract_10ch_telemetry(o) for o in obs_arr])  # (T, 10)
            ep_len = len(grp)

            for t in range(0, ep_len, 2):
                start = max(0, t - win + 1)
                seg = ch_arr[start:t + 1]
                if len(seg) < win:
                    # Left-pad with earliest step in episode
                    pad = np.repeat(seg[:1], win - len(seg), axis=0)
                    seg = np.vstack([pad, seg])

                ctx = np.concatenate([obs_arr[t, :, 7], mask_arr[t]]).astype(np.float32)
                past_list.append(seg)
                ctx_list.append(ctx)
                y_list.append(int(act_arr[t]))
                w_list.append(float(max(0.10, rew_arr[t])))

        return (
            np.stack(past_list),
            np.stack(ctx_list),
            np.array(y_list, dtype=np.int64),
            np.array(w_list, dtype=np.float32),
        )

    def _init_and_train(self) -> None:
        torch.manual_seed(42)
        self.loaded_from_hub = self._try_load_hub_encoder()

        store = FeatureStore()
        df_full = store.load_dataframe()
        cutoff_idx = int(len(df_full) * 0.80)
        cutoff_ts = df_full["event_time"].iloc[cutoff_idx].isoformat()
        _, _, _, df_train = store.get_training_view(as_of_time=cutoff_ts)

        p_np, c_np, y_np, w_np = self._build_windows(df_train, store)
        p_t = torch.as_tensor(p_np, dtype=torch.float32)
        c_t = torch.as_tensor(c_np, dtype=torch.float32)
        y_t = torch.as_tensor(y_np, dtype=torch.long)
        w_t = torch.as_tensor(w_np, dtype=torch.float32)

        self.net.train()
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.06)
        batch_sz = 128

        for _epoch in range(15):
            perm = torch.randperm(len(p_t))
            for i in range(0, len(p_t), batch_sz):
                idx = perm[i:i + batch_sz]
                opt.zero_grad()
                logits = self.net(p_t[idx], c_t[idx])
                loss = (loss_fn(logits, y_t[idx]) * w_t[idx]).mean()
                loss.backward()
                opt.step()

        self.net.eval()
        with torch.inference_mode():
            preds = torch.argmax(self.net(p_t, c_t), dim=-1)
            train_acc = float((preds == y_t).float().mean().item())

        save_model(self.net, self.weights_path)
        self.meta = {
            "model_name": self.name,
            "hf_repo_id": self.hf_repo_id,
            "loaded_from_hub": self.loaded_from_hub,
            "context_length": self.context_len,
            "patch_length": 4,
            "patch_stride": 2,
            "num_input_channels": self.num_channels,
            "num_parameters": int(sum(p.numel() for p in self.net.parameters())),
            "train_windows": int(len(p_t)),
            "train_accuracy": train_acc,
            "safetensors_artifact": os.path.relpath(self.weights_path, ROOT_DIR),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def _predict_impl(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        ch = _extract_10ch_telemetry(obs_2d)
        self._ts_buf.append(ch)

        seg = np.stack(list(self._ts_buf))
        if len(seg) < self.context_len:
            pad = np.repeat(seg[:1], self.context_len - len(seg), axis=0)
            seg = np.vstack([pad, seg])

        ctx = np.concatenate([obs_2d[:, 7], mask_1d]).astype(np.float32)

        with torch.inference_mode():
            p_in = torch.from_numpy(seg).unsqueeze(0)
            c_in = torch.from_numpy(ctx).unsqueeze(0)
            logits = self.net(p_in, c_in)[0].cpu().numpy()

        # Trend-aware spike anticipation: if traffic jumped >15% over the 16-step window, favor proactive capacity
        rate_delta = float(seg[-1, 9] - np.mean(seg[:4, 9]))
        if rate_delta > 0.08:
            logits[5] += 0.50
            logits[3] += 0.35

        return apply_action_mask_to_scores(logits, mask_1d)

    def get_metadata(self) -> Dict[str, Any]:
        out = super().get_metadata()
        out.update(self.meta)
        return out


if __name__ == "__main__":
    print("[TEST] Running Hugging Face Policy Adapters Verification...")

    # Test sample telemetry: 3 microservices x 8 features
    sample_obs = np.zeros((3, 8), dtype=np.float32)
    sample_obs[:, 0:3] = np.eye(3)
    sample_obs[:, 5] = [0.45, 0.70, 0.90]  # DB is high utilization
    sample_obs[:, 6] = 35.0  # High arrival rate
    sample_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # 1. Test Decision Transformer
    print("-> Loading DecisionTransformerPolicyAdapter...")
    dt = DecisionTransformerPolicyAdapter()
    dt_act = dt.predict(sample_obs, sample_mask)
    assert 0 <= dt_act <= 5
    assert sample_mask[dt_act] == 1.0
    print(f"   [OK] DecisionTransformer predicted action: {dt_act} (valid mask: {sample_mask[dt_act] == 1.0})")
    print(f"   Model metadata: params={dt.meta.get('num_parameters')}, format={dt.artifact_format}")

    # 2. Test PatchTST
    print("-> Loading PatchTSTPolicyAdapter...")
    pt = PatchTSTPolicyAdapter()
    pt_act = pt.predict(sample_obs, sample_mask)
    assert 0 <= pt_act <= 5
    assert sample_mask[pt_act] == 1.0
    print(f"   [OK] PatchTST predicted action: {pt_act} (valid mask: {sample_mask[pt_act] == 1.0})")
    print(f"   Model metadata: params={pt.meta.get('num_parameters')}, format={pt.artifact_format}")

    print("[SUCCESS] All Hugging Face adapters loaded, verified, and executed cleanly!")
