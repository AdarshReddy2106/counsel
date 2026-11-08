import os
from typing import Any, Dict

import numpy as np
import torch
from safetensors.torch import save_model

from .base import PolicyAdapter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class PPOPolicyAdapter(PolicyAdapter):
    """Adapter for the trained GCNActorCritic PPO checkpoint."""

    name = "ppo"
    family = "Graph Reinforcement Learning (GCN-PPO)"
    artifact_format = ".pt + .safetensors"

    def __init__(self, model_dir: str = "data/std-f5-c3", deterministic: bool = False):
        self.deterministic = deterministic
        full_dir = os.path.join(ROOT_DIR, model_dir)
        pt_path = os.path.join(full_dir, "pyt_save", "model.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Trained PPO model not found at {pt_path}")

        self.pt_path = pt_path
        try:
            self.model: Any = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            self.model = torch.load(pt_path, map_location="cpu")
        self.model.eval()

        registry_dir = os.path.join(ROOT_DIR, "data", "registry")
        os.makedirs(registry_dir, exist_ok=True)
        self.safetensors_path = os.path.join(registry_dir, "ppo_baseline.safetensors")
        if not os.path.exists(self.safetensors_path):
            save_model(self.model, self.safetensors_path)

        self.num_params = int(sum(p.numel() for p in self.model.parameters()))

    def _predict_impl(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        with torch.inference_mode():
            x = torch.as_tensor(obs_2d, dtype=torch.float32)
            m = torch.as_tensor(mask_1d, dtype=torch.float32)
            if self.deterministic:
                logits = self.model.pi._get_logits(x).clone()
                logits[m == 0] = float("-inf")
                return int(torch.argmax(logits).item())
            action, _, _ = self.model.step(x, m)
            return int(action)

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update(
            {
                "num_parameters": self.num_params,
                "source_checkpoint": os.path.relpath(self.pt_path, ROOT_DIR),
                "safetensors_artifact": os.path.relpath(self.safetensors_path, ROOT_DIR),
            }
        )
        return meta

