import os
import sys
from typing import Any, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_model  # noqa: E402

try:
    from .base import PolicyAdapter  # noqa: E402
except ImportError:
    from adapters.base import PolicyAdapter  # noqa: E402


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


if __name__ == "__main__":
    print("[TEST] Running PPOPolicyAdapter Verification...")
    ppo = PPOPolicyAdapter()
    sample_obs = np.zeros((3, 8), dtype=np.float32)
    sample_obs[:, 0:3] = np.eye(3)
    sample_obs[:, 5] = [0.45, 0.70, 0.90]
    sample_obs[:, 6] = 35.0
    sample_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    act = ppo.predict(sample_obs, sample_mask)
    assert 0 <= act <= 5
    assert sample_mask[act] == 1.0
    print(f"   [OK] PPO predicted action: {act} (valid mask: {sample_mask[act] == 1.0})")
    print(f"   Model metadata: params={ppo.num_params}, format={ppo.artifact_format}")
    print("[SUCCESS] PPO adapter loaded and executed cleanly!")
