import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

try:
    from .base import PolicyAdapter  # noqa: E402
except ImportError:
    from adapters.base import PolicyAdapter  # noqa: E402


class HeuristicPolicyAdapter(PolicyAdapter):
    """Rule-based utilization threshold autoscaler."""

    name = "heuristic"
    family = "Rule-Based Reactive Threshold Controller"
    artifact_format = "pure-python-rules (0 MB)"

    def __init__(self, scale_up_thresh: float = 0.80, scale_down_thresh: float = 0.45):
        self.scale_up_thresh = scale_up_thresh
        self.scale_down_thresh = scale_down_thresh

    @staticmethod
    def _first_valid(preference_order: List[int], mask_1d: np.ndarray) -> int:
        for act in preference_order:
            if mask_1d[act] > 0.5:
                return act
        return 0

    def _predict_impl(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        utils = obs_2d[:, 5]
        target_comp = int(np.argmax(obs_2d[:, 7]))
        target_util = float(utils[target_comp])
        max_util = float(np.max(utils))

        if target_util >= self.scale_up_thresh or max_util >= 0.90:
            return self._first_valid([5, 4, 3, 2, 1, 0], mask_1d)

        if target_util >= 0.65:
            return self._first_valid([3, 2, 4, 1, 0], mask_1d)

        if target_util < self.scale_down_thresh:
            return self._first_valid([2, 1, 3, 0], mask_1d)

        return 0

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update(
            {
                "num_parameters": 0,
                "scale_up_threshold": self.scale_up_thresh,
                "scale_down_threshold": self.scale_down_thresh,
            }
        )
        return meta


if __name__ == "__main__":
    print("[TEST] Running HeuristicPolicyAdapter Verification...")
    h = HeuristicPolicyAdapter()
    sample_obs = np.zeros((3, 8), dtype=np.float32)
    sample_obs[:, 0:3] = np.eye(3)
    sample_obs[:, 5] = [0.45, 0.70, 0.90]
    sample_obs[:, 6] = 35.0
    sample_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    act = h.predict(sample_obs, sample_mask)
    assert 0 <= act <= 5
    assert sample_mask[act] == 1.0
    print(f"   [OK] Heuristic predicted action: {act} (valid mask: {sample_mask[act] == 1.0})")
    print(f"   Model metadata: params={h.get_metadata().get('num_parameters')}, format={h.artifact_format}")
    print("[SUCCESS] Heuristic adapter loaded and executed cleanly!")
