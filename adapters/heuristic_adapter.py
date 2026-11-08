from typing import Any, Dict, List

import numpy as np

from .base import PolicyAdapter


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

