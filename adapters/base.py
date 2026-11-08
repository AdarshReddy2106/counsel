from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from schema import validate_action, validate_observation


def apply_action_mask_to_scores(scores: np.ndarray, mask: np.ndarray) -> int:
    """Mask out disallowed actions and return the highest-scoring legal action index."""
    masked_scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    mask_bool = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
    if not np.any(mask_bool):
        return 0
    masked_scores[~mask_bool] = -1e18
    return int(np.argmax(masked_scores))


class PolicyAdapter(ABC):
    """Base interface for interchangeable autoscaling policies."""

    name: str = "base"
    family: str = "abstract"
    artifact_format: str = "none"

    def predict(
        self,
        obs: Union[np.ndarray, List[float]],
        mask: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> int:
        v_obs, v_mask = validate_observation(obs, mask)
        raw_action = self._predict_impl(v_obs, v_mask)
        return validate_action(raw_action, v_mask)

    @abstractmethod
    def _predict_impl(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        """Run model prediction on validated (3, 8) obs and (6,) mask."""

    def reset_episode(self) -> None:
        """Reset rolling state at episode boundaries."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "artifact_format": self.artifact_format,
        }


