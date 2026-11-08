import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "config.yaml"
)


class DataContractError(ValueError):
    """Raised when environment observations or actions fail validation."""


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_cfg = cfg.get("project", {})
    for key in ("datacenter_country_code", "emergency_fallback_mode"):
        val = project_cfg.get(key)
        if isinstance(val, bool):
            raise DataContractError(
                f"Invalid boolean parsed for project.{key}: {val!r}. Expected string."
            )

    serving_cfg = cfg.get("serving", {})
    if isinstance(serving_cfg.get("cache_enabled"), bool):
        raise DataContractError(
            "Invalid boolean parsed for serving.cache_enabled. Expected quoted string."
        )

    return cfg


class ObservationContract(BaseModel):
    obs_flat: List[float] = Field(...)
    mask: List[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    schema_version: str = Field(default="1.0.0")
    act_type: int = Field(default=1)
    act_comp: int = Field(default=0)
    slo_latency_ms: float = Field(default=150.0)


def validate_observation(
    obs: Union[np.ndarray, List],
    mask: Optional[Union[np.ndarray, List]] = None,
    ncomp: int = 3,
    nactions: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    expected_cols = ncomp + 5
    expected_size = ncomp * expected_cols

    arr = np.asarray(obs, dtype=np.float32)
    if arr.size != expected_size:
        raise DataContractError(
            f"Invalid observation size: expected {expected_size}, got {arr.size}."
        )

    if not np.all(np.isfinite(arr)):
        raise DataContractError("Observation contains NaN or Inf values.")

    obs_2d = arr.reshape((ncomp, expected_cols))

    adj = obs_2d[:, :ncomp]
    if np.any(adj < -1e-5) or np.any(adj > 1.0 + 1e-4):
        raise DataContractError(
            f"Adjacency matrix out of [0, 1] bounds: min={adj.min():.4f}, max={adj.max():.4f}"
        )

    z_res = obs_2d[:, ncomp : ncomp + 2]
    if np.any(np.abs(z_res) > 10.0):
        raise DataContractError(
            f"Normalized CPU/Mem ratio out of [-10, 10] bounds: {np.max(np.abs(z_res)):.4f}"
        )

    util = obs_2d[:, ncomp + 2]
    if np.any(util < -1e-5) or np.any(util > 2.0 + 1e-4):
        raise DataContractError(
            f"Component utilization out of [0.0, 2.0] bounds: {util.tolist()}"
        )

    traffic = obs_2d[:, ncomp + 3]
    if np.any(traffic < 0.0):
        raise DataContractError(
            f"Arrival rate cannot be negative: {traffic.tolist()}"
        )

    if mask is None:
        mask_1d = np.ones(nactions, dtype=np.float32)
    else:
        mask_1d = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask_1d.shape != (nactions,):
            raise DataContractError(
                f"Invalid action mask shape: expected ({nactions},), got {mask_1d.shape}."
            )
        if not np.all(np.isfinite(mask_1d)):
            raise DataContractError("Action mask contains NaN or Inf values.")
        if not np.all(np.isin(mask_1d, [0.0, 1.0])):
            raise DataContractError(
                f"Action mask must contain only 0.0 or 1.0, got {mask_1d.tolist()}."
            )
        if mask_1d[0] != 1.0:
            raise DataContractError("Action mask[0] ('none') must be 1.0.")

    _ = ObservationContract(obs_flat=obs_2d.reshape(-1).tolist(), mask=mask_1d.tolist())

    return obs_2d, mask_1d


def validate_action(
    action: int, mask: Optional[Union[np.ndarray, List]] = None, nactions: int = 6
) -> int:
    try:
        act_int = int(action)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Action must be int, got {action!r}") from exc

    if act_int < 0 or act_int >= nactions:
        raise DataContractError(
            f"Action {act_int} out of valid range [0, {nactions - 1}]."
        )

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask_arr[act_int] == 0.0:
            raise DataContractError(
                f"Action {act_int} is disallowed by mask {mask_arr.tolist()}."
            )

    return act_int


if __name__ == "__main__":
    cfg = load_config()
    print(f"Loaded config: active_policy={cfg['policy']['active_policy']}")

    sample_obs = np.array(
        [
            [0.5, 0.408, 0.0, -1.16, -1.06, 0.62, 42.5, 1.0],
            [0.0, 0.5, 0.408, 1.27, 1.33, 0.54, 42.5, 0.0],
            [0.0, 0.0, 0.5, -0.11, -0.26, 0.48, 42.5, 0.0],
        ],
        dtype=np.float32,
    )
    sample_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    v_obs, v_mask = validate_observation(sample_obs, sample_mask)
    assert v_obs.shape == (3, 8) and v_mask.shape == (6,)
    assert validate_action(2, v_mask) == 2

    bad_nan = sample_obs.copy()
    bad_nan[1, 5] = np.nan
    try:
        validate_observation(bad_nan, sample_mask)
        raise AssertionError("Expected failure on NaN")
    except DataContractError:
        pass

    bad_traffic = sample_obs.copy()
    bad_traffic[:, 6] = -10.0
    try:
        validate_observation(bad_traffic, sample_mask)
        raise AssertionError("Expected failure on negative traffic")
    except DataContractError:
        pass

    try:
        validate_action(1, sample_mask)
        raise AssertionError("Expected failure on masked action")
    except DataContractError:
        pass

    print("All schema validation checks passed.")
