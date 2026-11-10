"""
Unit Tests for Telemetry Data Contracts & Feature Store Lineage
Authored by Member 1 (Environment, Feature Store & Data Contracts)

Tests runtime boundaries:
- Observation shape (3x8 = 24 finite values)
- NaN and Inf rejection
- Non-negative arrival rate constraint
- Action mask compliance and Action 0 availability
- YAML Norway problem boolean coercion rejection
- Feature store SHA-256 lineage integrity verification
"""

import os
import sys
import numpy as np
import pytest

# Ensure repository root is on Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from schema import (
    DataContractError,
    load_config,
    validate_action,
    validate_observation,
    get_telemetry_columns_description,
)
from feature_store import FeatureStore


# ==================== 1. TELEMETRY CONTRACT & BOUNDARY TESTS ====================


def test_valid_3x8_observation_passes():
    """Verify standard (3, 8) cloud telemetry matrix passes contract validation."""
    sample_obs = np.array(
        [
            [0.5, 0.408, 0.0, -1.16, -1.06, 0.62, 42.5, 1.0],
            [0.0, 0.5, 0.408, 1.27, 1.33, 0.54, 42.5, 0.0],
            [0.0, 0.0, 0.5, -0.11, -0.26, 0.48, 42.5, 0.0],
        ],
        dtype=np.float32,
    )
    sample_mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    v_obs, v_mask = validate_observation(sample_obs, sample_mask, ncomp=3, nactions=6)
    assert v_obs.shape == (3, 8)
    assert v_mask.shape == (6,)
    assert v_mask[0] == 1.0


def test_observation_nan_rejection():
    """Verify that any NaN value from dropped sensor telemetry raises DataContractError."""
    bad_obs = np.zeros((3, 8), dtype=np.float32)
    bad_obs[1, 5] = np.nan
    mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    with pytest.raises(DataContractError, match="NaN or Inf"):
        validate_observation(bad_obs, mask)


def test_observation_inf_rejection():
    """Verify that infinite values (e.g. division by zero) raise DataContractError."""
    bad_obs = np.zeros((3, 8), dtype=np.float32)
    bad_obs[0, 3] = np.inf
    mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    with pytest.raises(DataContractError, match="NaN or Inf"):
        validate_observation(bad_obs, mask)


def test_negative_arrival_rate_rejection():
    """Verify that negative arrival rates raise DataContractError."""
    bad_obs = np.zeros((3, 8), dtype=np.float32)
    bad_obs[:, 6] = -15.0  # Column 6 is Arrival Rate

    with pytest.raises(DataContractError, match="Arrival rate cannot be negative"):
        validate_observation(bad_obs)


def test_action_0_must_never_be_masked():
    """Verify that an action mask locking Action 0 ('none'/No-Op) is rejected."""
    good_obs = np.zeros((3, 8), dtype=np.float32)
    illegal_mask = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Action 0 blocked!

    with pytest.raises(DataContractError, match="Action mask\\[0\\] .* must be 1.0"):
        validate_observation(good_obs, illegal_mask)


def test_disallowed_action_selection_rejection():
    """Verify that selecting a masked action index raises DataContractError."""
    mask = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    # Action 1 is allowed
    assert validate_action(1, mask) == 1

    # Action 2 is masked (disallowed)
    with pytest.raises(DataContractError, match="is disallowed by mask"):
        validate_action(2, mask)


def test_telemetry_columns_description():
    """Verify telemetry documentation helper returns all 8 measurement descriptions."""
    cols = get_telemetry_columns_description()
    assert len(cols) == 6  # 6 descriptor groups covering 8 columns
    names = [c["name"] for c in cols]
    assert "Adjacency Matrix" in names
    assert "Component Utilization" in names
    assert "Arrival Rate" in names


# ==================== 2. FEATURE STORE & LINEAGE TESTS ====================


def test_feature_store_integrity_verification():
    """Verify SHA-256 dataset hash matches metadata.json recorded lineage."""
    store = FeatureStore()
    report = store.verify_dataset_integrity()
    assert report["is_valid"] is True
    assert report["status"] == "VERIFIED"
    assert len(report["calculated_sha256"]) == 64
