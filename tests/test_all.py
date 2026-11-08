"""
tests/test_all.py — Automated Quality Gates & Contract Test Suite
================================================================
Test suite verifying:
1. Data Contract & Schema Tests:
   - YAML parsing check.
   - ObservationContract boundary verification (NaN/Inf rejection, shape checks).
   - Action allowlisting & mask enforcement.
2. Model Adapter Contract Tests:
   - All 5 policies (`ppo`, `heuristic`, `random_forest`, `decision_transformer`, `patchtst`)
     return compliant integer actions within [0..5] that strictly obey valid_action_mask.
3. Serving Engine & LRU Caching Tests:
   - State hashing, cache hit acceleration, and dynamic zero-downtime hot-swapping.
4. SRE Observability & Drift Tests:
   - Population Stability Index (PSI) mathematical calculation and drift threshold classification.
5. DAG Orchestrator Tests:
   - Kahn's Algorithm topological sort ordering and cycle detection.
6. Model Release Gate:
   - Verifies all benchmarked models satisfy AverageEpRet >= 350.0 and budget_overruns == 0.
"""

import json
import os
import sys
import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from schema import (
    DEFAULT_CONFIG_PATH,
    DataContractError,
    load_config,
    validate_action,
    validate_observation,
)
from adapters import AVAILABLE_POLICIES, get_policy_adapter
from app import PolicyServingEngine, StateLRUCache
from monitor import calculate_psi, classify_drift
from pipeline_dag import topological_sort


# ==================== 1. CONFIGURATION & DATA CONTRACT TESTS ====================


def test_config_norway_problem_protection():
    """Verify that config.yaml quotes string scalars and prevents boolean parsing."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg["project"]["datacenter_country_code"], str)
    assert cfg["project"]["datacenter_country_code"] == "NO"
    assert isinstance(cfg["project"]["emergency_fallback_mode"], str)
    assert cfg["project"]["emergency_fallback_mode"] == "off"
    assert isinstance(cfg["serving"]["cache_enabled"], str)
    assert cfg["serving"]["cache_enabled"] == "yes"


def test_observation_contract_valid_inputs():
    """Test that valid (3, 8) observation matrix and 6-element mask pass contract."""
    obs = np.zeros((3, 8), dtype=np.float32)
    obs[:, 0:3] = np.eye(3)
    obs[:, 5] = 0.50  # 50% utilization
    obs[:, 6] = 20.0  # 20 req/s arrival rate
    mask = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    v_obs, v_mask = validate_observation(obs, mask, ncomp=3, nactions=6)
    assert v_obs.shape == (3, 8)
    assert len(v_mask) == 6
    assert np.all(np.isfinite(v_obs))


def test_observation_contract_rejects_nan_and_inf():
    """Test that NaN and Inf telemetry values trigger immediate DataContractError."""
    obs_nan = np.zeros((3, 8))
    obs_nan[0, 5] = np.nan
    with pytest.raises(DataContractError):
        validate_observation(obs_nan)

    obs_inf = np.zeros((3, 8))
    obs_inf[1, 6] = np.inf
    with pytest.raises(DataContractError):
        validate_observation(obs_inf)


def test_observation_contract_rejects_negative_traffic():
    """Test that negative traffic arrival rates are rejected at the ingestion boundary."""
    obs_neg = np.zeros((3, 8))
    obs_neg[0, 6] = -15.0
    with pytest.raises(DataContractError):
        validate_observation(obs_neg)


def test_action_contract_respects_action_mask():
    """Test that action validation allows unmasked actions and rejects masked ones."""
    # Mask: actions 0, 1, 3 are valid; 2, 4, 5 are masked out (0.0)
    mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])

    assert validate_action(0, mask) == 0
    assert validate_action(1, mask) == 1
    assert validate_action(3, mask) == 3

    # Attempting masked action 2 must raise DataContractError
    with pytest.raises(DataContractError):
        validate_action(2, mask)

    # Attempting out-of-bounds action index 9 must raise DataContractError
    with pytest.raises(DataContractError):
        validate_action(9, mask)


# ==================== 2. POLICY ADAPTER CONTRACT TESTS ====================


@pytest.mark.parametrize("policy_name", AVAILABLE_POLICIES)
def test_all_policy_adapters_return_valid_actions(policy_name):
    """
    Verify that all 5 plug-in models implement the BaseScalingPolicy contract,
    execute inference, and return an unmasked discrete action within [0..5].
    """
    adapter = get_policy_adapter(policy_name)
    adapter.reset_episode()

    # Create realistic observation
    obs = np.zeros((3, 8), dtype=np.float32)
    obs[:, 0:3] = np.eye(3)
    obs[:, 5] = [0.45, 0.70, 0.90]  # DB is high utilization
    obs[:, 6] = 35.0  # High arrival rate

    # Action mask where only flavors 0, 2, 4 are permitted
    mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    action = adapter.predict(obs, mask)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action <= 5
    assert mask[action] == 1.0, f"Policy '{policy_name}' selected masked action {action}!"


# ==================== 3. SERVING ENGINE & CACHING TESTS ====================


def test_lru_cache_hit_and_hot_swapping():
    """Test state LRU caching acceleration and zero-downtime policy hot-swapping."""
    engine = PolicyServingEngine()
    obs = np.ones((3, 8), dtype=np.float32) * 0.5
    mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # First predict call: Cache Miss
    res1 = engine.predict(obs, mask)
    assert res1["cache_hit"] is False
    assert 0 <= res1["action"] <= 5

    # Second predict call with identical state: Cache Hit
    res2 = engine.predict(obs, mask)
    assert res2["cache_hit"] is True
    assert res2["action"] == res1["action"]

    # Hot-swap to heuristic policy
    swap_res = engine.swap_policy("heuristic")
    assert swap_res["status"] == "SUCCESS"
    assert engine.active_policy_name == "heuristic"

    # Query with heuristic policy
    res3 = engine.predict(obs, mask)
    assert res3["policy"] == "heuristic"


# ==================== 4. SRE OBSERVABILITY & DRIFT TESTS ====================


def test_psi_calculation_identical_and_shifted():
    """Verify that calculate_psi gives ~0.0 for identical distributions and >0.25 for shifted."""
    dist_a = np.random.normal(50, 5, 2000)
    # Identical distribution -> PSI ~ 0.0 (< 0.01)
    psi_zero, _ = calculate_psi(dist_a, dist_a)
    assert psi_zero < 0.02
    assert classify_drift(psi_zero) == "NOMINAL (No Drift)"

    # Severely shifted distribution -> PSI >= 0.25
    dist_shifted = np.random.normal(90, 12, 2000)
    psi_high, _ = calculate_psi(dist_a, dist_shifted)
    assert psi_high >= 0.25
    assert classify_drift(psi_high) == "CRITICAL (Significant Drift - Action Required)"


# ==================== 5. DAG ORCHESTRATOR TESTS ====================


def test_kahn_topological_sort_ordering_and_cycle_detection():
    """Verify Kahn's algorithm resolves valid dependencies and catches circular cycles."""
    nodes = ["taskA", "taskB", "taskC", "taskD"]
    # Dependencies: A -> B -> D, A -> C -> D
    edges = [("taskA", "taskB"), ("taskA", "taskC"), ("taskB", "taskD"), ("taskC", "taskD")]

    order = topological_sort(nodes, edges)
    assert order.index("taskA") == 0
    assert order.index("taskD") == 3
    assert set(order) == set(nodes)

    # Circular cycle: A -> B -> C -> A
    cyclic_edges = [("taskA", "taskB"), ("taskB", "taskC"), ("taskC", "taskA")]
    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        topological_sort(["taskA", "taskB", "taskC"], cyclic_edges)


# ==================== 6. MODEL RELEASE GATE ====================


def test_model_release_gate_verification():
    """
    Verify that the multi-model benchmark results in logs/swap_comparison.json
    meet the Model Release Gate criteria:
      1. AverageEpRet >= 350.0
      2. budget_overruns == 0
    """
    swap_json = os.path.join(ROOT_DIR, "logs", "swap_comparison.json")
    assert os.path.exists(swap_json), f"Benchmark comparison missing at {swap_json}"

    with open(swap_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    assert len(results) >= 5, f"Expected 5 evaluated models, found {len(results)}"

    for r in results:
        policy = r["policy"]
        avg_ret = r["average_ep_ret"]
        overruns = r["budget_overruns"]
        gate_passed = r["release_gate_passed"]

        assert (
            avg_ret >= 350.0
        ), f"Policy '{policy}' failed return threshold: {avg_ret:.2f} < 350.0"
        assert (
            overruns == 0
        ), f"Policy '{policy}' had budget overruns: {overruns} > 0"
        assert gate_passed is True, f"Policy '{policy}' marked as failed in release gate!"

