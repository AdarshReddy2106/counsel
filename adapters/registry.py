from typing import Dict, Optional, Type

from schema import DEFAULT_CONFIG_PATH, load_config

from .base import PolicyAdapter
from .heuristic_adapter import HeuristicPolicyAdapter
from .hf_adapters import DecisionTransformerPolicyAdapter, PatchTSTPolicyAdapter
from .ppo_adapter import PPOPolicyAdapter
from .sklearn_adapter import RandomForestPolicyAdapter

ADAPTER_CLASSES: Dict[str, Type[PolicyAdapter]] = {
    "ppo": PPOPolicyAdapter,
    "heuristic": HeuristicPolicyAdapter,
    "random_forest": RandomForestPolicyAdapter,
    "decision_transformer": DecisionTransformerPolicyAdapter,
    "patchtst": PatchTSTPolicyAdapter,
}

AVAILABLE_POLICIES = list(ADAPTER_CLASSES.keys())


def get_policy_adapter(
    policy_name: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> PolicyAdapter:
    """Instantiate the requested policy adapter or fall back to `config.yaml`."""
    cfg = load_config(config_path)
    chosen = policy_name or str(cfg["policy"]["active_policy"])
    if chosen not in ADAPTER_CLASSES:
        raise ValueError(
            f"Unknown policy {chosen!r}. Available policies: {AVAILABLE_POLICIES}"
        )

    if chosen == "ppo":
        model_dir = str(cfg["policy"].get("ppo_model_dir", "data/std-f5-c3"))
        return PPOPolicyAdapter(model_dir=model_dir)

    return ADAPTER_CLASSES[chosen]()

