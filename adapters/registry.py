import os
import sys
from typing import Dict, Optional, Type

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from schema import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

try:
    from .base import PolicyAdapter  # noqa: E402
    from .heuristic_adapter import HeuristicPolicyAdapter  # noqa: E402
    from .hf_adapters import DecisionTransformerPolicyAdapter, PatchTSTPolicyAdapter  # noqa: E402
    from .ppo_adapter import PPOPolicyAdapter  # noqa: E402
    from .sklearn_adapter import RandomForestPolicyAdapter  # noqa: E402
except ImportError:
    from adapters.base import PolicyAdapter  # noqa: E402
    from adapters.heuristic_adapter import HeuristicPolicyAdapter  # noqa: E402
    from adapters.hf_adapters import DecisionTransformerPolicyAdapter, PatchTSTPolicyAdapter  # noqa: E402
    from adapters.ppo_adapter import PPOPolicyAdapter  # noqa: E402
    from adapters.sklearn_adapter import RandomForestPolicyAdapter  # noqa: E402

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


if __name__ == "__main__":
    print("=" * 60)
    print("COUNSEL Model Registry — Available Plug-in Policies")
    print("=" * 60)
    for name in AVAILABLE_POLICIES:
        ad = get_policy_adapter(name)
        meta = ad.get_metadata()
        print(f"• {name:<22} | Family: {meta.get('family', 'N/A')}")
        print(f"  Artifact Format: {meta.get('artifact_format', 'N/A')}")
        print(f"  Parameters:      {meta.get('num_parameters', 'N/A')}")
        print("-" * 60)
    print(f"[SUCCESS] All {len(AVAILABLE_POLICIES)} model adapters registered and instantiable!")
