import hashlib
import json
import os
import sys
from typing import Any, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from feature_store import FeatureStore  # noqa: E402

try:
    from .base import PolicyAdapter, apply_action_mask_to_scores  # noqa: E402
except ImportError:
    from adapters.base import PolicyAdapter, apply_action_mask_to_scores  # noqa: E402



def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class RandomForestPolicyAdapter(PolicyAdapter):
    """Random Forest ensemble trained on `offline_store.parquet`."""

    name = "random_forest"
    family = "Classical ML Ensemble (Scikit-Learn RandomForest)"
    artifact_format = ".joblib + SHA-256 verified .json"

    def __init__(self, force_retrain: bool = False):
        registry_dir = os.path.join(ROOT_DIR, "data", "registry")
        os.makedirs(registry_dir, exist_ok=True)
        self.model_path = os.path.join(registry_dir, "random_forest.joblib")
        self.meta_path = os.path.join(registry_dir, "random_forest_meta.json")

        if force_retrain or not (
            os.path.exists(self.model_path) and os.path.exists(self.meta_path)
        ):
            self._train_from_feature_store()
        else:
            self._load_verified()

    def _train_from_feature_store(self) -> None:
        store = FeatureStore()
        df_full = store.load_dataframe()
        cutoff_idx = int(len(df_full) * 0.80)
        cutoff_ts = df_full["event_time"].iloc[cutoff_idx].isoformat()
        x_train_obs, x_train_mask, y_train_act, df_train = store.get_training_view(
            as_of_time=cutoff_ts
        )
        x_train = np.hstack([x_train_obs, x_train_mask])
        y_train_breach = df_train["slo_breach"].astype(int).to_numpy()

        df_val = df_full.iloc[cutoff_idx + 1:]
        x_val = np.hstack(
            [
                df_val[store.OBS_COLUMNS].to_numpy(dtype=np.float32),
                df_val[store.MASK_COLUMNS].to_numpy(dtype=np.float32),
            ]
        )
        y_val_breach = df_val["slo_breach"].astype(int).to_numpy()

        act_clf = RandomForestClassifier(
            n_estimators=60, max_depth=8, random_state=42, n_jobs=-1
        )
        act_clf.fit(x_train, y_train_act)
        act_clf.n_jobs = 1

        risk_clf = RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        risk_clf.fit(x_train, y_train_breach)
        risk_clf.n_jobs = 1

        breach_val_acc = (
            float(risk_clf.score(x_val, y_val_breach)) if len(y_val_breach) > 0 else 1.0
        )

        bundle = {"act_clf": act_clf, "risk_clf": risk_clf}
        joblib.dump(bundle, self.model_path)
        meta = {
            "model_name": self.name,
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "point_in_time_cutoff": cutoff_ts,
            "val_breach_prediction_accuracy": breach_val_acc,
            "sha256": _sha256(self.model_path),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.act_clf = act_clf
        self.risk_clf = risk_clf
        self.meta = meta

    def _load_verified(self) -> None:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if _sha256(self.model_path) != meta.get("sha256"):
            raise ValueError(
                "SHA-256 mismatch on random_forest.joblib! Refusing to load untrusted artifact."
            )
        bundle = joblib.load(self.model_path)
        self.act_clf = bundle["act_clf"]
        self.risk_clf = bundle["risk_clf"]
        self.act_clf.n_jobs = 1
        self.risk_clf.n_jobs = 1
        self.meta = meta

    def _predict_impl(self, obs_2d: np.ndarray, mask_1d: np.ndarray) -> int:
        feat = np.concatenate([obs_2d.reshape(-1), mask_1d]).reshape(1, -1)
        probs = self.act_clf.predict_proba(feat)[0]
        scores = np.zeros(6, dtype=np.float64)
        for cls_idx, prob in zip(self.act_clf.classes_, probs):
            scores[int(cls_idx)] = float(prob)

        risk_probs = self.risk_clf.predict_proba(feat)[0]
        breach_risk = float(risk_probs[1]) if len(risk_probs) > 1 else 0.0
        is_scale_up = bool(np.all(mask_1d > 0.5))

        if is_scale_up and breach_risk >= 0.65:
            scores[4] += 1.50
            scores[3] += 1.10
        elif is_scale_up and breach_risk >= 0.42:
            scores[3] += 1.25

        return apply_action_mask_to_scores(scores, mask_1d)

    def get_metadata(self) -> Dict[str, Any]:
        out = super().get_metadata()
        out.update(self.meta)
        return out


if __name__ == "__main__":
    print("[TEST] Running RandomForestPolicyAdapter Verification...")
    rf = RandomForestPolicyAdapter()
    sample_obs = np.zeros((3, 8), dtype=np.float32)
    sample_obs[:, 0:3] = np.eye(3)
    sample_obs[:, 5] = [0.45, 0.70, 0.90]
    sample_obs[:, 6] = 35.0
    sample_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    act = rf.predict(sample_obs, sample_mask)
    assert 0 <= act <= 5
    assert sample_mask[act] == 1.0
    print(f"   [OK] RandomForest predicted action: {act} (valid mask: {sample_mask[act] == 1.0})")
    print(f"   Model metadata: trees={rf.meta.get('n_estimators')}, sha256={rf.meta.get('sha256')[:12]}...")
    print("[SUCCESS] RandomForest adapter loaded and executed cleanly!")
