# COUNSEL Policy Engine — Model Card

## 1. Intended Use

- **Primary Purpose:** Microservice resource autoscaling for a 3-stage cloud service chain (`Firewall -> WebServer -> Database`). Given a `(3, 8)` observation matrix (`24` values) and a `6`-dimensional binary action mask, the active policy selects one of `6` VM instance flavors (`"none"`, `"2"`, `"5"`, `"10"`, `"13"`, `"19"`) to provision or de-provision.
- **Intended Users:** Cloud SREs, MLOps engineers, and cluster controllers operating microservice chains under latency SLO (`150 ms`) and CPU/RAM budget (`[2500 CPU, 3600 GB]`) constraints.
- **Out-of-Scope Uses:** Deployment without `schema.py` input/output validation, or service chains with different topologies (`ncomp != 3`, `nconf != 5`) without retraining.

---

## 2. Model Details

All 5 interchangeable policies implement `PolicyAdapter.predict(obs, mask) -> int (0..5)` and store artifacts under `data/registry/`:

| Policy ID (`config.yaml`) | Model Family | Source Backbone | Parameters | Registry Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **`ppo`** | Online Graph RL (`GCNActorCritic`) | Trained 1,000 epochs (`4.96M` simulator steps) on GPU cluster (`knob=0.05`) | `13,383` | `data/registry/ppo_baseline.safetensors` (`.safetensors` + `.pt`) |
| **`heuristic`** | Reactive Threshold Controller | Utilization threshold rules (`scale_up >= 80%`, `scale_down < 45%`) | `0` | Pure Python rules (`0 MB`) |
| **`random_forest`** | Tree Ensemble | Dual Scikit-Learn `RandomForestClassifier` trained on `offline_store.parquet` (`7,201` rows) | `120 trees` | `data/registry/random_forest.joblib` + SHA-256 `.json` |
| **`patchtst`** | Multivariate Time-Series Transformer | Hugging Face `PatchTSTForClassification` (`ibm-granite/granite-timeseries-patchtst`, `L=16, patch=4, 10-ch`) | `407,372` | `data/registry/patchtst.safetensors` |
| **`decision_transformer`** | Offline-RL Sequence Transformer | Hugging Face `DecisionTransformerModel` (`edbeeching/decision-transformer-gym-hopper-medium`, `K=8`, `RTG=750`) | `998,565` | `data/registry/decision_transformer.safetensors` |

---

## 3. Performance Evaluation (`20` Episodes = `9,000` Steps, `seed = 42`)

All 5 policies were evaluated via `evaluate_swap.py` on the same `20` workload episodes (`9,000` steps per model, `seed=42`, `slo=150 ms`, `knob=0.05`). Full output is stored in `logs/swap_comparison.json`:

| Rank | Policy | Avg Episode Return | Mean Inference Latency | P99 Inference Latency | Mean Utilization | SLO Breach Rate | Budget Overruns | Release Gate (`>= 350`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1st** | **`ppo`** | **555.35** | `1.100 ms` | `3.337 ms` | **`58.56%`** | `24.27%` | **0** | **PASS** |
| **2nd** | **`random_forest`** | `513.65` | `4.614 ms` | `9.687 ms` | `40.10%` | `2.29%` | **0** | **PASS** |
| **3rd** | **`patchtst`** | `477.34` | `3.018 ms` | `5.961 ms` | `53.13%` | `25.36%` | **0** | **PASS** |
| **4th** | **`heuristic`** | `445.80` | **`0.092 ms`** | **`0.336 ms`** | `34.88%` | **`2.04%`** | **0** | **PASS** |
| **5th** | **`decision_transformer`** | `444.57` | `3.169 ms` | `7.275 ms` | `38.74%` | `8.11%` | **0** | **PASS** |

### Observations
1. **Online RL vs. Offline Feature Store Models:**
   - `ppo` reaches the highest return (`555.35`) because it trained directly against the environment over `4.96M` transitions and keeps average utilization higher (`58.56%`).
   - Offline models (`random_forest`, `patchtst`, `decision_transformer`) train in a few seconds from `offline_store.parquet` (`7,201` rows) without running live simulator rollouts during training.
2. **Latency vs. Resource Headroom:**
   - `heuristic` and `random_forest` keep lower SLO breach rates (`2.04%` and `2.29%`) by provisioning extra VM headroom (`34.88%`–`40.10%` utilization), which lowers net reward compared to `ppo`.

---

## 4. Ethical Considerations

- **Resource Quotas:** All 5 policies respect the `[2500 CPU, 3600 GB]` budget (`0` overruns across `45,000` evaluation steps).
- **Compute Footprint:** `PatchTST` (`407K` params) and `DecisionTransformer` (`998K` params) run quickly on CPU without requiring large pretrained text models.

---

## 5. Deployment & Maintenance

- **Policy Switching:** Update `policy.active_policy` in `configs/config.yaml` to switch between registered policies without changing environment code.
- **Point-in-Time Splits:** Offline models train using `FeatureStore.get_training_view(as_of_time=...)` with an 80/20 chronological split (`2026-11-06T10:00:00+00:00`) to avoid data leakage.
- **Drift Monitoring:** Production `arrival_rate` is tracked with Population Stability Index (PSI), using `0.10` for moderate drift warnings and `0.25` for retraining alerts.

---

## 6. Responsible AI

- **Input/Output Validation:** `PolicyAdapter.predict()` validates `(obs, mask)` via `validate_observation` and checks chosen actions against `validate_action` before applying scaling steps.
- **Safe No-Op Fallback:** Action `0` (`"none"`) is always kept valid (`mask[0] == 1.0`) so the controller can hold existing capacity safely.

