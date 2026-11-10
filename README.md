# COUNSEL — MLOps Extension

> **Original work:** Proximal Policy Optimization (Clip) on a [GCN](https://arxiv.org/abs/1609.02907v4)-embedded Actor-Critic Model for Dynamic Resource Configuration Management on the Cloud.
>
> [![DOI](https://zenodo.org/badge/553064046.svg)](https://zenodo.org/badge/latestdoi/553064046)

This repository extends the original COUNSEL system (Hegde et al., CCGrid 2023) with a full MLOps pipeline: plug-in model adapters, a feature store, a serving API, automated CI/CD, and an SRE monitoring layer.

---

## What We Changed

The original system was a proof-of-concept with a single hard-coded GCN-PPO model. The table below summarises every gap we identified and the component we built to close it.

| # | Gap in original | What we added |
|---|-----------------|---------------|
| 1 | Locked to one model — `inference.py` contained a single PyTorch GCN-PPO | **Model Registry** (`adapters/`) — five policies behind one `predict(obs, mask) -> int` interface; switch with one line in `config.yaml` |
| 2 | No input validation — raw arrays passed straight to the model | **Data contracts** (`schema.py`) — Pydantic checks on the 3×8 state and 6-value action mask; rejects NaN, Inf, negative arrival rates, and unsafe actions |
| 3 | No stored data — simulator states vanished after each run | **Feature store** (`feature_store.py`) — 9,000 validated steps in Parquet with SHA-256 lineage, three timestamps per row, and a point-in-time train/val split |
| 4 | No serving layer — no API, cache, or hot-swap | **Serving API** (`app.py`) — FastAPI with an LRU cache (< 0.25 ms hit latency), zero-downtime `/swap/{policy}`, `/health`, and `/metrics` endpoints |
| 5 | No observability — no signals, no drift tracking | **Monitoring** (`monitor.py`) — four golden signals, error-budget burn rate, and PSI drift alerts with a retrain trigger at PSI ≥ 0.25 |

---

## Architecture

Six decoupled stages, each with its own contract:

```
configs/config.yaml  →  schema.py  →  feature_store.py  →  adapters/
       ↓                                                        ↓
  pipeline_dag.py  ←─────────────────  evaluate_swap.py  ←  app.py
       ↓
  monitor.py
```

| Stage | File | What it does |
|-------|------|--------------|
| 1 Config & contracts | `configs/config.yaml`, `schema.py` | Single source of truth (SLO 150 ms, budget 2 500–3 600); quoted scalars fix the YAML 1.1 Norway problem |
| 2 Feature store | `feature_store.py` | 9,000 rows × 55 columns in Parquet; 20 PPO episodes, seed 42; cutoff `2026-11-06T10:00:00Z` |
| 3 Model registry | `adapters/` | GCN-PPO, Heuristic, Random Forest, PatchTST, Decision Transformer — all share one `predict` interface |
| 4 Evaluation & gate | `evaluate_swap.py` | 45,000 steps scored; gate: return ≥ 350 and zero budget overruns |
| 5 Serving API | `app.py` | LRU-cached FastAPI; HTTP 422 on bad input; live hot-swap |
| 6 Monitoring & drift | `monitor.py` | PSI, four golden signals, error-budget burn; writes `charts/psi_drift_report.png` |

---

## Model Registry — Five Policies, One Interface

All offline models train on the same feature-store split.

| Model | Type | Params | Latency |
|-------|------|--------|---------|
| **GCN-PPO** | Online RL baseline | 13,383 | ~3.3 ms |
| **Heuristic** | Rule-based | 0 | ~0.3 ms |
| **Random Forest** | Classical ML (120 trees) | — | ~9.7 ms |
| **PatchTST** | Time-series transformer | 407,372 | ~6.0 ms |
| **Decision Transformer** | Offline-RL transformer | 998,565 | ~7.3 ms |

Switch the active model with one line in `configs/config.yaml` — no simulator code changes required.

---

## Evaluation Results

All five models pass the release gate (return ≥ 350, zero budget overruns).

| Model | Return | SLO breach | P99 latency | Resource util. | Gate |
|-------|--------|-----------|-------------|----------------|------|
| GCN-PPO | **555.35** | 24.27% | 3.34 ms | 58.6% | PASS |
| Random Forest | 513.65 | 2.29% | 9.69 ms | 40.1% | PASS |
| PatchTST | 477.34 | 25.36% | 5.96 ms | 53.1% | PASS |
| Heuristic | 445.80 | 2.04% | 0.34 ms | 34.9% | PASS |
| Decision Transformer | 444.57 | 8.11% | 7.28 ms | 38.7% | PASS |

**Key takeaway:** PPO has the best return (555.4) but breaches the SLO 24% of the time. Random Forest reaches 92.5% of PPO's return with only 2.3% SLO breaches, and trains in seconds from Parquet vs. 4.96 M simulator steps for PPO.

---

## Serving API (`app.py`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | Validate 3×8 state and return an action |
| `POST` | `/swap/{policy}` | Switch the live model — no restart needed |
| `GET` | `/health` | Active model, request count, mean latency |
| `GET` | `/metrics` | Cache hit ratio, throughput |

- **Cache-hit latency:** < 0.25 ms (LRU, 512-state capacity)
- **Bad input:** HTTP 422 with a `DataContractError` message

---

## CI/CD (`.github/workflows/ci.yml`)

Three gates run on every push:

1. **Gate 1 — Lint** (`flake8`): PEP 8, undefined names, bad imports.
2. **Gate 2 — Tests** (`pytest tests/test_all.py`): 14 tests covering Norway-quoting safety, NaN/Inf rejection, action-mask allowlist, all 5 adapters, LRU cache, live swap, PSI math, and DAG sort.
3. **Gate 3 — Release gate**: automated evaluator checks return ≥ 350 and zero budget overruns across all models.

**Result:** 14 / 14 tests passed in 33.8 s.

---

## Monitoring (`monitor.py`)

Four golden signals tracked over live telemetry:

- **Latency** — P50 and P99 decision time per policy
- **Traffic** — steps and request arrival rate
- **Errors** — SLO breaches against a 1% error budget with burn rate
- **Saturation** — CPU and memory per service; 0 budget overruns in 45,000 steps

**Drift detection (PSI):**

| PSI range | Status |
|-----------|--------|
| < 0.10 | Normal |
| 0.10 – 0.25 | Warning |
| ≥ 0.25 | **Critical — retrain trigger** |

A simulated 2.8× traffic surge fires the critical alert.

---

## Orchestration Pipeline (`pipeline_dag.py`)

Kahn's topological sort runs six tasks in dependency order (~10 s end-to-end):

| Step | Task | Time | Status |
|------|------|------|--------|
| 1 | `validate_config` | 0.68 s | PASS |
| 2 | `extract_features` | 6.75 s | PASS |
| 3 | `score_policies` | 0.001 s | PASS |
| 4 | `monitor_drift` | 2.59 s | PASS |
| 5 | `model_card` | 0.00 s | PASS |
| 6 | `notify_summary` | 0.01 s | CERTIFIED |

Retry with exponential backoff (`delay = base × factor^attempt`) handles transient simulator socket disconnects. Each run is logged to `logs/pipeline_execution.json`.

---

## Usage & Commands

### Prerequisites

```bash
pip install -r requirements.txt
```

---

### Run Individual Model Adapters

Each adapter can be run standalone to verify it loads and predicts correctly:

```bash
# Hugging Face models (PatchTST, Decision Transformer)
python adapters/hf_adapters.py

# Scikit-learn model (Random Forest)
python adapters/sklearn_adapter.py

# Rule-based policy
python adapters/heuristic_adapter.py

# GCN-PPO (online RL baseline)
python adapters/ppo_adapter.py
```

---

### Inspect the Model Registry

Prints all 5 registered models — file formats, parameter counts, and artifact paths:

```bash
python adapters/registry.py
```

---

### Multi-Model Comparative Benchmark

Runs any combination of models against the synthetic environment under identical traffic conditions:

```bash
# Benchmark all 5 models across 20 episodes
python evaluate_swap.py \
  --policies ppo heuristic random_forest decision_transformer patchtst \
  --episodes 20

# Benchmark a single model (e.g. Decision Transformer)
python evaluate_swap.py --policies decision_transformer --episodes 10
```

---

### End-to-End Orchestrated MLOps Pipeline

Executes all 6 pipeline stages sequentially via Kahn's Topological Sort:  
`Contract Verification → Feature Store → Multi-Model Benchmark → SRE Monitor → Model Card → Summary`

```bash
python pipeline_dag.py
```

Expect 6/6 tasks completed in ~10 s with status `RELEASE_CERTIFIED`.

---

### Automated Quality Gate Test Suite (14 Tests)

Covers schema contracts, adapter interfaces, LRU caching, PSI math, and release gates:

```bash
pytest tests/test_all.py -v
```

Expect **14 / 14 passed** in ~34 s.

---

### Build & Update Feature Store

Generates `data/feature_store/offline_store.parquet` with time-travel snapshots:

```bash
python feature_store.py
```

---

### SRE Golden Signals & Drift Monitor

Calculates PSI, error-budget burn rate, and saves the drift report chart:

```bash
python monitor.py
```

Output: `charts/psi_drift_report.png`

---

### Production Serving API & Dynamic Hot-Swapping

**Step 1 — Start the server:**

```bash
python app.py 127.0.0.1 8080
```

**Step 2 — Send requests from another terminal:**

```bash
# Health check
curl http://127.0.0.1:8080/health

# SRE metrics & cache hit ratio
curl http://127.0.0.1:8080/metrics

# Send a telemetry prediction request
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "obs": [
      [1,0,0,0,0,0.85,50,150],
      [0,1,0,0,0,0.60,50,150],
      [0,0,1,0,0,0.75,50,150]
    ],
    "mask": [1,1,1,1,1,1]
  }'

# Zero-downtime hot-swap — switch to Decision Transformer
curl -X POST http://127.0.0.1:8080/swap/decision_transformer

# Hot-swap to PatchTST
curl -X POST http://127.0.0.1:8080/swap/patchtst

# Hot-swap back to PPO
curl -X POST http://127.0.0.1:8080/swap/ppo
```

---

## Original Reproduction Steps (GCN-PPO baseline)

1. Clone repository from GitHub or Zenodo:
   - https://github.com/TheMonocledHamster/Counsel
   - https://zenodo.org/badge/latestdoi/553064046

2. Ensure Docker Engine and Docker Compose are installed:
   - https://docs.docker.com/engine/install/
   - https://docs.docker.com/compose/install/

3. From the root directory (`counsel/`), run `docker compose up`. The build process takes 5–10 minutes; training may require ~100 hours depending on hardware.
   *(Alternatively: `docker pull themonocledhamster/counsel` and `docker pull themonocledhamster/load-gen`, then `docker compose up`.)*

4. Hyperparameters can be configured in `model/configs/hyperparams.json`. For limited hardware, vary `threadcount` and epoch length.

5. Run `bash copy_data.sh` (or `zsh copy_data.sh`) at any time to fetch the latest training output from the container.

6. Install plotting dependencies and run the visualisation scripts from the `plotters/` folder:
   ```bash
   pip install matplotlib pandas scipy cycler
   ```
   - **Figure 4:** `plot_prov.py`
   - **Figure 5:** `plot_hp.py`
   - **Figure 6:** `plot_infer_time.py`

   Partial graphs can be plotted while training is ongoing; the final plot requires training completion.

---

## Baseline Reproduced

Training job **#93812** on the Madhava GPU cluster — 1,000 epochs, GCN-PPO:

- **Return:** 555.35
- **Budget overruns:** 0

---

## Team Contributions

| Contributor | Area | Key deliverables |
|-------------|------|-----------------|
| **Varun** | Data & contracts | `config.yaml` with quoted scalars, `schema.py` and action masks, Parquet feature store (9,000 steps), SHA-256 lineage, point-in-time split |
| **Adarsh** | Models & adapters | `adapters/base.py` interface, five model wrappers, offline training, `evaluate_swap.py` (45,000 steps), 6-section `MODEL_CARD.md` |
| **Yasin** | Serving & CI/CD | `app.py` (predict, swap, health, metrics), LRU cache and HTTP 422 validation, zero-downtime hot-swap, `ci.yml` with 3 gates and 14 tests |
| **Vivek** | SRE & pipeline | `monitor.py` (four golden signals), error-budget burn, PSI dashboard, `pipeline_dag.py` (Kahn's sort + backoff), telemetry log per run |
