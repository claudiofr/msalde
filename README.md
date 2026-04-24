# PLM Framework — A flexible learning framework for protein sequence optimization using protein language models (PLMs).

**PLM Framework** is a Python framework for simulating active learning campaigns in directed protein evolution. It iteratively selects protein variants to "evaluate" from a deep mutational scanning (DMS) dataset, trains a surrogate model on the revealed fitness scores, and uses an acquisition strategy to prioritise the next batch of variants. This loop continues for a configurable number of rounds, enabling systematic benchmarking of different model and strategy combinations across many protein datasets. It also supports n fold cross validation and predictions based on log likehood ratios derived from plm scores. It is an extensible framework that allows you to plug in plm learners and acquisition strategies.

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Configuration](#configuration)
  - [Main config (msaldem.yaml)](#main-config-msaldeyaml)
  - [Simulation config (simulations.yaml)](#simulation-config-simulationsyaml)
- [API reference](#api-reference)
  - [ALDEContainer](#aldecontainer)
  - [DESimulator.run_simulations](#desimulatorrun_simulations)
- [Components](#components)
  - [Learners](#learners)
  - [Embedders](#embedders)
  - [Acquisition strategies](#acquisition-strategies)
  - [Log-likelihood computers](#log-likelihood-computers)
- [Quickstart example](#quickstart-example)
- [Running the benchmark script](#running-the-benchmark-script)
- [Database output](#database-output)

---

## Overview

The active learning loop implemented by PLM Framework works as follows:

1. **Load data** — a DMS dataset is read from a CSV file. Each row is a protein variant with an experimentally measured fitness score.
2. **Embed sequences** — optionally, each variant is encoded into a fixed-length feature vector using a protein language model (ESM2) or pre-computed embeddings loaded from a file.
3. **Round 1 (cold start)** — a random subset of variants is selected; their assay scores are revealed from the dataset.
4. **Rounds 2+ (active learning)** —
   - A surrogate regression model is trained on all variants selected so far.
   - The model predicts scores for all remaining variants.
   - An acquisition strategy ranks candidates and selects the next batch.
   - Their assay scores are revealed and added to the training set.
5. **Persist** — all run metadata, round-level metrics (RMSE, R², Spearman ρ), and optionally all predictions are written to a SQLite database for downstream analysis.

Multiple independent simulations can be run per configuration, and multiple learner/strategy combinations (sub-runs) can be swept in a single call.

---

## Installation

PLM Framework is not packaged on PyPI. Clone the repo and install dependencies into a virtual environment:

```bash
git clone <repo-url>
cd plmlearn
python -m venv venv_msalde
source venv_msalde/bin/activate   # Windows: venv_msalde\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `fair-esm`, `scikit-learn`, `scipy`, `SQLAlchemy`, `OmegaConf`, `pandas`, `biopython`.

Python 3.10 or newer is required.

---

## Project structure

```
msalde/
├── config/
│   ├── msaldem.yaml          # Main configuration (DB, datasets, embedder, run configs)
│   └── simulations.yaml     # Sub-run parameter grid (learner & strategy combos)
├── db/
│   └── msalde.db            # SQLite results database (auto-created)
├── msalde/                  # Python package
│   ├── container.py         # ALDEContainer — dependency injection entry point
│   ├── simulator.py         # DESimulator — core simulation engine
│   ├── active_learner.py    # Ridge & Random Forest learners
│   ├── esm_learner.py       # ESM2-based learners
│   ├── esm_embedder.py      # ESM2 sequence embedder
│   ├── file_load_embedder.py# Pre-computed embedding loader
│   ├── acquisition_strategy.py  # Greedy, UCB, EI, Thompson, Variance, Random
│   ├── esm_log_likelihood_computer.py  # ESM2 log-likelihood ratio features
│   ├── repository.py        # SQLAlchemy persistence layer
│   ├── query_repository.py  # Query helpers
│   ├── model.py             # Dataclasses (Variant, AssayResult, ...)
│   └── dbmodel.py           # ORM models
├── scripts/runs/
│   └── run_maves.py         # Benchmark script for MAVES/DMS datasets
├── notebooks/               # Analysis notebooks
└── requirements.txt
```

---

## Configuration

All configuration is YAML-based and loaded via [OmegaConf](https://omegaconf.readthedocs.io/).

### Main config (msaldem.yaml)

```yaml
general:
  store_acquired_variants: true   # persist per-round selected variants to DB
  data_output_dir: "./output/data"

db:
  url: "sqlite:///./db/msalde.db" # SQLAlchemy connection string

sub_runs:
  config_file: "./config/simulations.yaml"

# Embedder used by the embedding_extractor utility (not the simulation embedder)
embedding_extractor:
  type: esm
  model_name: facebook/esm2_t6_8M_UR50D
  parameters:
    cache_dir: "./model_cache"
    device: "cpu"          # or "cuda"
    batch_size: 8
    quantize: false
    compression:
      method: fisher_vector
      num_gaussian_mixture_components: 64
      pca_dim: 64

# Dataset definitions — add one entry per protein dataset
datasets:
  my_protein:
    data_loader_type: file_loader
    input_path: "./data/my_protein_labels.csv"
    wild_type_id: "WT"
    column_names:
      id_col: "variant"
      score_col: "fitness"
    embeddings_file: "./data/my_protein_esm2.csv"  # optional pre-computed embeddings

# Run configurations — each key is a config_id passed to run_simulations()
c3_1:
  default_dataset: my_protein
  simulation_config_id: c3_1        # key into simulations.yaml
  embedder:
    type: file_loader
    model_name: esm2_t6_8M_UR50D   # used to match embeddings_file column naming

# ESM2 log-likelihood ratio example config
c10:
  default_dataset: my_protein
  simulation_config_id: c10
  log_likelihood_ratio_computer:
    type: ESM2LLRComputer
    parameters:
      base_model_name: facebook/esm2_t6_8M_UR50D
      batch_size: 32
```

**Dataset CSV format** (minimum required columns, names configurable):

| variant | fitness |
|---------|---------|
| WT      | 1.0     |
| A42G    | 0.87    |
| L10F    | 1.23    |

**Pre-computed embeddings CSV format**: rows are variants, columns are embedding dimensions. A variant ID column must be present matching `id_col`.

### Simulation config (simulations.yaml)

Defines the parameter grid swept across sub-runs. Each top-level key is a `simulation_config_id` referenced from the main config.

```yaml
c3_1:
  - name: "RF50_sweep"
    learner:
      type: "RandomForestRegression"
      name: "RF50"
      uses_embedder: false
      uses_random_seed: true
      parameters:
        n_estimators: 50
        criterion: friedman_mse
    first_round_acquisition_strategy:
      type: "Random"
      name: "Random"
      parameters: {}
      uses_random_seed: true
      uses_sub_run_context: false
    acquisition_strategies:
      - type: "Greedy"
        name: "Greedy"
        parameters: {}
        uses_random_seed: false
      - type: "UCB"
        name: "UCB"
        parameters:
          exploration_weight: 1.0
        uses_random_seed: false
      - type: "ThompsonSampling"
        name: "TS"
        parameters: {}
        uses_random_seed: true
```

Each entry in `acquisition_strategies` produces a separate sub-run. A configuration with one learner and three strategies produces three sub-runs, all sharing the same learner.

---

## API reference

### ALDEContainer

```python
from msalde.container import ALDEContainer

container = ALDEContainer(config_file="./config/msaldem.yaml")
```

`ALDEContainer` is the main entry point. It reads the YAML config, creates the database schema if needed, and wires together all components.

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `simulator` | `DESimulator` | Core simulation engine |
| `query_repository` | `QueryRepository` | Query helpers for reading results from the DB |
| `dataset_repository` | `DatasetRepository` | Load and cache variant datasets |
| `variant_repository` | `VariantRepository` | Access reference variant data |
| `plotter` | `Plotter` | Visualisation utilities |
| `embedding_extractor` | `EmbeddingExtractor` | Extract and save ESM2 embeddings |

---

### DESimulator.run_simulations

```python
container.simulator.run_simulations(
    config_id,
    name,
    descrip=None,
    num_simulations=1,
    num_rounds=5,
    num_selected_variants_first_round=10,
    num_top_acquistion_score_variants_per_round=10,
    num_top_prediction_score_variants_per_round=10,
    num_predictions_for_top_n_mean=10,
    test_fraction=0.2,
    random_seed=42,
    dataset_name=None,
    save_all_predictions=False,
    save_last_round_predictions=False,
    n_fold_cv=False,
)
```

Runs a complete active learning campaign and persists results to the database.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_id` | `str` | — | Key into `msaldem.yaml` selecting the run configuration |
| `name` | `str` | — | Label stored in the database for this run |
| `descrip` | `str` | `None` | Optional free-text description |
| `num_simulations` | `int` | `1` | Independent simulation replicates per sub-run |
| `num_rounds` | `int` | `5` | Active learning rounds per simulation |
| `num_selected_variants_first_round` | `int` | `10` | Variants selected randomly in round 1 |
| `num_top_acquistion_score_variants_per_round` | `int` | `10` | Variants selected by acquisition score in rounds 2+ |
| `num_top_prediction_score_variants_per_round` | `int` | `10` | Additional variants selected by predicted score in rounds 2+ |
| `num_predictions_for_top_n_mean` | `int` | `10` | Pool size for top-N performance metrics |
| `test_fraction` | `float` | `0.2` | Fraction of data held out as a fixed test set |
| `random_seed` | `int` | `42` | Seed for data splitting and first-round sampling |
| `dataset_name` | `str` | `None` | Dataset key from `msaldem.yaml`; defaults to `default_dataset` in the config |
| `save_all_predictions` | `bool` | `False` | Persist per-variant predictions for every round |
| `save_last_round_predictions` | `bool` | `False` | Persist per-variant predictions for the final round only |
| `n_fold_cv` | `bool` | `False` | Run n-fold cross-validation mode (requires `num_rounds=2`; `num_simulations` sets the number of folds) |

---

## Components

### Learners

| `type` in config | Description |
|-----------------|-------------|
| `RidgeRegression` | Bagging ensemble of Ridge regressors |
| `RandomForestRegression` | Scikit-learn Random Forest regressor |
| `ESM2RandomForestRegression` | Random Forest with ESM2 embeddings computed on-the-fly |
| `ESM2MLPRegression` | Multi-layer perceptron head on ESM2 embeddings |
| `ESM2HingeForestRegression` | Hinge forest on ESM2 embeddings |
| `ESM2LogLikelihood` | Uses ESM2 log-likelihood ratios as the prediction score (no training) |

All learners except `ESM2LogLikelihood` support optional PCA dimensionality reduction via the `input_dim` parameter.

### Embedders

| `type` in config | Description |
|-----------------|-------------|
| `file_loader` | Load pre-computed embeddings from a CSV file (fast; recommended for large sweeps) |
| `esm` | Compute ESM2 embeddings on-the-fly using a Hugging Face model |

**ESM2 compression methods** (set via `embedder.parameters.compression.method`):

| `method` | Description |
|----------|-------------|
| `mean_pooling` | Mean over residue positions |
| `max_pooling` | Max over residue positions |
| `window_pooling` | Segmented pooling with configurable overlap |
| `cnn` | 1-D CNN with multiple kernel sizes |
| `fisher_vector` | Fisher Vector encoding with GMM + PCA |
| `none` | Raw per-residue embeddings |

### Acquisition strategies

| `type` in config | Description |
|-----------------|-------------|
| `Random` | Uniform random sampling |
| `Greedy` | Highest predicted mean score |
| `UCB` | Upper Confidence Bound: mean + `exploration_weight` x std |
| `ThompsonSampling` | Sample a prediction from the ensemble of component models |
| `ExpectedImprovement` | Expected improvement over the current best observed score |
| `Variance` | Highest model uncertainty (std across ensemble) |

### Log-likelihood computers

| `type` in config | Description |
|-----------------|-------------|
| `ESM2LLRComputer` | Computes per-variant log-likelihood ratio (LLR) relative to wild-type using ESM2. The LLR is appended as an additional feature to each variant. |

---

## Quickstart example

```python
from msalde.container import ALDEContainer

# 1. Initialise from config
container = ALDEContainer("./config/msaldem.yaml")
simulator = container.simulator

# 2. Run an active learning campaign.
#    config_id "c3_1" uses a Random Forest with pre-computed ESM2 embeddings.
#    The sub-run grid in simulations.yaml sweeps acquisition strategies.
simulator.run_simulations(
    config_id="c3_1",
    name="RF_AL_VAL",
    dataset_name="SRC",
    num_simulations=5,           # 5 independent replicates
    num_rounds=11,               # 10 active learning rounds
    num_selected_variants_first_round=16,
    num_top_acquistion_score_variants_per_round=50,
    num_top_prediction_score_variants_per_round=0,
    num_predictions_for_top_n_mean=16,
    test_fraction=0.0,
    random_seed=42,
    save_last_round_predictions=True,
    save_all_predictions=True,
    n_fold_cv=False
)
```

### N-fold cross-validation mode

Set `n_fold_cv=True` to run leave-one-fold-out cross-validation instead of iterative active learning. `num_simulations` becomes the number of folds and `num_rounds` must be exactly `2`.

```python
simulator.run_simulations(
    config_id="c3_2",
    name="RF_5_FOLD_CV",
    dataset_name="SRC",
    num_simulations=5,           # 5 folds
    num_rounds=2,               # 2 for n fold cv
    num_selected_variants_first_round=16,
    num_top_acquistion_score_variants_per_round=50,
    num_top_prediction_score_variants_per_round=0,
    num_predictions_for_top_n_mean=16,
    test_fraction=0.0,
    random_seed=42,
    save_last_round_predictions=True,
    save_all_predictions=True,
    n_fold_cv=True
)
```

### LLR mode

```python
simulator.run_simulations(
    config_id="c10",
    name="ESM2_LLR",
    dataset_name="SRC",
    num_simulations=1,          #
    num_rounds=2,               # 2 for llr
    num_selected_variants_first_round=1,
    num_top_acquistion_score_variants_per_round=50,
    num_top_prediction_score_variants_per_round=0,
    num_predictions_for_top_n_mean=16,
    test_fraction=0.0,
    random_seed=42,
    save_last_round_predictions=True,
    save_all_predictions=True,
    n_fold_cv=False
)
```

### Querying results

After a run completes, use `query_repository` to retrieve scores and assay results:

```python
qr = container.query_repository
# Example: fetch last round scores for a run
results = qr.get_last_round_scores_by_config_dataset_run(
        config_id="c3_1", dataset_name="SRC", run_name="RF_AL_VAL")
```

Use `variant_repository` to retrieve labels for proteins where labels are
explicitly stored originating from MAVE papers. We only need this for MC4R

```python
vr = container.var_repository
# Example: Fetch labels for MC4R which is the only protein where we have labels here.
results = vr.get_variant_assay(assay_source="MC4R",
                          assay_type="Gs", 
                          protein_symbol="MC4R")
```

---

## Running the benchmark script

[scripts/runs/run_maves_minerva.py](scripts/runs/run_maves_minerva.py) runs the full MAVES/DMS benchmark across a list of protein datasets:

```bash
cd /path/to/plmlearn
source venv_msalde/bin/activate
python scripts/runs/run_maves_minerva.py
```

The datasets list and config IDs are set directly in the script. Results are written to the SQLite database specified in `msaldem.yaml`.

---

## Database output

Results are stored in `./db/msalde.db` (path configured in `msaldem.yaml`). Key tables:

| Table | Description |
|-------|-------------|
| `alde_run` | One row per `run_simulations()` call; stores hyperparameters and timing |
| `alde_sub_run` | One row per learner/strategy combination within a run |
| `alde_simulation` | One row per simulation replicate within a sub-run |
| `alde_round` | One row per active learning round; stores RMSE, R², Spearman ρ for train/validation/test sets |
| `alde_round_top_variant` | Top predicted variants saved each round |
| `alde_round_acquired_variant` | Variants selected for evaluation each round (if `store_acquired_variants: true`) |
| `alde_last_round_score` | All predictions from the final round (if `save_last_round_predictions=True`) |

Use any SQLite browser or the `query_repository` API to analyse results.
