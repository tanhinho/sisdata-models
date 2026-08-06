# Training Module

This module trains fish growth models, tracks experiments in MLflow, registers model versions, and updates the `best` alias when a better model is found.

## What This Folder Contains

- `run_training.py`: main entrypoint for training + registration + best-model update.
- `datasets/`: dataset loading, preprocessing, scaling, and sequence generation.
- `models/`: model architectures.
- `optimizers/`: hyperparameter optimization strategies for each model.
- `Dockerfile`: container image used by `docker compose` for training runs.

## Training Flow

`run_training.py` executes this flow:

1. Set MLflow tracking URI and experiment.
2. For each dataset and optimizer/model combination:
3. Run Optuna hyperparameter optimization.
4. Train/evaluate final model with best params.
5. Log metrics/params and model artifacts to MLflow.
6. Register the model under `fish-growth`.
7. Tag model version metadata (`sha`, `dataset`, `architecture`).
8. After all runs, compare against current alias `best` and update alias only if loss improves.

## Optimization and MLflow Run Hierarchy

- Hyperparameter search is performed with Optuna through the optimizer classes.
- Individual Optuna trials are logged as nested MLflow runs.
- For each model architecture, MLflow creates a parent run named after that architecture.
- Under this parent run, you get nested subruns for Optuna trials and a nested final model run.
- Final comparison metrics and promoted artifacts used for registration are logged to the parent run, which is also the run used to register the model version.
- Saved artifacts include the model `state_dict` (PyTorch weights) and the preprocessing scaler.

## Inputs

- Raw CSV files expected by dataset classes.
- For Dataset A, current code reads from `train-data/dataset_a.csv`.
- Environment variables (see below).

## Outputs

- MLflow runs with params, metrics, and artifacts.
- Artifacts include the saved model `state_dict` and scaler.
- Registered model versions under `fish-growth`.
- Alias updates for `best` when a new model outperforms previous best.

## Environment Variables

- `MLFLOW_TRACKING_URI` (default: `http://localhost:5050`)
- `MLFLOW_EXPERIMENT_NAME` (default: `local-experiment` in script; Compose defaults to `sisdata` unless overridden)
- `COMMIT_SHA` (default: `local-dev`)

Optional image/env variables used in Compose:

- `TRAINING_IMAGE`

## Run Locally With uv

From repository root:

```bash
uv sync --frozen
MLFLOW_TRACKING_URI=http://localhost:5050 MLFLOW_EXPERIMENT_NAME=sisdata COMMIT_SHA=local-dev uv run python training/run_training.py
```

On Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5050"
$env:MLFLOW_EXPERIMENT_NAME="sisdata"
$env:COMMIT_SHA="local-dev"
uv run python training/run_training.py
```

## Run With Docker Compose

1. Ensure MLflow is running:

```bash
docker compose up -d mlflow
```

1. Ensure the expected training data path exists at repository root:

- `train-data/dataset_a.csv`

1. Run training container:

```bash
docker compose run --rm training
```

## Data Mount Note

In `docker-compose.yml`, the training service bind-mounts:

- `./train-data:/app/train-data`

So any local raw files that should be visible to training code must be present under the repository's `train-data/` directory (or you must change the bind mount and dataset file paths consistently).
