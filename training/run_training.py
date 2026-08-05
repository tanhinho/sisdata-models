from typing import List

import joblib
import mlflow
import tempfile
import torch
import os
from datasets import DatasetA
from datasets import BaseDataset
from optimizers import BaseOptimizer, LSTMOptimizer, TCNOptimizer, RandomForestOptimizer, TransformerOptimizer

SEED = 42

COMMIT_SHA = os.getenv('COMMIT_SHA', 'local-dev')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5050')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'local-experiment')
REGISTERED_MODEL_NAME = "fish-growth"

DATASETS: List[type[BaseDataset]] = [DatasetA]
OPTIMIZERS: List[type[BaseOptimizer]] = [
    LSTMOptimizer,
    TCNOptimizer,
    RandomForestOptimizer,
    TransformerOptimizer,
]

if not COMMIT_SHA:
    raise EnvironmentError("Missing required env var: COMMIT_SHA")


class ModelArtifactWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Artifact paths are resolved automatically by MLflow
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model_weights_path = context.artifacts["model_weights"]


def run_training():
    print("Starting training runs...")
    for dataset_cls in DATASETS:
        print(f"Starting runs for dataset: {dataset_cls.NAME}")
        dataset = dataset_cls()
        for optim_cls in OPTIMIZERS:
            model_cls = optim_cls.MODEL

            mlflow.start_run(run_name=model_cls.NAME)
            mlflow.set_tags({
                "dataset": dataset_cls.NAME,
                "model": model_cls.NAME,
                "run_type": "parent",
                "sha": COMMIT_SHA,
            })

            optimizer = optim_cls(dataset=dataset, seed=SEED)
            print(f"Running {model_cls.NAME} optimizer for {dataset_cls.NAME}...")
            study = optimizer.optimize()
            print(f"Optimizer completed for dataset {dataset_cls.NAME}.\n")
            print(f"Best parameters found: {study.best_params}")
            print(f"Best loss achieved: {study.best_value}\n")
            print(f"Training model {model_cls.NAME} with best parameters...")
            model = model_cls(dataset=dataset, is_optimizing=False, **study.best_params)
            test_loss = model.fit_and_evaluate(run_name=f"final_model")
            print(f"Model {model_cls.NAME} trained. Test loss: {test_loss}\n")

            # Log the child's best parameters and test loss to the parent run
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"loss": test_loss})

            with tempfile.TemporaryDirectory() as tmpdir:

                # Save the model to a temporary directory
                model_path = os.path.join(tmpdir, "model.pt")
                scaler_path = os.path.join(tmpdir, "scaler.pkl")

                # Get the underlying PyTorch model
                torch.save(model.state_dict(), model_path)
                joblib.dump(model.dataset.scaler, scaler_path)

                artifacts = {
                    "model": model_path,
                    "scaler": scaler_path,
                }

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=ModelArtifactWrapper(),
                    artifacts=artifacts
                )

            # Save the parent run ID before leaving the run.
            run_id = mlflow.active_run().info.run_id

            # Register this model as a new version of the same registered model.
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
            mlflow.set_tag("model_version", model_version.version)

            client = mlflow.MlflowClient()
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=model_version.version,
                key="sha",
                value=COMMIT_SHA
            )
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=model_version.version,
                key="dataset",
                value=dataset_cls.NAME
            )
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=model_version.version,
                key="architecture",
                value=model_cls.NAME
            )

            print(
                f"Registered {model_cls.NAME} as "
                f"{REGISTERED_MODEL_NAME} "
                f"version {model_version.version}"
            )

            mlflow.end_run()
        print(f"Completed all models for dataset {dataset_cls.NAME}.\n")


def update_best_model():
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")

    experiment_id = experiment.experiment_id

    # Get all parent model runs created by this commit
    current_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.sha = '{COMMIT_SHA}' "
            "AND tags.run_type = 'parent'"
        ),
        order_by=["metrics.loss ASC"],
    )

    if not current_runs:
        raise RuntimeError(
            f"No parent runs found for commit {COMMIT_SHA}"
        )

    # Best model from the current commit
    current_best = current_runs[0]
    current_best_loss = current_best.data.metrics["loss"]
    current_best_version = current_best.data.tags.get("model_version")

    print(
        f"Best model for {COMMIT_SHA}: "
        f"{current_best.info.run_id} "
        f"(version={current_best_version}, loss={current_best_loss})"
    )

    # Find the model currently marked as best
    try:
        previous_best = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias="best")

        previous_best_run = client.get_run(previous_best.run_id)
        previous_best_loss = previous_best_run.data.metrics["loss"]
    except mlflow.exceptions.RestException:
        previous_best = None

    # No previous best, best from this commit automatically becomes best
    if previous_best is None:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="best",
            version=current_best_version
        )
        print(f"No previous best model. Version {current_best_version} is now the best.")
        return

    print(
        f"Previous best: {previous_best_run.info.run_id} "
        f"(version={previous_best.version}, loss={previous_best_loss})"
    )

    # Only replace the best model if the new one is better
    if current_best_loss < previous_best_loss:
        # Mark new model as best
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="best",
            version=current_best_version
        )

        print(
            f"New best model! "
            f"{current_best_loss} < {previous_best_loss}"
        )

    else:
        print("Current models did not beat the existing best.")


def main():
    print("Starting MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print("MLflow tracking started.")
    run_training()
    update_best_model()


if __name__ == "__main__":
    main()
