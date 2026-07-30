from typing import List

import mlflow
import os
from datasets import DatasetA
from models import BaseModel, LSTMModel
from datasets import BaseDataset

SEED = 42

COMMIT_SHA = os.getenv('COMMIT_SHA')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5050')

DATASETS: List[type[BaseDataset]] = [DatasetA]
MODELS: List[type[BaseModel]] = [LSTMModel]

if not COMMIT_SHA:
    raise EnvironmentError("Missing required env var: COMMIT_SHA")


def get_best_existing_model():
    """ Find the best existing model in MLflow based on MSE metric.

    Raises:
        RuntimeError: If the experiment does not exist, if no runs are found,
        or if a model artifact path cannot be determined for the best run.

    Returns:
        tuple[str, str]: A pair ``(run_id, artifact_name)`` where ``run_id``
        is the MLflow run identifier and ``artifact_name`` is the name of the logged model artifact.
    """
    client = mlflow.tracking.MlflowClient()

    # Get the experiment
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default')
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise RuntimeError(f"No experiment found with name: {experiment_name}")

    # Search for all runs in the experiment, ordered by MSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.mse ASC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No existing runs found. You must train models first.")

    best_run = runs[0]
    best_mse = best_run.data.metrics.get('mse')

    print(f"Best existing model found:")
    print(f"  Run ID: {best_run.info.run_id}")
    print(f"  MSE: {best_mse}")
    print(f"  Tags: {best_run.data.tags}")

    # Check which artifact was logged
    artifacts = client.list_artifacts(best_run.info.run_id)
    artifact_name = None

    for artifact in artifacts:
        # TODO: Add the actual artifact paths you expect for your models here
        if artifact.path in []:
            # The artifact path is also the model's name (e.g. "linear_regression", "random_forest", etc.)
            artifact_name = artifact.path
            break

    if not artifact_name:
        raise RuntimeError("Could not determine model artifact name from best run")

    return best_run.info.run_id, artifact_name


def main():
    print("Starting MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sisdata")
    print("MLflow tracking started.")
    print("Starting training runs...")
    for dataset in DATASETS:
        mlflow.start_run(run_name=f"Dataset: {dataset.__name__}", nested=True)
        print(f"Starting runs for dataset: {dataset.__name__}")
        for model_cls in MODELS:
            mlflow.start_run(run_name=f"Model: {model_cls.__name__}", nested=True)

            optimizer = model_cls.OPTIMIZER(dataset=dataset, seed=SEED)
            print(f"Running optimizer {optimizer.__name__} for dataset {dataset.__name__}...")
            result = optimizer.optimize(dataset=dataset, seed=SEED)
            print(f"Optimizer {optimizer.__name__} completed for dataset {dataset.__name__}.\n")
            print(f"Best parameters found: {result['best_params']}")
            print(f"Best loss achieved: {result['best_loss']}\n")

            print(f"Training model {model_cls.__name__} with best parameters...")
            model = model_cls(dataset=dataset, is_optimizing=False, **result['best_params'])
            test_loss = model.fit_and_evaluate(run_name=f"optimized_{model_cls.__name__}")
            print(f"Model {model_cls.__name__} trained. Test loss: {test_loss}\n")

            mlflow.log_params(result['best_params'])
            mlflow.log_metrics({"test_loss": test_loss})
            mlflow.end_run()
        print(f"Completed all models for dataset {dataset.__name__}.\n")
        mlflow.end_run()


"""     # At this point, the experiment has all old runs + today's runs.
    # Use ALL runs to pick the true best model.
    best_model, artifact_name = get_best_existing_model()

    best_model_uri = f"runs:/{best_model.run_id}/{artifact_name}"
    model = mlflow.register_model(best_model_uri, "best_model")

    try:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(name=model.name, alias=COMMIT_SHA, version=model.version)
        print(f"Set alias '{COMMIT_SHA}' for {model.name} version {model.version}")
    except Exception as e:
        print(f"Could not set model alias: {e}")
        raise e
 """

if __name__ == "__main__":
    main()
