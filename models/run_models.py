import mlflow
import os

COMMIT_SHA = os.getenv('COMMIT_SHA')
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
    print("Running x Model...")
    # TODO: Implement the actual training and logging for the x model
    print("x Model run complete.\n")
    # TODO: Repeat for other models (y, z, etc.), each logging:
    # - metrics.mse
    # - a model artifact with a unique artifact path (e.g. "linear_regression", "random_forest", etc.)

    # At this point, the experiment has all old runs + today's runs.
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


if __name__ == "__main__":
    main()
