import tempfile
from typing import List

import joblib
import mlflow
import os

import torch
from datasets import BaseDataset, DatasetA, DatasetB, DatasetC, DatasetD
from optimizers import BaseOptimizer, LSTMOptimizer, TCNOptimizer, RandomForestOptimizer, TransformerOptimizer, XGBoostOptimizer
from models import TCNPyFuncWrapper

SEED = 42

COMMIT_SHA = os.getenv('COMMIT_SHA', 'local-dev')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5050')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'local-experiments')
MODEL_TO_DEPLOY = 'lstm'
FORECAST_HORIZON_TO_DEPLOY = 3
DATASET_TO_DEPLOY = 'dataset_a'

DATASETS: List[type[BaseDataset]] = [DatasetB, DatasetD]
OPTIMIZERS: List[type[BaseOptimizer]] = [
    XGBoostOptimizer,
    LSTMOptimizer,
    TCNOptimizer,
    RandomForestOptimizer,
    TransformerOptimizer,
]

FORECAST_HORIZON = [3]

if not COMMIT_SHA:
    raise EnvironmentError("Missing required env var: COMMIT_SHA")


def run_optimizer(optim_cls: type[BaseOptimizer], dataset_cls: type[BaseDataset], dataset: BaseDataset, forecast_horizon: int):
    model_cls = optim_cls.MODEL

    mlflow.start_run(run_name=model_cls.NAME)
    mlflow.set_tags({
        "dataset": dataset_cls.NAME,
        "model": model_cls.NAME,
        "run_type": "parent",
        "sha": COMMIT_SHA,
    })

    optimizer = optim_cls(dataset=dataset, seed=SEED, forecast_horizon=forecast_horizon)
    print(f"Running {model_cls.NAME} optimizer for {dataset_cls.NAME}...")
    study = optimizer.optimize()
    best_params = study.best_params
    print(f"Optimizer completed for dataset {dataset_cls.NAME}.\n")
    print(f"Best parameters found: {best_params}")
    print(f"Best mse achieved: {study.best_value}\n")
    print(f"Training model {model_cls.NAME} with best parameters...")

    # Log the child's best parameters to the parent run
    # To get more information, access the child runs in MLflow UI
    mlflow.log_params(best_params)

    if model_cls.NAME == "tcn":
        best_params["num_inputs"] = len(dataset.FEATURE_COLS)

    model = model_cls(
        dataset=dataset,
        is_optimizing=False,
        forecast_horizon=forecast_horizon,
        **best_params,
    )
    test_mse = model.fit_and_evaluate(run_name=f"final_model")
    print(f"Model {model_cls.NAME} trained. Test mse: {test_mse}\n")

    registered_model_name = f"{model_cls.NAME}-{dataset_cls.NAME}-forecast_horizon-{forecast_horizon}"
    print(f"Logging model {model_cls.NAME} to MLflow with name: {registered_model_name}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_scaler_path = os.path.join(tmp_dir, "target_scaler.pkl")
        feature_scaler_path = os.path.join(tmp_dir, "feature_scaler.pkl")

        joblib.dump(dataset.target_scaler, target_scaler_path)
        joblib.dump(dataset.feature_scaler, feature_scaler_path)

        mlflow.log_artifact(target_scaler_path, artifact_path="scalers")
        mlflow.log_artifact(feature_scaler_path, artifact_path="scalers")

        if model.NAME in ["random_forest", "xgboost"]:
            model_info = mlflow.sklearn.log_model(
                model,
                name=registered_model_name,
                registered_model_name=registered_model_name,
            )
        elif model.NAME == "tcn":
            weights_path = os.path.join(tmp_dir, "tcn_weights.pt")
            torch.save(model.state_dict(), weights_path)

            model_info = mlflow.pyfunc.log_model(
                name=registered_model_name,
                registered_model_name=registered_model_name,
                python_model=TCNPyFuncWrapper(),
                artifacts={
                    "weights": weights_path,
                }
            )
        else:
            model_info = mlflow.pytorch.log_model(
                model,
                name=registered_model_name,
                registered_model_name=registered_model_name,
            )

        mlflow.log_metric(key="val_mse", value=test_mse, model_id=model_info.model_id)

    mlflow.end_run()


def run_dataset(dataset_cls: type[BaseDataset]):
    print(f"Starting runs for dataset: {dataset_cls.NAME}")
    dataset = dataset_cls()
    for forecast_horizon in FORECAST_HORIZON:
        print(f"Running optimizers for forecast horizon: {forecast_horizon}")
        for optim_cls in OPTIMIZERS:
            run_optimizer(optim_cls, dataset_cls, dataset, forecast_horizon)

    print(f"Completed all models for dataset {dataset_cls.NAME}.\n")


def run_training():
    print("Starting training runs...")
    for dataset_cls in DATASETS:
        run_dataset(dataset_cls)
    print("Completed all training runs.")


def update_best_model():
    # Get all models ordered by accuracy
    registered_model_name = f"{MODEL_TO_DEPLOY}-{DATASET_TO_DEPLOY}-forecast_horizon-{FORECAST_HORIZON_TO_DEPLOY}"
    print(f"Promoting best model for registered model name: {registered_model_name}...")

    models = mlflow.search_logged_models(
        filter_string=f"name='{registered_model_name}'",
        order_by=[{
            "field_name": "metrics.val_mse",
            "ascending": False
        }],
        output_format="list",)
    best_model = models[0]

    client = mlflow.MlflowClient()

    model_versions = client.search_model_versions(
        f"name = '{registered_model_name}' and run_id = '{best_model.source_run_id}'"
    )
    best_version = model_versions[0].version

    # Promote the best model by assigning the "best" alias
    client.set_registered_model_alias(registered_model_name, "best", best_version)

    print(
        f"Best model: {registered_model_name},"
        f" run_id: {best_model.source_run_id},"
        f" version: {best_version}"
    )


def main():
    print("Starting MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print("MLflow tracking started.")
    run_training()
    update_best_model()


if __name__ == "__main__":
    main()
