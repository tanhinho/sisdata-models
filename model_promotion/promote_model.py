import os
from mlflow.tracking import MlflowClient

FROM_ALIAS = os.getenv('FROM_ALIAS')
if not FROM_ALIAS:
    raise EnvironmentError("Missing required env var: FROM_ALIAS")

TO_ALIAS = os.getenv('TO_ALIAS')
if not TO_ALIAS:
    raise EnvironmentError("Missing required env var: TO_ALIAS")

MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
if not MLFLOW_MODEL_NAME:
    raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5050')


def promote_model():
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get model versions under the staging alias
    versions = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, FROM_ALIAS)
    if not versions:
        raise RuntimeError(f"No model tagged '{FROM_ALIAS}' found for {MLFLOW_MODEL_NAME}")

    version = versions.version
    print(f"Found {MLFLOW_MODEL_NAME} version {version} under alias '{FROM_ALIAS}'")

    # Promote
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME, alias=TO_ALIAS, version=version
    )

    print(f"✅ Promoted {MLFLOW_MODEL_NAME} version {version} to '{TO_ALIAS}'")


if __name__ == "__main__":
    promote_model()
