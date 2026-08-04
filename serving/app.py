import os
from contextlib import asynccontextmanager

import torch
import joblib
import mlflow
from fastapi import FastAPI

from .app import PredictionRequest, PredictionResponse

REGISTERED_MODEL_NAME = "fish-growth"
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "best")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5050",
)

model = None
scaler = None
seq_length = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global scaler
    global seq_length
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Loading model '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}'...")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    print(f"Model loaded successfully: {model_uri}")

    print(f"Getting model version for '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}'...")
    model_version = mlflow.MlflowClient().get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
    )
    print(f"Model version for '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}': {model_version.version}")

    print(f"Loading scaler for model '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}'...")
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id,
        artifact_path="preprocessing/scaler.pkl",
    )
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded successfully for model '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}'.")

    print(f"Getting sequence length for '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}'...")
    run = mlflow.get_run(model_version.run_id)
    seq_length = int(run.data.params["seq_length"])
    print(f"Sequence length for '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}': {seq_length}")

    yield

    model = None
    scaler = None
    seq_length = None
    print(f"Model '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}' unloaded successfully.")


app = FastAPI(
    title="SISdATA Fish Growth Prediction API",
    lifespan=lifespan,
)


@app.get("/", tags=["health"])
async def root():
    return {"message": "SISdATA fish growth prediction API is running"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(payload: PredictionRequest):
    # Create the input in the same order/features used during training
    input_data = torch.tensor(
        [[
            payload.temperature_c,
            payload.turbidity_ntu,
            payload.dissolved_oxygen_g_ml,
            payload.ph,
            payload.ammonia_g_ml,
            payload.nitrate_g_ml,
            payload.population,
        ]],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        prediction = model(input_data)

    # If your model returns (output, h_n, c_n)
    prediction_value = prediction[0].item()

    return PredictionResponse(
        prediction=round(prediction_value, 4)
    )
