# Serving Module

This module exposes a FastAPI service that loads the currently promoted MLflow model alias and serves prediction requests.

## What It Does

- Starts a FastAPI app on port `8080`.
- Loads model `fish-growth@best` from MLflow at startup.
- Loads preprocessing scaler artifact from the selected model version run.
- Serves health and prediction endpoints.

## API Endpoints

- `GET /`: basic service message.
- `GET /health`: health status.
- `POST /predict`: prediction endpoint.

## Environment Variables

- `MLFLOW_TRACKING_URI` (default: `http://localhost:5050`)
- `MODEL_ALIAS` (default: `best`)
- `SERVING_PORT` (Compose mapping default: `8080`)
- `SERVING_IMAGE` (required when running via Docker Compose)

## Run Locally With uv

From repository root:

```bash
uv sync --frozen
MLFLOW_TRACKING_URI=http://localhost:5050 MODEL_ALIAS=best uv run uvicorn serving.app:app --host 0.0.0.0 --port 8080
```

On Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5050"
$env:MODEL_ALIAS="best"
uv run uvicorn serving.app:app --host 0.0.0.0 --port 8080
```

## Run With Docker Compose

```bash
docker compose up -d mlflow
docker compose up serving
```

The serving container will connect to MLflow using `MLFLOW_TRACKING_URI` and load the model alias configured by `MODEL_ALIAS`.

## Configuration Note

For advanced container settings (networks, volumes, port remapping, environment overrides, and other service-level customization), check `docker-compose.yml` and adjust values there.
