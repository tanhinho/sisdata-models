# SISdATA Models

This repository contains the machine learning models and serving code used by the SISdATA application. It includes:

- Training code and preprocessing logic.
- A FastAPI service for serving models in production.
- MLflow integration for experiment tracking and model promotion.
- Docker and Docker Compose definitions for local development and CI.

---

## Project layout

- `models/` – model training, preprocessing and packaging.
- `serving/` – FastAPI app that serves trained models.
- `mlflow/` – MLflow tracking server Docker image.
- `model_promotion/` – scripts to promote models between stages using MLflow.
- `tests/` – unit and end-to-end tests for models and serving.
- `docker-compose.yml` – local multi-service setup (MLflow, models, serving).
- `.github/workflows/` – CI/CD workflows building, testing and promoting models.

---

## Local development

To bring up MLflow, the serving API and model container together:

```bash
docker compose up --build
```

This will build and start:

- `mlflow` on port `5050`.
- `serving` on port `8080`.
- `models` as a batch container using MLflow for tracking.

You can then call the serving API at <http://127.0.0.1:8080>.

## CI/CD overview

The repository defines four main GitHub Actions workflows:

- `1_continuous_integration.yml` – runs unit tests and builds Docker images for `models` and `serving`.
- `2_continuous_delivery.yml` – pulls the built images, runs model selection and end-to-end tests, and promotes the selected model in MLflow.
- `3_continuous_staging.yml` – deploys the `serving:staging` image to a staging VM, runs staging E2E tests, and promotes the model alias from `staging` to `production` in MLflow.
- `4_continuous_deployment.yml` – deploys the `serving:production` image to the production VM.

Together, these workflows continuously build, validate, stage, and deploy the AI models that power the SISdATA application.
