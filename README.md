# SISdATA Models

This repository contains the machine learning models and serving code used by the SISdATA application. It includes:

- Training code and preprocessing logic.
- A FastAPI service for serving models in production.
- MLflow integration for experiment tracking and model promotion.
- Docker and Docker Compose definitions for local development and CI.

---

## Project layout

- `training/` – model training, preprocessing and packaging.
- `serving/` – FastAPI app that serves trained models.
- `mlflow/` – MLflow tracking server Docker image.
- `model_promotion/` – scripts to promote models between stages using MLflow.
- `tests/` – unit and end-to-end tests for models and serving.
- `docker-compose.yml` – local multi-service setup (MLflow, models, serving).
- `.github/workflows/` – CI/CD workflows building, testing and promoting models.

---

## Local development

First, MLflow needs to be running. You can start it with:

```bash
docker compose up mlflow
```

MLFlow will be available at <http://localhost:5050>.

Next, a model needs to be trained and registered in MLflow. First, set the environment variables for the MLflow tracking server, experiment name, and the training image name:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5050
export MLFLOW_EXPERIMENT_NAME=MyExperiment
export TRAINING_IMAGE=myregistry/sisdata-models/training:latest
```

Then, you can run the training container to train and register the model in MLflow. Make sure you have the data available in the `train-data` folder, as it will be mounted into the container. You can run the training container with:

```bash
docker compose run --rm training
```

Finally, you can start the serving API. You must set the environment variables for the MLflow tracking server, experiment name, and the serving image name:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5050
export MLFLOW_EXPERIMENT_NAME=MyExperiment
export SERVING_IMAGE=myregistry/sisdata-models/serving:latest
```

Then, you can start the serving container with:

```bash
docker compose up serving
```

You can then call the serving API at <http://localhost:8080>.

## CI/CD overview

The repository defines four main GitHub Actions workflows:

- `1_continuous_integration.yml` – runs unit tests and builds Docker images for `models` and `serving`.
- `2_continuous_delivery.yml` – pulls the built images, runs model selection and end-to-end tests, and promotes the selected model in MLflow.
- `3_continuous_staging.yml` – deploys the `serving:staging` image to a staging VM, runs staging E2E tests, and promotes the model alias from `staging` to `production` in MLflow.
- `4_continuous_deployment.yml` – deploys the `serving:production` image to the production VM.

Together, these workflows continuously build, validate, stage, and deploy the AI model that power the SISdATA application.
