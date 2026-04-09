from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="SISdATAModels API")


class PredictionRequest(BaseModel):
    # TODO: adjust fields to match real model inputs
    feature1: float
    feature2: float


class PredictionResponse(BaseModel):
    # TODO: adjust fields to match real model outputs
    prediction: float


@app.get("/", tags=["health"])
async def root():
    return {"message": "SISdATAModels API is running"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(payload: PredictionRequest):
    # TODO: This is just a placeholder using a simple formula.
    dummy_prediction = payload.feature1 + payload.feature2
    return PredictionResponse(prediction=dummy_prediction)
