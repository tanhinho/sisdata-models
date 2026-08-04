from pydantic import BaseModel


class PredictionRequest(BaseModel):
    temperature_c: float
    turbidity_ntu: float
    dissolved_oxygen_g_ml: float
    ph: float
    ammonia_g_ml: float
    nitrate_g_ml: float
    population: float


class PredictionResponse(BaseModel):
    prediction: float
