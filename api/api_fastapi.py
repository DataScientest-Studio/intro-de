from typing import Optional
import json

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from pydantic import BaseModel, Extra


class WineMeasurementData(BaseModel, extra=Extra.forbid):
    alcohol: float
    flavanoids: float
    proline: float
    color_intensity: float
    hue: float

class WineTypePredictionProba(BaseModel):
    class_0 : float
    class_1 : float
    class_2 : float

class WineTypePredictionResponse(BaseModel):
    predicted_class: int
    proba: WineTypePredictionProba



with open("classifier.joblib", "rb") as f:
    ml_model = joblib.load(f)

with open("features.json", "r") as f:
    features = json.load(f)


api = FastAPI()


@api.get(
    "/status",
    response_model=int,
    description="Returns 1 if API is healthy."
)
async def status():
    return 1

@api.post(
    '/predict',
    response_model=WineTypePredictionResponse,
    description="Returns predicted wine type and probabilities."
)
async def predict(wine_measurement_data : WineMeasurementData):

    data = pd.DataFrame([wine_measurement_data.dict()])
    data = data[features]
    predicted_class = ml_model.predict(data)[0]
    predicted_proba = ml_model.predict_proba(data)[0].tolist()

    return {
        "predicted_class": int(predicted_class),
        "proba": {
            "class_0": predicted_proba[0],
            "class_1": predicted_proba[1],
            "class_2": predicted_proba[2]
        }
    }

