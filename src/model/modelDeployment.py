"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Dec 2024
updated: July 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Define the request schema
class PredictRequest(BaseModel):
    features: list

# Load the trained model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        prediction = model.predict([request.features])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))