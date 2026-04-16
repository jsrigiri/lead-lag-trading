import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("artifacts/model.joblib")
feature_columns = joblib.load("artifacts/feature_columns.joblib")

class PredictRequest(BaseModel):
    features: dict

@app.get("/")
def root():
    return {"status": "running", "message": "Use POST /predict or open /docs"}

@app.post("/predict")
def predict(request: PredictRequest):
    row = {col: request.features.get(col, 0.0) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}