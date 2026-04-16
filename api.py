import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from config import CONFIG

app = FastAPI()

model = joblib.load(CONFIG["model_path"])
feature_columns = joblib.load(CONFIG["feature_cols_path"])

class PredictRequest(BaseModel):
    features: dict

@app.get("/")
def root():
    return {"status": "running", "message": "Use POST /predict or open /docs"}

@app.post("/predict")
def predict(request: PredictRequest):
    row = {col: request.features.get(col, 0.0) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)

    if CONFIG["task_type"] == "classification" and hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        return {
            "task_type": CONFIG["task_type"],
            "model_type": CONFIG["model_type"],
            "prediction": pred,
            "probability_up": prob
        }

    pred = model.predict(X)[0]

    return {
        "task_type": CONFIG["task_type"],
        "model_type": CONFIG["model_type"],
        "prediction": float(pred) if CONFIG["task_type"] == "regression" else int(pred)
    }