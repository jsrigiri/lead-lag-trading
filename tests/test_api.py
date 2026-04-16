import os
import joblib
import pandas as pd
from fastapi.testclient import TestClient

from config import CONFIG
from src.model import train_model


def setup_test_artifacts():
    os.makedirs(CONFIG["artifacts_dir"], exist_ok=True)

    X = pd.DataFrame([{
        "X": 100.0,
        "Y": 99.0,
        "X_ret": 0.1,
        "Y_ret": -0.1,
        "X_lag_1": 0.1,
        "X_lag_2": 0.05,
        "X_lag_3": 0.03,
        "X_lag_4": 0.02,
        "X_lag_5": 0.01,
        "Y_lag_1": -0.02,
        "Y_lag_2": 0.01,
        "Y_lag_3": 0.02,
        "Y_lag_4": -0.01,
        "Y_lag_5": 0.03,
    }] * 20)

    if CONFIG["task_type"] == "classification":
        y = pd.Series([0, 1] * 10)
    else:
        y = pd.Series([0.01 * ((-1) ** i) for i in range(20)])

    model, _ = train_model(
        X,
        y,
        model_type=CONFIG["model_type"],
        use_gpu=CONFIG.get("use_gpu", False),
        lightgbm_gpu_backend=CONFIG.get("lightgbm_gpu_backend", "gpu"),
    )

    joblib.dump(model, CONFIG["model_path"])
    joblib.dump(list(X.columns), CONFIG["feature_cols_path"])


def test_api_root_and_predict():
    setup_test_artifacts()

    from api import app

    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert root_resp.json()["status"] == "running"

    payload = {
        "features": {
            "X": 100.0,
            "Y": 99.0,
            "X_ret": 0.1,
            "Y_ret": -0.1,
            "X_lag_1": 0.1,
            "X_lag_2": 0.05,
            "X_lag_3": 0.03,
            "X_lag_4": 0.02,
            "X_lag_5": 0.01,
            "Y_lag_1": -0.02,
            "Y_lag_2": 0.01,
            "Y_lag_3": 0.02,
            "Y_lag_4": -0.01,
            "Y_lag_5": 0.03
        }
    }

    pred_resp = client.post("/predict", json=payload)
    assert pred_resp.status_code == 200

    body = pred_resp.json()
    assert "task_type" in body
    assert "model_type" in body
    assert "prediction" in body

    if CONFIG["task_type"] == "classification" and "probability_up" in body:
        assert 0.0 <= body["probability_up"] <= 1.0