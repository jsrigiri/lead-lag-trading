import os
import joblib
import pandas as pd
from config import CONFIG
from src.features import create_features
from src.model import train_model

os.makedirs(CONFIG["artifacts_dir"], exist_ok=True)

df = pd.read_csv("data/sample.csv")

if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

df = create_features(
    df,
    CONFIG["x_lag"],
    CONFIG["y_lag"],
    CONFIG["task_type"]
)

split = int(len(df) * CONFIG["train_ratio"])
train = df[:split]

X_train = train.drop(columns=["target"])
y_train = train["target"]

model, used_device = train_model(
    X_train,
    y_train,
    model_type=CONFIG["model_type"],
    use_gpu=CONFIG["use_gpu"],
    lightgbm_gpu_backend=CONFIG["lightgbm_gpu_backend"]
)

joblib.dump(model, CONFIG["model_path"])
joblib.dump(list(X_train.columns), CONFIG["feature_cols_path"])

print("Saved model to", CONFIG["model_path"])
print("Saved feature columns to", CONFIG["feature_cols_path"])
print("Task:", CONFIG["task_type"])
print("Model:", CONFIG["model_type"])
print("Device:", used_device)