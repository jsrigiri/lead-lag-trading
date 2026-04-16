import os
import joblib
import pandas as pd
from config import CONFIG
from src.features import create_features
from src.model import train_model

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/sample.csv")
df = create_features(df, CONFIG["x_lag"], CONFIG["y_lag"])

split = int(len(df) * CONFIG["train_ratio"])
train = df[:split]

X_train = train.drop(columns=["target"])
y_train = train["target"]

model = train_model(X_train, y_train)

joblib.dump(model, "artifacts/model.joblib")
joblib.dump(list(X_train.columns), "artifacts/feature_columns.joblib")

print("Saved model to artifacts/model.joblib")