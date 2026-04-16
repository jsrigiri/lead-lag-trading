import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import CONFIG
from src.features import create_features
from src.model import train_model
from src.backtest import backtest

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
train, test = df[:split], df[split:]

X_train = train.drop(columns=["target"])
y_train = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]

model, used_device = train_model(
    X_train,
    y_train,
    model_type=CONFIG["model_type"],
    use_gpu=CONFIG["use_gpu"],
    lightgbm_gpu_backend=CONFIG["lightgbm_gpu_backend"]
)

if CONFIG["task_type"] == "regression":
    preds = model.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds))
    }

    actual_returns = test["Y_ret"].values

elif CONFIG["task_type"] == "classification":
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.5).astype(int)
    else:
        preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }

    # use actual next return for pnl
    actual_returns = test["Y_ret"].values

else:
    raise ValueError("Unsupported task_type")

equity, pos = backtest(
    preds,
    actual_returns,
    CONFIG["entry_threshold"],
    CONFIG["transaction_cost"],
    CONFIG["task_type"]
)

print("Task:", CONFIG["task_type"])
print("Model:", CONFIG["model_type"])
print("Device:", used_device)
print("Metrics:", metrics)

plt.plot(equity)
plt.title(f"Equity Curve - {CONFIG['model_type']} ({CONFIG['task_type']})")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL")
plt.tight_layout()
plt.show()