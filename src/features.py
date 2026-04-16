import pandas as pd
import numpy as np

def create_features(df, x_lag, y_lag, task_type="regression"):
    df = df.copy()

    df["X_ret"] = df["X"].diff()
    df["Y_ret"] = df["Y"].diff()

    for i in range(1, x_lag + 1):
        df[f"X_lag_{i}"] = df["X_ret"].shift(i)

    for i in range(1, y_lag + 1):
        df[f"Y_lag_{i}"] = df["Y_ret"].shift(i)

    if task_type == "regression":
        df["target"] = df["Y_ret"].shift(-1)
    elif task_type == "classification":
        df["target"] = (df["Y_ret"].shift(-1) > 0).astype(int)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return df.dropna()