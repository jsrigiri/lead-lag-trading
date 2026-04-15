import pandas as pd

def create_features(df, x_lag, y_lag):
    df["X_ret"] = df["X"].diff()
    df["Y_ret"] = df["Y"].diff()

    for i in range(1, x_lag + 1):
        df[f"X_lag_{i}"] = df["X_ret"].shift(i)

    for i in range(1, y_lag + 1):
        df[f"Y_lag_{i}"] = df["Y_ret"].shift(i)

    df["target"] = df["Y_ret"].shift(-1)
    return df.dropna()