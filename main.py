import pandas as pd
from config import CONFIG
from src.features import create_features
from src.model import train_model
from src.backtest import backtest
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/sample.csv")

# Features
df = create_features(df, CONFIG["x_lag"], CONFIG["y_lag"])

# Split
split = int(len(df) * CONFIG["train_ratio"])
train, test = df[:split], df[split:]

X_train = train.drop(columns=["target"])
y_train = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]

# Train
model = train_model(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Backtest
equity, pos = backtest(
    preds,
    y_test.values,
    CONFIG["entry_threshold"],
    CONFIG["transaction_cost"]
)

# Plot
plt.plot(equity)
plt.title("Equity Curve")
plt.show()