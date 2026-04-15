import numpy as np
import pandas as pd

np.random.seed(42)

n = 1500  # number of data points

# Simulate X as random walk
x = np.cumsum(np.random.normal(0, 0.5, n))

# Simulate Y with lead-lag relationship to X
y = np.zeros(n)
for i in range(1, n):
    y[i] = (
        0.3 * x[i-1] +     # lagged influence (lead-lag signal)
        0.1 * y[i-1] +     # autoregressive component
        np.random.normal(0, 0.5)
    )

df = pd.DataFrame({
    "X": x,
    "Y": y
})

df.to_csv("data/sample.csv", index=False)

print("Generated data/sample.csv with", n, "rows")