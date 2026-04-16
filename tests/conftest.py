import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_price_df():
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        "X": np.linspace(100, 110, n) + np.random.normal(0, 0.2, n),
        "Y": np.linspace(200, 210, n) + np.random.normal(0, 0.3, n),
    })


@pytest.fixture
def regression_xy():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "f1": np.linspace(0, 1, n),
        "f2": np.linspace(1, 2, n),
        "f3": np.random.randn(n),
    })
    y = 0.7 * X["f1"] - 0.3 * X["f2"] + 0.15 * X["f3"]
    return X, y


@pytest.fixture
def classification_xy():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "f1": np.linspace(0, 1, n),
        "f2": np.linspace(1, 2, n),
        "f3": np.random.randn(n),
    })
    score = X["f1"] + 0.5 * X["f3"]
    y = (score > score.median()).astype(int)
    return X, y