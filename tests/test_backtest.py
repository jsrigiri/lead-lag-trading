import numpy as np
from src.backtest import backtest


def test_backtest_regression():
    preds = np.array([0.10, 0.20, -0.30, 0.00, 0.50])
    actuals = np.array([0.05, 0.03, -0.02, 0.01, 0.04])

    equity, pos = backtest(
        preds=preds,
        actuals=actuals,
        threshold=0.05,
        cost=0.001,
        task_type="regression",
    )

    assert len(pos) == len(preds)
    assert len(equity) == len(preds) - 1


def test_backtest_classification():
    preds = np.array([1, 0, 1, 1, 0])
    actuals = np.array([0.05, -0.03, 0.02, 0.01, -0.04])

    equity, pos = backtest(
        preds=preds,
        actuals=actuals,
        threshold=0.5,
        cost=0.001,
        task_type="classification",
    )

    assert len(pos) == len(preds)
    assert len(equity) == len(preds) - 1
    assert set(np.unique(pos)).issubset({-1, 1})


def test_backtest_invalid_task():
    preds = np.array([0.1, 0.2, 0.3])
    actuals = np.array([0.01, 0.02, 0.03])

    try:
        backtest(preds, actuals, threshold=0.1, cost=0.001, task_type="invalid")
        assert False, "Expected ValueError for invalid task_type"
    except ValueError as e:
        assert "Unsupported task_type" in str(e)