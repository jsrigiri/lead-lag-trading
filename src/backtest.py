import numpy as np

def backtest(preds, actuals, threshold, cost, task_type="regression"):
    if task_type == "regression":
        position = np.where(
            preds > threshold, 1,
            np.where(preds < -threshold, -1, 0)
        )
    elif task_type == "classification":
        # preds expected as 0/1 labels or probabilities
        if preds.ndim == 1:
            position = np.where(preds >= 0.5, 1, -1)
        else:
            raise ValueError("Unsupported preds shape for classification")
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    pnl = position[:-1] * actuals[1:]
    trades = np.abs(np.diff(position))
    pnl = pnl - trades * cost

    return pnl.cumsum(), position