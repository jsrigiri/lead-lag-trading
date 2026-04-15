import numpy as np

def backtest(preds, actuals, threshold, cost):
    position = np.where(preds > threshold, 1,
                np.where(preds < -threshold, -1, 0))

    pnl = position[:-1] * actuals[1:]
    trades = np.abs(np.diff(position))
    pnl -= trades * cost

    return pnl.cumsum(), position