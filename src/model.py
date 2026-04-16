from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

def train_model(X, y, model_type="ridge"):
    if model_type == "ridge":
        model = Ridge(alpha=1.0)

    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)

    elif model_type == "xgboost_reg":
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    elif model_type == "lightgbm_reg":
        model = LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    elif model_type == "xgboost_clf":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss"
        )

    elif model_type == "lightgbm_clf":
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X, y)
    return model