CONFIG = {
    "x_lag": 5,
    "y_lag": 5,
    "train_ratio": 0.7,
    "entry_threshold": 0.0005,
    "transaction_cost": 0.0001,

    "task_type": "regression",   # regression or classification
    "model_type": "xgboost_reg", # ridge, xgboost_reg, lightgbm_reg, logistic, xgboost_clf, lightgbm_clf

    "artifacts_dir": "artifacts",
    "model_path": "artifacts/model.joblib",
    "feature_cols_path": "artifacts/feature_columns.joblib"
}