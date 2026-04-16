from src.features import create_features


def test_create_features_regression(sample_price_df):
    out = create_features(sample_price_df, x_lag=3, y_lag=3, task_type="regression")

    expected_cols = {
        "X", "Y", "X_ret", "Y_ret",
        "X_lag_1", "X_lag_2", "X_lag_3",
        "Y_lag_1", "Y_lag_2", "Y_lag_3",
        "target",
    }

    assert expected_cols.issubset(set(out.columns))
    assert len(out) > 0
    assert out.isnull().sum().sum() == 0


def test_create_features_classification(sample_price_df):
    out = create_features(sample_price_df, x_lag=2, y_lag=2, task_type="classification")

    assert "target" in out.columns
    assert set(out["target"].unique()).issubset({0, 1})
    assert out.isnull().sum().sum() == 0


def test_create_features_invalid_task(sample_price_df):
    try:
        create_features(sample_price_df, x_lag=2, y_lag=2, task_type="invalid")
        assert False, "Expected ValueError for invalid task_type"
    except ValueError as e:
        assert "Unsupported task_type" in str(e)