import numpy as np
import pandas as pd

from sigma.forecast import evaluate_forecasts


def test_evaluate_forecasts_returns_metrics_per_model():
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    forecasts = pd.DataFrame(
        {"unique_id": "0101", "ds": dates, "auto_arima": [1.0, 2.0, 4.0]}
    )
    test_df = pd.DataFrame({"unique_id": "0101", "ds": dates, "y": [1.0, 2.0, 3.0]})

    metrics = evaluate_forecasts(forecasts, test_df)

    assert set(metrics["metric"]) == {"mse", "mae", "rmse"}
    assert np.isclose(metrics.loc[metrics["metric"] == "mae", "auto_arima"].iloc[0], 1 / 3)
