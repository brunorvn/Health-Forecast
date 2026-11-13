import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTBATS,
    AutoTheta,
    Theta,
)

from sigma.config import MODEL_DIR, MODEL_PARAMS


class ForecastModel:
    """Wrapper class for forecasting models with support for multiple algorithms."""

    def __init__(self, model_name: str | list[dict] = "auto_arima", **model_params):
        """
        Initialize the forecasting model(s).

        Args:
            model_name: Either:
                      - String name of a single model (e.g., 'auto_arima')
                      - List of model configurations (e.g., [{'name': 'auto_arima'}, {'name': 'auto_ets'}])
            **model_params: Additional parameters for the models
        """
        self.model_params = MODEL_PARAMS.copy()
        self.model_params.update(model_params)

        if isinstance(model_name, str):
            self.model_name = model_name.lower()
            self.model_list = [{"name": self.model_name}]
        elif isinstance(model_name, list):
            self.model_list = model_name
            self.model_name = "ensemble" if len(model_name) > 1 else model_name[0]["name"].lower()
        else:
            raise TypeError("model_name must be either a string or a list of model configurations")

        self.models = None
        self.fitted = False
        self.freq = "MS"  # Monthly seasonality as default
        self.horizon = 12  # Default forecast horizon
        self.model_objects = self._init_models()

    def _init_models(self):
        """Initialize the specified models."""
        model_map = {
            "auto_arima": AutoARIMA,
            "auto_ets": AutoETS,
            "theta": Theta,
            "auto_ces": AutoCES,
            "auto_tbats": AutoTBATS,
            "auto_theta": AutoTheta,
        }

        model_objects = []
        for model_cfg in self.model_list:
            model_name = model_cfg["name"].lower()
            if model_name not in model_map:
                raise ValueError(
                    f"Unsupported model: {model_name}. Available models: {list(model_map.keys())}"
                )

            # Get model-specific params, falling back to class-level params
            model_params = self.model_params.copy()
            model_params.update(model_cfg.get("params", {}))

            # Initialize model with its parameters
            model_class = model_map[model_name]
            model_obj = model_class(
                season_length=model_params.get("season_length", 12),
                **{
                    k: v for k, v in model_params.items() if k not in ["season_length"]
                },  # Already handled above
            )
            model_objects.append(model_obj)

        return model_objects

    def train(
        self,
        df: pd.DataFrame,
        freq: str = "MS",
        test_size: int | None = None,
        n_windows: int = 5,
        step_size: int = 1,
    ) -> dict:
        """
        Train the forecasting model with optional cross-validation.

        Args:
            df: DataFrame with columns ['unique_id', 'ds', 'y']
            freq: Frequency of the time series (default: 'MS' for monthly)
            test_size: Number of periods to use for testing (if None, no test split)
            n_windows: Number of cross-validation windows
            step_size: Step size between cross-validation windows

        Returns:
            Dictionary containing training metrics and results
        """
        self.freq = freq
        self.models = None

        # Ensure proper column names and types
        df = df.rename(columns={"date": "ds", "value": "y"})
        required_cols = ["unique_id", "ds", "y"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Split into train/test if test_size is provided
        if test_size is not None:
            test_cutoff = df["ds"].max() - pd.Timedelta(
                days=test_size * 30
            )  # Approximate for months
            train = df[df["ds"] <= test_cutoff]
            test = df[df["ds"] > test_cutoff]
        else:
            train = df
            test = None

        # Initialize and fit the models
        self.models = StatsForecast(models=self.model_objects, freq=self.freq, n_jobs=-1)

        # Fit the models
        self.models.fit(train)
        self.fitted = True

        # Evaluate on test set if available
        metrics = {}
        if test is not None:
            forecasts = self.predict(horizon=len(test))
            metrics = self.evaluate(test, forecasts)

        return {
            "train_data": train,
            "test_data": test,
            "metrics": metrics,
            "model": self.model_name,
        }

    def predict(self, horizon: int | None = None, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            horizon: Number of periods to forecast
            **kwargs: Additional arguments for the forecast method

        Returns:
            DataFrame with forecasts
        """
        if not self.fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        horizon = horizon or self.horizon
        return self.models.predict(h=horizon, **kwargs)

    @staticmethod
    def evaluate(
        y_true: pd.DataFrame, y_pred: pd.DataFrame, metrics: list[str] | None = None
    ) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            y_true: DataFrame with true values
            y_pred: DataFrame with predicted values
            metrics: List of metrics to calculate. Options: 'mae', 'rmse', 'mse'

        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ["mae", "rmse"]

        results = {}
        y_true = y_true.set_index(["unique_id", "ds"])["y"]
        y_pred = y_pred.set_index(["unique_id", "ds"]).iloc[
            :, 0
        ]  # Get first column of predictions

        common_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true[common_idx]
        y_pred = y_pred[common_idx]

        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y_true, y_pred)
        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        if "mse" in metrics:
            results["mse"] = mean_squared_error(y_true, y_pred)

        return results

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_windows: int = 5,
        step_size: int = 1,
        horizon: int = 12,
    ) -> dict:
        """
        Perform time series cross-validation.

        Args:
            df: Input DataFrame
            n_windows: Number of cross-validation windows
            step_size: Step size between windows
            horizon: Forecast horizon for each window

        Returns:
            Dictionary with cross-validation results
        """
        if not self.fitted:
            self.train(df)

        cv_results = self.models.cross_validation(
            df=df, h=horizon, n_windows=n_windows, step_size=step_size, n_jobs=-1
        )

        return cv_results

    def save(self, path: str | Path | None = None) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model. If None, uses MODEL_DIR from config.
        """
        if not self.fitted:
            raise RuntimeError("Model has not been trained. Nothing to save.")

        if path is None:
            path = Path(MODEL_DIR) / f"{self.model_name}_model"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        self.models.save(str(path))

        # Save model metadata next to the pickle (StatsForecast.save writes a file)
        metadata = {
            "model_name": self.model_name,
            "model_list": self.model_list,
            "model_params": self.model_params,
            "freq": self.freq,
            "horizon": self.horizon,
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ForecastModel":
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded ForecastModel instance
        """
        path = Path(path)

        # Load metadata (sibling .json; fall back to the old directory layout)
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            meta_path = path / "metadata.json"
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Create model instance with saved configuration
        model = cls(
            model_name=metadata.get("model_name") or metadata.get("model_list"),
            **metadata["model_params"],
        )

        # Load the StatsForecast model
        model.models = StatsForecast.load(str(path))
        model.fitted = True
        model.freq = metadata["freq"]
        model.horizon = metadata["horizon"]
        model.model_list = metadata.get("model_list", [{"name": metadata["model_name"]}])

        return model

    def plot_forecast(
        self,
        df: pd.DataFrame,
        forecasts: pd.DataFrame,
        unique_id: str | None = None,
        plot_anomalies: bool = False,
    ) -> None:
        """
        Plot forecasts against actual values.

        Args:
            df: Original DataFrame with actual values
            forecasts: DataFrame with forecasted values
            unique_id: ID of the time series to plot (if multiple)
            plot_anomalies: Whether to highlight anomalies
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        if unique_id:
            df_plot = df[df["unique_id"] == unique_id].copy()
            forecast_plot = forecasts[forecasts["unique_id"] == unique_id].copy()
        else:
            df_plot = df.copy()
            forecast_plot = forecasts.copy()

        # Plot actual values
        plt.plot(df_plot["ds"], df_plot["y"], label="Actual", color="blue", alpha=0.7)

        # Plot forecast
        forecast_plot = forecast_plot.melt(
            id_vars=["unique_id", "ds"], var_name="metric", value_name="value"
        )

        # Plot prediction intervals if available
        if "lo-90" in forecast_plot["metric"].values:
            lo = forecast_plot[forecast_plot["metric"] == "lo-90"]
            hi = forecast_plot[forecast_plot["metric"] == "hi-90"]

            plt.fill_between(
                lo["ds"],
                lo["value"],
                hi["value"],
                color="blue",
                alpha=0.2,
                label="90% Prediction Interval",
            )

        # Plot point forecast
        point_forecast = forecast_plot[forecast_plot["metric"] == self.model_name]
        plt.plot(
            point_forecast["ds"],
            point_forecast["value"],
            label="Forecast",
            color="red",
            linestyle="--",
        )

        plt.title(f"Forecast vs Actuals - {unique_id if unique_id else ''}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage

    import pandas as pd

    # Generate sample data
    dates = pd.date_range(start="2020-01-01", periods=100, freq="MS")
    values = np.sin(np.linspace(0, 20, 100)) * 10 + 50 + np.random.normal(0, 2, 100)
    df = pd.DataFrame({"unique_id": "1", "ds": dates, "y": values})

    # Initialize and train model
    model = ForecastModel(model_name="auto_arima", season_length=12)
    results = model.train(df, test_size=12)

    # Generate forecasts
    forecasts = model.predict(horizon=12)

    # Plot results
    model.plot_forecast(df, forecasts)
