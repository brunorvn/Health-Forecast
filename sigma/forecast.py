"""Forecasting pipeline and evaluation."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from utilsforecast.evaluation import evaluate

from .config import OUTPUT_DIR


def evaluate_forecasts(
    forecasts: pd.DataFrame, test_df: pd.DataFrame, metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Evaluate forecasts against test data.

    Args:
        forecasts: DataFrame with forecasted values
        test_df: DataFrame with actual values
        metrics: List of metrics to calculate (default: ['mse'])

    Returns:
        pd.DataFrame: Evaluation metrics
    """
    if metrics is None:
        metrics = ["mse"]

    # Ensure we have the required columns
    required_cols = ["unique_id", "ds", "y"]
    if not all(col in test_df.columns for col in required_cols):
        raise ValueError(f"Test data must contain columns: {required_cols}")

    # Merge forecasts with actual values
    eval_df = forecasts.merge(
        test_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left"
    )

    # Rename columns to match utilsforecast expected format
    eval_df = eval_df.rename(columns={"y": "y_test"})

    # Calculate metrics
    results = []
    for model in [
        c
        for c in forecasts.columns
        if c not in ["unique_id", "ds"] and not any(x in c for x in ["-lo-", "-hi-"])
    ]:
        model_eval = evaluate(
            df=eval_df,
            metrics=metrics,
            models=[model],
            target_df=test_df,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )
        results.append(model_eval)

    return pd.concat(results)


def plot_forecasts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    model_name: str,
    unique_id: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot forecasts against actual values.

    Args:
        train_df: Training data
        test_df: Test data
        forecasts: Forecasted values
        model_name: Name of the model (for plot title)
        unique_id: Specific time series ID to plot (if None, plots first one)
        save_path: Path to save the plot (if None, displays plot)
    """
    if unique_id is None:
        unique_id = train_df["unique_id"].iloc[0]

    # Filter data for the specified time series
    train_ts = train_df[train_df["unique_id"] == unique_id].sort_values("ds")
    test_ts = test_df[test_df["unique_id"] == unique_id].sort_values("ds")
    forecast_ts = forecasts[forecasts["unique_id"] == unique_id].sort_values("ds")

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(train_ts["ds"], train_ts["y"], label="Training Data", color="#1f77b4", alpha=0.7)

    # Plot test data
    plt.plot(test_ts["ds"], test_ts["y"], label="Actual", color="#2ca02c", alpha=0.7)

    # Plot forecasts
    plt.plot(
        forecast_ts["ds"],
        forecast_ts[model_name],
        label=f"{model_name} Forecast",
        color="#ff7f0e",
        linestyle="--",
    )

    # Add prediction intervals if available
    lo_col = f"{model_name}-lo-90"
    hi_col = f"{model_name}-hi-90"

    if all(col in forecast_ts.columns for col in [lo_col, hi_col]):
        plt.fill_between(
            forecast_ts["ds"],
            forecast_ts[lo_col],
            forecast_ts[hi_col],
            color="#ff7f0e",
            alpha=0.2,
            label="90% Prediction Interval",
        )

    plt.title(f"Forecast for {unique_id} - {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_forecasts(forecasts: pd.DataFrame, model_name: str, output_dir: Path = None) -> Path:
    """
    Save forecasts to a CSV file.

    Args:
        forecasts: DataFrame with forecasted values
        model_name: Name of the model
        output_dir: Directory to save the output (default: OUTPUT_DIR from config)

    Returns:
        Path: Path to the saved file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{model_name}_forecasts.csv"
    forecasts.to_csv(file_path, index=False)
    return file_path
