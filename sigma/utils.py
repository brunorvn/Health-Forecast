"""Utility functions for the forecasting pipeline."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the config file. If None, uses default path.

    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: Configuration dictionary to save
        config_path: Path to save the config file. If None, uses default path.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate various forecast accuracy metrics.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        dict: Dictionary of metric names and values
    """
    # Calculate errors
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    # Percentage errors are undefined at zero demand, which is common in the
    # HC series: average over the non-zero points and return NaN otherwise.
    nonzero_true = y_true != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero_true] / y_true[nonzero_true])) * 100)
        if nonzero_true.any()
        else float("nan")
    )
    denom = np.abs(y_true) + np.abs(y_pred)
    nonzero_denom = denom != 0
    smape = (
        float(2.0 * np.mean(np.abs(errors[nonzero_denom]) / denom[nonzero_denom]) * 100)
        if nonzero_denom.any()
        else float("nan")
    )

    # Calculate metrics
    metrics = {
        "mae": np.mean(abs_errors),  # Mean Absolute Error
        "mse": np.mean(squared_errors),  # Mean Squared Error
        "rmse": np.sqrt(np.mean(squared_errors)),  # Root Mean Squared Error
        "mape": mape,  # Mean Absolute Percentage Error (zero-safe)
        "smape": smape,  # Symmetric MAPE (zero-safe)
    }

    return metrics


def ensure_directory_exists(directory: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_latest_file(directory: Path, pattern: str = "*") -> Path | None:
    """
    Get the most recently modified file in a directory matching a pattern.

    Args:
        directory: Directory to search in
        pattern: File pattern to match (e.g., '*.csv')

    Returns:
        Path: Path to the latest file, or None if no files match
    """
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)
