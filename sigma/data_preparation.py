"""Data preparation module for the forecasting pipeline."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from sigma.config import DATA_PATH


def prepare_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Read and preprocess the time series data.

    Args:
        file_path: Path to the CSV file. If None, uses the path from config.

    Returns:
        pd.DataFrame: Processed DataFrame with columns ['unique_id', 'ds', 'y']
    """
    if file_path is None:
        file_path = DATA_PATH

    # Ensure the file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Define data types and parse dates
    data_types = {"ds": "str", "unique_id": "category", "y": np.int64}

    # Read CSV with specified data types and parse dates
    df = pd.read_csv(file_path, dtype=data_types, parse_dates=["ds"])

    # Convert 'ds' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m")

    # Ensure we have the required columns in the right order
    required_columns = ["unique_id", "ds", "y"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input data must contain columns: {required_columns}")

    return df[required_columns].copy()


def train_test_split(df: pd.DataFrame, test_size: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets.

    Args:
        df: Input DataFrame with time series data
        test_size: Number of periods to include in the test set

    Returns:
        tuple: (train_df, test_df)
    """
    # Sort by date to ensure proper time-based splitting
    df = df.sort_values(["unique_id", "ds"]).copy()

    # Get the last 'test_size' periods for each time series
    test_dates = df.groupby("unique_id")["ds"].nlargest(test_size).reset_index(level=0, drop=True)

    # Split into train and test
    test_mask = df.index.isin(test_dates.index)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, test_df
