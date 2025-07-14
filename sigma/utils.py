"""Utility functions for the forecasting pipeline."""
from typing import Dict, Any, Optional
import json
from pathlib import Path
import numpy as np


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the config file. If None, uses default path.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'config.json'
    
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the config file. If None, uses default path.
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    squared_errors = errors ** 2
    
    # Calculate metrics
    metrics = {
        'mae': np.mean(abs_errors),  # Mean Absolute Error
        'mse': np.mean(squared_errors),  # Mean Squared Error
        'rmse': np.sqrt(np.mean(squared_errors)),  # Root Mean Squared Error
        'mape': np.mean(np.abs(errors / y_true)) * 100,  # Mean Absolute Percentage Error
        'smape': 2.0 * np.mean(np.abs(errors) / (np.abs(y_true) + np.abs(y_pred))) * 100  # Symmetric MAPE
    }
    
    return metrics


def ensure_directory_exists(directory: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_latest_file(directory: Path, pattern: str = '*') -> Optional[Path]:
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
