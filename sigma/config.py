"""Configuration settings for the forecasting pipeline."""
from pathlib import Path
import configparser
import os

# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = Path(__file__).parent.parent / 'config.ini'
config.read(config_path)

# Paths
DATA_PATH = config['Paths'].get('dataset_data')
OUTPUT_DIR = Path(config['Paths'].get('output_dir', 'outputs'))
MODEL_DIR = Path(config['Paths'].get('model_dir', 'models'))

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    'season_length': 12,  # Monthly seasonality
    'n_jobs': -1  # Use all available cores
}

# Forecast horizon
FORECAST_HORIZON = 12  # 12 months ahead
