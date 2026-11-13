"""Configuration settings for the forecasting pipeline."""

import configparser
from pathlib import Path

config = configparser.ConfigParser()
config_path = Path(__file__).parent.parent / "config.ini"
config.read(config_path)

_paths: dict[str, str] = {}
if config.has_section("Paths"):
    _paths = dict(config["Paths"])

DATA_PATH = _paths.get("dataset_data", "data/raw/dataset.csv")
OUTPUT_DIR = Path(_paths.get("output_dir", "outputs"))
MODEL_DIR = Path(_paths.get("model_dir", "models"))

# CCDS compatibility
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
INTERIM_DATA_DIR = Path("data/interim")
for _dir in (OUTPUT_DIR, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

MODEL_PARAMS = {
    "season_length": 12,
}

FORECAST_HORIZON = 12
