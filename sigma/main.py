"""Main script for the forecasting pipeline."""
from typing import Optional, List, Dict, Union
from pathlib import Path
import typer
from loguru import logger
import json

from sigma.config import DATA_PATH, OUTPUT_DIR, MODEL_DIR, FORECAST_HORIZON
from sigma.data_preparation import prepare_data, train_test_split
from sigma.models import ForecastModel, create_ensemble
from sigma.forecast import evaluate_forecasts, plot_forecasts, save_forecasts
from sigma.utils import ensure_directory_exists

app = typer.Typer(help="Time Series Forecasting Pipeline")


def run_pipeline(
    data_path: Path = typer.Option(
        DATA_PATH,
        "--data-path",
        help="Path to the input CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        OUTPUT_DIR,
        "--output-dir",
        help="Directory to save outputs",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    model_dir: Path = typer.Option(
        MODEL_DIR,
        "--model-dir",
        help="Directory to save/load models",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    horizon: int = typer.Option(
        FORECAST_HORIZON,
        "--horizon",
        help="Forecast horizon (number of periods)",
        min=1,
    ),
    model_name: str = typer.Option(
        None,
        "--model",
        help="Model to use for forecasting (comma-separated for multiple models, e.g., 'auto_arima,auto_ets')",
        case_sensitive=False,
    ),
    model_config: str = typer.Option(
        None,
        "--model-config",
        help="Path to a JSON file containing model configurations",
    ),
    train: bool = typer.Option(
        False, "--train", help="Train a new model"
    ),
    evaluate: bool = typer.Option(
        False, "--evaluate", help="Evaluate model on test set"
    ),
    forecast: bool = typer.Option(
        False, "--forecast", help="Generate forecasts"
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Generate forecast plots"
    ),
):
    """Run the forecasting pipeline with the given configuration."""
    # Define available models from models.py
    available_models = ["auto_arima", "auto_ets", "theta", "auto_ces", "auto_tbats", "auto_theta"]
    
    # Parse model configurations
    if model_config:
        # Load model configurations from file
        try:
            with open(model_config, 'r') as f:
                model_configs = json.load(f)
            if not isinstance(model_configs, list):
                raise ValueError("Model config must be a list of model configurations")
            model_list = model_configs
            model_name = 'ensemble' if len(model_configs) > 1 else model_configs[0].get('name', 'model')
        except Exception as e:
            raise typer.BadParameter(f"Error loading model config: {str(e)}")
    elif model_name:
        # Parse comma-separated model names
        model_names = [m.strip() for m in model_name.split(',')]
        model_list = [{'name': m} for m in model_names]
        model_name = 'ensemble' if len(model_names) > 1 else model_names[0]
    else:
        raise typer.BadParameter("Either --model or --model-config must be specified")
    
    # Validate model names
    for model_cfg in model_list:
        model_name = model_cfg.get('name', '').lower()
        if model_name not in available_models:
            raise typer.BadParameter(
                f"Invalid model: {model_name}. Available models: {', '.join(available_models)}"
            )
    
    logger.info(f"Using models: {[m['name'] for m in model_list]}")
    
    # Ensure at least one action is specified
    if not any([train, evaluate, forecast]):
        raise typer.BadParameter("At least one of --train, --evaluate, or --forecast must be specified")
    
    # Ensure output and model directories exist
    for directory in [output_dir, model_dir]:
        ensure_directory_exists(directory)
    
    # Convert string paths to Path objects if they aren't already
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    
    # Ensure directories exist
    for directory in [output_dir, model_dir]:
        ensure_directory_exists(directory)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    df = prepare_data(data_path)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=horizon)
    logger.info(f"Training data: {len(train_df)} rows, Test data: {len(test_df)} rows")
    
    # Initialize model(s)
    model = ForecastModel(model_name=model_list)
    
    # Train model if requested
    if train:
        logger.info(f"Training {model_name} model...")
        model.train(train_df)
        
        # Save the trained model
        model_path = model_dir / f"{model_name}_model.pkl"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    else:
        # Try to load a pre-trained model
        model_path = model_dir / f"{model_name}_model.pkl"
        if model_path.exists():
            model = ForecastModel.load(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.warning(f"No pre-trained model found at {model_path}. Use --train to train a new model.")
            return
    
    # Evaluate model on test set
    if evaluate and not test_df.empty:
        logger.info("Evaluating model on test set...")
        
        # Generate forecasts for the test period
        forecasts = model.predict(h=horizon)
        
        # For multiple models, no need to rename as StatsForecast handles this
        if len(model_list) == 1:
            # For single model, ensure consistent column naming
            forecast_columns = {col: f"{model_name}_{col}" for col in forecasts.columns 
                             if col not in ['unique_id', 'ds'] and not any(m in col for m in available_models)}
            if forecast_columns:
                forecasts = forecasts.rename(columns=forecast_columns)
        
        # Evaluate forecasts
        metrics = evaluate_forecasts(forecasts, test_df)
        logger.info("\nEvaluation Metrics:")
        logger.info(metrics.to_string())
        
        # Save evaluation results
        metrics_path = output_dir / f"{model_name}_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate forecasts
    if forecast:
        logger.info("Generating forecasts...")
        
        # Generate forecasts for the specified horizon
        forecasts = model.predict(h=horizon)
        
        # Save forecasts
        forecast_path = save_forecasts(forecasts, model_name, output_dir)
        logger.info(f"Forecasts saved to {forecast_path}")
        
        # Generate and save plots if requested
        if plot:
            plot_dir = output_dir / 'plots'
            ensure_directory_exists(plot_dir)
            
            # Plot for each unique time series and each model
            for unique_id in forecasts['unique_id'].unique():
                for model_cfg in model_list:
                    current_model = model_cfg['name']
                    plot_path = plot_dir / f"{current_model}_forecast_{unique_id}.png"
                    plot_forecasts(
                        train_df=train_df,
                        test_df=test_df,
                        forecasts=forecasts,
                        model_name=current_model,
                        unique_id=unique_id,
                        save_path=plot_path
                    )
            logger.info(f"Plots saved to {plot_dir}")


@app.callback()
def main():
    """Time Series Forecasting Pipeline."""
    pass


if __name__ == "__main__":
    logger.remove()
    logger.add("forecasting.log", rotation="10 MB")
    app()
