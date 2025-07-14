# SIGMA

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Sistema Inteligente de Gestão da Monitoria e Antecipação

# HC Demand Forecasting

Production analysis and prediction of UFPE university hospital using time series forecasting models.

## Features

- **Multiple Models**: Supports AutoARIMA and AutoETS models
- **Easy Configuration**: Simple configuration via `config.ini`
- **Modular Design**: Well-organized codebase for easy extension
- **Comprehensive Evaluation**: Multiple evaluation metrics and visualization
- **Command-line Interface**: Simple CLI for training, evaluation, and forecasting

## Project Structure

```
hc-demand-forecasting/
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- This file
├── config.ini         <- Configuration file
├── requirements.txt   <- Python dependencies
│
├── data/             <- All project data, ignored by git
│   ├── external/      <- Data from third party sources
│   ├── interim/       <- Intermediate data that has been transformed
│   ├── processed/     <- The final, canonical data sets for modeling
│   └── raw/           <- The original, immutable data dump
│
├── docs/             <- Documentation
│
├── models/           <- Trained and serialized models
│
├── notebooks/        <- Jupyter notebooks for exploration and analysis
│
├── outputs/          <- Generated outputs (plots, reports, etc.)
│
└── src/              <- Source code for this project
    ├── config.py          # Configuration settings
    ├── data_preparation.py # Data loading and preprocessing
    ├── models.py          # Model definitions and training
    ├── forecast.py        # Forecasting pipeline and evaluation
    ├── utils.py           # Utility functions
    └── main.py            # Main script
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/hc-demand-forecasting.git
   cd hc-demand-forecasting
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration

1. Copy the example configuration file:

   ```bash
   cp config.ini.example config.ini
   ```

2. Edit `config.ini` with your data paths and settings.

### Command Line Interface

```bash
# Train and evaluate a model
python -m src.main --train --evaluate --model auto_arima

# Generate forecasts with a pre-trained model
python -m src.main --forecast --plot --model auto_arima

# Use an ensemble of models
python -m src.main --train --evaluate --forecast --plot --model ensemble
```

### Available Models

- `auto_arima`: AutoARIMA model
- `auto_ets`: ETS (Error, Trend, Seasonality) model
- `ensemble`: Ensemble of AutoARIMA and AutoETS models

### Data Format

The input CSV file should have the following columns:

- `unique_id`: Identifier for the time series
- `ds`: Date in YYYY-MM-DD format
- `y`: Numeric value to forecast

Example:

```csv
unique_id,ds,y
A,2023-01-01,100
A,2023-02-01,120
B,2023-01-01,50
B,2023-02-01,55
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
ruff check --fix .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
