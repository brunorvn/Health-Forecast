"""Generate synthetic HC-like dataset for reproducible demo.

Replaces the CCDS stub. Produces monthly series that mimic the real HC data
(unique_id, ds, y) with season_length=12, so the repo is runnable without the
private HC dump. Deterministic via seed.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from sigma.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(help="Generate synthetic dataset")


def _one_series(uid: str, start: str, periods: int, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="MS")
    t = np.arange(periods)
    # trend + yearly seasonality + noise, keeping y >= 0
    trend = 80 + 0.6 * t + 8 * np.sin(2 * np.pi * t / 12)
    # uid-specific level and noise scale so series are distinguishable
    level = rng.integers(-20, 20)
    seasonal_amp = rng.uniform(10, 30)
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / 12 + rng.uniform(0, np.pi))
    noise = rng.normal(0, 12, size=periods)
    y = np.maximum(0, np.round(trend + level + seasonal + noise)).astype(int)
    return pd.DataFrame({"unique_id": uid, "ds": dates, "y": y})


@app.command()
def main(
    output_path: Path = typer.Option(
        RAW_DATA_DIR / "dataset.csv", "--output", "-o", help="Where to write raw CSV"
    ),
    processed_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "dataset.csv", "--processed", help="Where to write processed copy"
    ),
    n_series: int = typer.Option(6, "--n-series", help="Number of unique_id series"),
    start: str = typer.Option("2018-01-01", "--start", help="First month (YYYY-MM-DD)"),
    periods: int = typer.Option(72, "--periods", help="Months per series (72=6y)"),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility"),
):
    """Generate synthetic dataset and write raw + processed copies."""
    rng = np.random.default_rng(seed)
    # HC-like IDs: 0101, 0102, 0201 ... keep categorical-like strings
    base_ids = ["0101", "0102", "0201", "0202", "0301", "0401", "0501", "0701"]
    uids = base_ids[:n_series]

    logger.info(f"Generating {n_series} series x {periods} months (seed={seed})")
    df = pd.concat([_one_series(uid, start, periods, rng) for uid in uids], ignore_index=True)
    df["unique_id"] = df["unique_id"].astype("category")
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    for p in (output_path, processed_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        logger.success(f"Wrote {len(df)} rows to {p}")

    # integrity check
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert df["y"].min() >= 0
    logger.info(f"Done. Example:\n{df.head().to_string(index=False)}")


if __name__ == "__main__":
    app()
