import pandas as pd
import numpy as np
from pathlib import Path

from sigma.data_preparation import prepare_data, train_test_split
from sigma.utils import calculate_metrics


def test_prepare_data_reads_csv(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    csv.write_text("unique_id,ds,y\n0101,2018-01-01,10\n0101,2018-02-01,12\n0102,2018-01-01,5\n")
    df = prepare_data(csv)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])


def test_train_test_split_temporal():
    dates = pd.date_range("2020-01-01", periods=14, freq="MS")
    df = pd.DataFrame(
        {
            "unique_id": ["0101"] * 14,
            "ds": dates,
            "y": range(14),
        }
    )
    train, test = train_test_split(df, test_size=3)
    assert len(test) == 3
    assert len(train) == 11
    assert test["ds"].min() > train["ds"].max()


def test_prepare_data_wide_pt(tmp_path: Path):
    # DATASUS-like wide: cada mês é coluna, com Total
    csv = tmp_path / "wide.csv"
    csv.write_text(
        "Procedimento;Jan/2018;Fev/2018;Mar/2018;Total\n"
        "0101;10;20;30;60\n"
        "0102;5;6;7;18\n"
        "Total;15;26;37;78\n",
        encoding="latin-1",
    )
    df = prepare_data(csv)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert len(df) == 6  # 2 ids × 3 meses, Total removido
    assert set(df["unique_id"].cat.categories) == {"0101", "0102"}
    assert df["y"].tolist() == [10, 20, 30, 5, 6, 7]


def test_prepare_data_wide_full_month(tmp_path: Path):
    csv = tmp_path / "wide2.csv"
    csv.write_text(
        "SubGrup proced;Janeiro/2018;Fevereiro/2018;Março/2018\nA;1;2;3\nB;4;5;6\n",
        encoding="utf-8",
    )
    df = prepare_data(csv)
    assert len(df) == 6
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])


def test_calculate_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    m = calculate_metrics(y_true, y_pred)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    # off by 1
    y_pred2 = np.array([11.0, 21.0, 31.0])
    m2 = calculate_metrics(y_true, y_pred2)
    assert m2["mae"] == 1.0
    assert "mape" in m2 and "smape" in m2
