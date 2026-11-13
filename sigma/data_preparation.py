"""Data preparation — deep module.

Public API: prepare_data(path) -> DataFrame[unique_id, ds, y]
Esconde: detecção wide/long, delimiter/encoding, Total, melt, parse PT-BR.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sigma.config import DATA_PATH

# ---- private: wide detection & parsing (não exposto) --------------------

_PT_MONTH = {
    "janeiro": "01",
    "fevereiro": "02",
    "março": "03",
    "marco": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12",
    "jan": "01",
    "fev": "02",
    "mar": "03",
    "abr": "04",
    "mai": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "set": "09",
    "sep": "09",
    "out": "10",
    "nov": "11",
    "dez": "12",
}

_DATE_PAT = re.compile(r"^\s*([A-Za-zÀ-ú]+)\s*[/\-\s]\s*(\d{4})\s*$")


def _parse_pt_date(s: str) -> pd.Timestamp | None:
    s = str(s).strip()
    if not s or s.lower() == "total":
        return None
    # tenta ISO primeiro
    try:
        return pd.to_datetime(s, format="mixed")
    except Exception:
        pass
    m = _DATE_PAT.match(s)
    if not m:
        # tenta "01/2018" ou "1/2018"
        if re.match(r"^\s*\d{1,2}\s*[/\-]\s*\d{4}\s*$", s):
            try:
                return pd.to_datetime(s, format="mixed", dayfirst=True)
            except Exception:
                return None
        return None
    mon, year = m.group(1).lower(), m.group(2)
    mm = _PT_MONTH.get(mon)
    if not mm:
        return None
    return pd.to_datetime(f"{year}-{mm}-01")


def _looks_like_date_col(name: str) -> bool:
    return _parse_pt_date(name) is not None


def _is_wide(df: pd.DataFrame) -> bool:
    if {"unique_id", "ds", "y"}.issubset(df.columns):
        return False
    # wide: primeira coluna é id, demais parecem Jan/2018, Janeiro/2018, 01/2018...
    if len(df.columns) < 3:
        return False
    date_like = sum(_looks_like_date_col(c) for c in df.columns[1:])
    return date_like >= 2 and date_like / (len(df.columns) - 1) >= 0.5


def _read_csv_robust(path: Path) -> pd.DataFrame:
    # tenta utf-8 e latin-1, , e ;
    for enc in ("utf-8", "latin-1"):
        for sep in (",", ";"):
            try:
                # não força dtype ainda — precisamos inspecionar colunas
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str, engine="python")
                # heurística: se só 1 coluna, sep errado
                if len(df.columns) == 1 and sep == ",":
                    continue
                return df
            except Exception:
                continue
    # fallback
    return pd.read_csv(path, dtype=str)


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # copia para não mutar original
    df = df.copy()
    # remove coluna Total se existir
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])
    # remove linha Total (case-insensitive, em qualquer coluna)
    id_col = df.columns[0]
    # normaliza id_col: strip
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df[~df[id_col].str.lower().eq("total")]
    # também remove onde id_col vazio
    df = df[df[id_col].astype(str).str.strip().ne("")]

    # melt
    long = df.melt(id_vars=[id_col], var_name="ds", value_name="y")
    # parse ds
    long["ds"] = long["ds"].apply(_parse_pt_date)
    long = long.dropna(subset=["ds"])
    # y para int
    long["y"] = pd.to_numeric(long["y"], errors="coerce").fillna(0).astype(int)
    long = long.rename(columns={id_col: "unique_id"})
    # unique_id: mantém string, corta para 10 como no original mas preserva original se curto
    long["unique_id"] = long["unique_id"].astype(str).str.strip()
    long = long.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return long[["unique_id", "ds", "y"]]


# ---- public --------------------------------------------------------------


def prepare_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Lê CSV long (unique_id,ds,y) ou wide (id, Jan/2018, Fev/2018, ...).

    Wide é o padrão do DATASUS/Ministério — cada mês é uma coluna.
    Detecção é automática; melt fica escondido no deep module.
    """
    if file_path is None:
        file_path = DATA_PATH

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    raw = _read_csv_robust(file_path)

    # normaliza nomes: strip
    raw.columns = [str(c).strip() for c in raw.columns]

    if _is_wide(raw):
        df = _wide_to_long(raw)
    else:
        # long path — já no contrato, mas tolera variações de nome
        # aceita 'date'/'value' legados
        df = raw.rename(columns={"date": "ds", "value": "y"})
        # garante colunas
        if not {"unique_id", "ds", "y"}.issubset(df.columns):
            raise ValueError(
                f"Input data must contain columns ['unique_id','ds','y'] "
                f"or be wide with date columns. Got: {list(raw.columns)[:6]}..."
            )
        df = df[["unique_id", "ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"], format="mixed", errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(np.int64)
        df = df.dropna(subset=["ds"])

    # tipos finais
    df["unique_id"] = df["unique_id"].astype("category")
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(np.int64)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df[["unique_id", "ds", "y"]].copy()


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
    test_dates = (
        df.groupby("unique_id", observed=True)["ds"]
        .nlargest(test_size)
        .reset_index(level=0, drop=True)
    )

    # Split into train and test
    test_mask = df.index.isin(test_dates.index)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, test_df
