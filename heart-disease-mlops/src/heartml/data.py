"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heartml import config

log = logging.getLogger(__name__)

# ISLR mirror -> UCI code mappings
_CP_MAP = {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4}
_THAL_MAP = {"normal": 3, "fixed": 6, "reversable": 7}


def _fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec - fixed URLs
        return resp.read()


def parse_uci(raw: bytes) -> pd.DataFrame:
    """Parse processed.cleveland.data (no header, '?' marks missing)."""
    return pd.read_csv(io.BytesIO(raw), header=None, names=config.UCI_COLUMNS, na_values="?")


def parse_mirror(raw: bytes) -> pd.DataFrame:
    """Convert the ISLR-format mirror into the UCI column schema."""
    df = pd.read_csv(io.BytesIO(raw), index_col=0)
    out = pd.DataFrame(
        {
            "age": df["Age"],
            "sex": df["Sex"],
            "cp": df["ChestPain"].map(_CP_MAP),
            "trestbps": df["RestBP"],
            "chol": df["Chol"],
            "fbs": df["Fbs"],
            "restecg": df["RestECG"],
            "thalach": df["MaxHR"],
            "exang": df["ExAng"],
            "oldpeak": df["Oldpeak"],
            "slope": df["Slope"],
            "ca": df["Ca"],
            "thal": df["Thal"].map(_THAL_MAP),
            # Mirror only has the binary label; keep UCI semantics (0 = absent).
            "num": (df["AHD"] == "Yes").astype(int),
        }
    )
    return out.reset_index(drop=True)


def download_raw(dest: Path = config.RAW_DATA_PATH, use_mirror: bool = False) -> pd.DataFrame:
    """Download the dataset, trying UCI first and the GitHub mirror second."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sources = [] if use_mirror else [("UCI", config.UCI_URL, parse_uci)]
    sources.append(("mirror", config.MIRROR_URL, parse_mirror))

    last_err: Exception | None = None
    for name, url, parser in sources:
        try:
            log.info("Downloading from %s: %s", name, url)
            df = parser(_fetch(url))
            df.to_csv(dest, index=False)
            log.info("Saved %d rows to %s (source=%s)", len(df), dest, name)
            return df
        except Exception as err:  # noqa: BLE001 - try next source
            log.warning("Source %s failed: %s", name, err)
            last_err = err
    raise RuntimeError(f"All download sources failed: {last_err}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw table.

    * coerce every column to numeric ('?' and junk -> NaN)
    * values outside plausible clinical ranges / unknown codes -> NaN
    * drop exact duplicate rows and rows without a label
    * binarise the target: num > 0 -> 1 (disease present)

    Remaining NaNs are deliberately kept: they are imputed *inside* the model
    pipeline so the same logic runs at training and serving time without leakage.
    """
    df = df.copy()
    missing_cols = set(config.UCI_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    for col in config.UCI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, (lo, hi) in config.VALID_RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    for col, codes in config.VALID_CODES.items():
        df.loc[~df[col].isin(codes) & df[col].notna(), col] = np.nan

    df = df.dropna(subset=["num"]).drop_duplicates().reset_index(drop=True)
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df[config.FEATURES + [config.TARGET]]


def load_clean(path: Path = config.CLEAN_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURES], df[config.TARGET]
