"""Load CIC-IDS2017 data. Supports the short CSV (10k) and the full zip (~2.8M rows)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .utils import get_logger

log = get_logger(__name__)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_short_csv(path: str | Path) -> pd.DataFrame:
    log.info(f"Loading short CSV: {path}")
    df = pd.read_csv(path)
    df = _clean_columns(df)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} cols")
    return df


def load_full_zip(zip_path: str | Path) -> pd.DataFrame:
    """Read all CSVs inside cicids2017_full.zip and concat them."""
    zip_path = Path(zip_path)
    log.info(f"Loading full dataset from {zip_path}")
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        log.info(f"Found {len(csv_names)} CSVs inside zip")
        for name in csv_names:
            with zf.open(name) as f:
                df = pd.read_csv(f, low_memory=False, encoding="latin-1")
                df = _clean_columns(df)
                log.info(f"  {name}: {len(df):,} rows")
                frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    log.info(f"Concatenated: {len(full):,} rows, {full.shape[1]} cols")
    return full


def load_dataset(cfg: dict) -> pd.DataFrame:
    if cfg["data"]["use_full_dataset"]:
        return load_full_zip(cfg["paths"]["raw_zip"])
    return load_short_csv(cfg["paths"]["raw_csv"])
