"""
data_io.py — typed table I/O (parquet-preferred, CSV fallback)
==============================================================
The analytical frames (bookings, daily KPIs, external regressors) are read all
over the backend and the training pipeline. CSV re-parses types on every read
(hence the scattered ``parse_dates=``) and is slow on the 180k-row bookings
file; parquet is columnar, typed, and ~4x smaller.

``read_table`` prefers a sibling ``.parquet`` next to the requested ``.csv`` when
one exists and pyarrow is installed, otherwise it falls back to CSV. ``write_table``
writes both, so CSV stays as the portable, human-readable, version-controlled
copy (and the "messy data" demonstration), while parquet is the fast path.

pyarrow is an optional import here: if it's absent the code transparently uses
CSV, so nothing hard-depends on it.
"""
from __future__ import annotations

import os

import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def _parquet_sibling(csv_path: str) -> str:
    """`.../foo.csv` → `.../foo.parquet`."""
    return (csv_path[:-4] if csv_path.endswith(".csv") else csv_path) + ".parquet"


def read_table(csv_path: str, parse_dates=None) -> pd.DataFrame:
    """Read a table, preferring the typed parquet sibling when available.

    `parse_dates` only applies to the CSV path — parquet already carries dtypes,
    so date columns come back as datetimes without re-parsing.
    """
    parquet_path = _parquet_sibling(csv_path)
    if HAS_PARQUET and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path, parse_dates=parse_dates)


def write_table(df: pd.DataFrame, csv_path: str, index: bool = False) -> None:
    """Write CSV (canonical, portable) and, when pyarrow is present, a parquet
    sibling for fast typed reads."""
    df.to_csv(csv_path, index=index)
    if HAS_PARQUET:
        df.to_parquet(_parquet_sibling(csv_path), index=index)
