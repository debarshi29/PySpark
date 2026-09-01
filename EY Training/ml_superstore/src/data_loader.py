"""Loading and cleaning helpers for the Sample - Superstore dataset."""
from __future__ import annotations

import pandas as pd

from .config import DATE_COLS, RAW_CSV

# The public Superstore CSV ships in US m/d/Y format with a latin-1 codepage.
_ENCODINGS = ("utf-8", "latin-1", "cp1252")


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV, trying a few encodings that this file is known to use."""
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as err:  # pragma: no cover
            last_err = err
    raise RuntimeError(f"Could not read {path} with {_ENCODINGS}") from last_err


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, de-duplicated copy of the raw frame.

    * parses the two date columns
    * strips whitespace from object columns
    * drops the fully duplicated ``Row ID`` rows if any
    * adds ``Ship Days`` (calendar days between order and ship)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)

    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    df = df.drop_duplicates()
    if "Row ID" in df:
        df = df.drop_duplicates(subset="Row ID").sort_values("Row ID")

    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df.reset_index(drop=True)


def load_clean(path=RAW_CSV) -> pd.DataFrame:
    """Convenience wrapper: ``load_raw`` followed by ``clean``."""
    return clean(load_raw(path))


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, null counts, cardinality and sample values."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_null": int(s.isna().sum()),
                "pct_null": round(s.isna().mean() * 100, 3),
                "n_unique": int(s.nunique(dropna=True)),
                "sample": ", ".join(map(str, s.dropna().unique()[:3])),
            }
        )
    return pd.DataFrame(rows)
