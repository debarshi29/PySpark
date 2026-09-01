"""Central paths and constants for the Superstore ML project."""
from __future__ import annotations

from pathlib import Path

# ``src`` -> ``ml_superstore``
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ``ml_superstore`` -> ``EY Training``
EY_ROOT = PROJECT_ROOT.parent

RAW_CSV = EY_ROOT / "data" / "Sample - Superstore.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

PROCESSED_DIR = PROJECT_ROOT / "data_processed"

for _d in (FIGURE_DIR, TABLE_DIR, MODEL_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Column groups -----------------------------------------------------------------
DATE_COLS = ["Order Date", "Ship Date"]
NUMERIC_COLS = ["Sales", "Quantity", "Discount", "Profit"]
CATEGORICAL_COLS = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
    "State",
]

RANDOM_STATE = 42
