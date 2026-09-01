"""Feature engineering for the Superstore dataset.

Two consumers:

* **Supervised models** (`04`/`05`) — line-level model matrix via
  ``build_feature_frame`` / ``make_supervised``.
* **Forecasting** (`04`) — ``monthly_sales`` returns a clean monthly Series.

Historical customer aggregates are computed with ``groupby(...).shift`` /
expanding windows so a row never sees its own or future orders (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE  # noqa: F401  (re-exported for convenience)

# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["profit_margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], 0.0)
    df["is_loss"] = (df["Profit"] < 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# date parts
# ---------------------------------------------------------------------------
def add_date_features(df: pd.DataFrame, col: str = "Order Date") -> pd.DataFrame:
    df = df.copy()
    d = df[col].dt
    df["order_year"] = d.year
    df["order_month"] = d.month
    df["order_quarter"] = d.quarter
    df["order_week"] = d.isocalendar().week.astype(int)
    df["order_dayofweek"] = d.dayofweek
    df["order_dayofyear"] = d.dayofyear
    df["order_is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["order_is_month_end"] = d.is_month_end.astype(int)
    df["order_is_quarter_end"] = d.is_quarter_end.astype(int)
    # cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["order_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["order_dayofweek"] / 7)
    return df


# ---------------------------------------------------------------------------
# line-level derived features
# ---------------------------------------------------------------------------
def add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_per_unit"] = df["Sales"] / df["Quantity"].replace(0, np.nan)
    df["price_per_unit"] = df["price_per_unit"].fillna(df["Sales"])
    df["log_sales"] = np.log1p(df["Sales"].clip(lower=0))
    df["is_discounted"] = (df["Discount"] > 0).astype(int)
    df["discount_bucket"] = pd.cut(
        df["Discount"],
        bins=[-0.01, 0.0, 0.15, 0.30, 0.50, 1.0],
        labels=["none", "low", "mid", "high", "extreme"],
    ).astype(str)
    df["ship_days"] = df.get("Ship Days", (df["Ship Date"] - df["Order Date"]).dt.days)
    return df


# ---------------------------------------------------------------------------
# leakage-safe historical customer aggregates
# ---------------------------------------------------------------------------
def add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Customer ID", "Order Date", "Row ID"]).copy()
    g = df.groupby("Customer ID", group_keys=False)

    df["cust_prior_lines"] = g.cumcount()
    df["cust_prior_sales_sum"] = (
        g["Sales"].apply(lambda s: s.shift().expanding().sum()).fillna(0.0)
    )
    df["cust_prior_sales_mean"] = (
        g["Sales"].apply(lambda s: s.shift().expanding().mean()).fillna(0.0)
    )
    df["cust_prior_loss_rate"] = (
        g["is_loss"].apply(lambda s: s.shift().expanding().mean()).fillna(0.0)
    )
    first_order = g["Order Date"].transform("min")
    df["cust_days_since_first"] = (df["Order Date"] - first_order).dt.days
    df["cust_is_first_order"] = (df["cust_prior_lines"] == 0).astype(int)

    return df.sort_values("Row ID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# orchestrators
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Sales", "Quantity", "Discount", "ship_days",
    "price_per_unit", "log_sales", "is_discounted",
    "order_year", "order_month", "order_quarter", "order_week",
    "order_dayofweek", "order_dayofyear", "order_is_weekend",
    "order_is_month_end", "order_is_quarter_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "cust_prior_lines", "cust_prior_sales_sum", "cust_prior_sales_mean",
    "cust_prior_loss_rate", "cust_days_since_first", "cust_is_first_order",
]
CATEGORICAL_FEATURES = [
    "Ship Mode", "Segment", "Region", "Category", "Sub-Category",
    "discount_bucket",
]
TARGET_REG = "Profit"
TARGET_CLF = "is_loss"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full line-level feature matrix (targets included)."""
    df = add_targets(df)
    df = add_date_features(df)
    df = add_line_features(df)
    df = add_customer_history(df)
    return df


def make_supervised(df: pd.DataFrame):
    """Return ``(X, y_reg, y_clf)`` ready for an sklearn ``Pipeline``.

    ``X`` keeps categoricals as raw strings — encode them inside the model
    pipeline (see ``05_supervised_models.ipynb``).
    """
    feat = build_feature_frame(df)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = feat[cols].copy()
    y_reg = feat[TARGET_REG].copy()
    y_clf = feat[TARGET_CLF].copy()
    return X, y_reg, y_clf


# ---------------------------------------------------------------------------
# forecasting helpers
# ---------------------------------------------------------------------------
def monthly_sales(df: pd.DataFrame, value: str = "Sales") -> pd.Series:
    """Month-start indexed Series of summed ``value`` with no gaps."""
    s = (
        df.set_index("Order Date")
        .sort_index()[value]
        .resample("MS")
        .sum()
        .asfreq("MS", fill_value=0.0)
    )
    s.name = f"monthly_{value.lower()}"
    return s


def supervised_time_features(s: pd.Series, n_lags: int = 12) -> pd.DataFrame:
    """Turn a monthly Series into a lag/rolling feature table for regression."""
    out = pd.DataFrame({"y": s})
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = s.shift(lag)
    out["roll_mean_3"] = s.shift(1).rolling(3).mean()
    out["roll_mean_6"] = s.shift(1).rolling(6).mean()
    out["roll_std_3"] = s.shift(1).rolling(3).std()
    out["month"] = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["time_idx"] = np.arange(len(out))
    return out


def rfm_table(df: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    """Classic Recency / Frequency / Monetary table with 1-5 quintile scores."""
    snapshot = pd.Timestamp(snapshot) if snapshot is not None else df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        recency=("Order Date", lambda x: (snapshot - x.max()).days),
        frequency=("Order ID", "nunique"),
        monetary=("Sales", "sum"),
    )
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)
    return rfm.reset_index()
