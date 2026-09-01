"""Supervised modelling helpers: preprocessing, model zoo, time split, evaluation.

Used by ``05_supervised_models.ipynb`` and ``src/pipeline.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame, make_supervised


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    return ColumnTransformer(
        [("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def _pipe(model, scale: bool = True) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(scale)), ("model", model)])


# ---------------------------------------------------------------------------
# model zoos
# ---------------------------------------------------------------------------
def regressors() -> dict[str, Pipeline]:
    return {
        "Dummy(mean)": _pipe(DummyRegressor(strategy="mean"), scale=False),
        "Ridge": _pipe(Ridge(alpha=1.0)),
        "RandomForest": _pipe(
            RandomForestRegressor(n_estimators=300, max_depth=None,
                                  min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
            scale=False,
        ),
        "GradBoost": _pipe(
            GradientBoostingRegressor(random_state=RANDOM_STATE), scale=False
        ),
    }


def classifiers() -> dict[str, Pipeline]:
    return {
        "Dummy(prior)": _pipe(DummyClassifier(strategy="prior"), scale=False),
        "LogReg": _pipe(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "RandomForest": _pipe(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                   class_weight="balanced", random_state=RANDOM_STATE),
            scale=False,
        ),
        "GradBoost": _pipe(
            GradientBoostingClassifier(random_state=RANDOM_STATE), scale=False
        ),
    }


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, cutoff: str = "2017-01-01"):
    """Return boolean masks ``(train, test)`` on ``Order Date`` (line level)."""
    cutoff = pd.Timestamp(cutoff)
    order_date = df["Order Date"]
    return order_date < cutoff, order_date >= cutoff


def build_xy(df: pd.DataFrame, cutoff: str = "2017-01-01"):
    """One-stop: features + targets + time-based train/test split."""
    feat = build_feature_frame(df)
    X, y_reg, y_clf = make_supervised(df)
    tr, te = time_split(feat, cutoff)
    return {
        "X_train": X[tr], "X_test": X[te],
        "yreg_train": y_reg[tr], "yreg_test": y_reg[te],
        "yclf_train": y_clf[tr], "yclf_test": y_clf[te],
        "train_mask": tr, "test_mask": te,
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def eval_regression(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def eval_classification(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "F1": f1_score(y_true, pred),
        "threshold": threshold,
    }


def best_f1_threshold(y_true, proba, grid=None) -> tuple[float, float]:
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 19)
    scores = [(t, f1_score(y_true, (np.asarray(proba) >= t).astype(int))) for t in grid]
    return max(scores, key=lambda x: x[1])
