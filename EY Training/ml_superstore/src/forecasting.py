"""Monthly-sales forecasting: model wrappers, metrics and a rolling backtest.

All forecasters share a tiny interface — ``fit(train)`` then
``predict(h)`` returning a Series indexed by future month-starts — so the
backtest loop treats them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import supervised_time_features

FREQ = "MS"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def score(y, yhat) -> dict:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat)}


def _future_index(train: pd.Series, h: int) -> pd.DatetimeIndex:
    start = train.index[-1] + pd.tseries.frequencies.to_offset(FREQ)
    return pd.date_range(start, periods=h, freq=FREQ)


# ---------------------------------------------------------------------------
# forecasters
# ---------------------------------------------------------------------------
class SeasonalNaive:
    """Repeat the value from ``season`` months ago."""

    def __init__(self, season: int = 12):
        self.season = season

    def fit(self, train: pd.Series):
        self.train_ = train
        return self

    def predict(self, h: int) -> pd.Series:
        hist = self.train_.values
        out = [hist[-self.season + (i % self.season)] for i in range(h)]
        return pd.Series(out, index=_future_index(self.train_, h), name="SeasonalNaive")


class HoltWinters:
    def __init__(self, trend="add", seasonal="add", seasonal_periods: int = 12):
        self.kw = dict(trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)

    def fit(self, train: pd.Series):
        self.train_ = train
        self.res_ = ExponentialSmoothing(train, **self.kw).fit()
        return self

    def predict(self, h: int) -> pd.Series:
        fc = self.res_.forecast(h)
        fc.index = _future_index(self.train_, h)
        return fc.rename("HoltWinters")


class Sarima:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train: pd.Series):
        self.train_ = train
        self.res_ = SARIMAX(
            train, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, h: int) -> pd.Series:
        fc = self.res_.get_forecast(h).predicted_mean
        fc.index = _future_index(self.train_, h)
        return fc.rename("SARIMA")

    def forecast_ci(self, h: int, alpha: float = 0.05) -> pd.DataFrame:
        f = self.res_.get_forecast(h)
        out = f.conf_int(alpha=alpha)
        out.columns = ["lower", "upper"]
        out["mean"] = f.predicted_mean.values
        out.index = _future_index(self.train_, h)
        return out


class LagRegressor:
    """Direct-ish recursive forecaster on lag/rolling/seasonal features."""

    def __init__(self, n_lags: int = 12, model=None):
        self.n_lags = n_lags
        self.model = model or RandomForestRegressor(
            n_estimators=400, max_depth=6, random_state=42, n_jobs=-1
        )

    def fit(self, train: pd.Series):
        self.train_ = train.copy()
        tf = supervised_time_features(train, self.n_lags).dropna()
        self.feat_cols_ = [c for c in tf.columns if c != "y"]
        self.model.fit(tf[self.feat_cols_], tf["y"])
        return self

    def predict(self, h: int) -> pd.Series:
        hist = self.train_.copy()
        preds = []
        idx = _future_index(self.train_, h)
        for ts in idx:
            hist_ext = pd.concat([hist, pd.Series([np.nan], index=[ts])])
            row = supervised_time_features(hist_ext, self.n_lags).iloc[[-1]]
            yhat = float(self.model.predict(row[self.feat_cols_])[0])
            preds.append(yhat)
            hist.loc[ts] = yhat
        return pd.Series(preds, index=idx, name="LagRegressor")


# ---------------------------------------------------------------------------
# rolling-origin backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    name: str
    horizon: int
    folds: int
    metrics: dict
    per_fold: pd.DataFrame = field(repr=False)
    predictions: pd.Series = field(repr=False)


def rolling_backtest(series: pd.Series, make_model, *, horizon: int = 3,
                     min_train: int = 24, step: int = 3, name: str = "model") -> BacktestResult:
    """Expanding-window backtest. ``make_model`` is a zero-arg factory."""
    rows, all_pred = [], []
    origins = range(min_train, len(series) - horizon + 1, step)
    for o in origins:
        train, test = series.iloc[:o], series.iloc[o:o + horizon]
        model = make_model().fit(train)
        pred = model.predict(horizon).reindex(test.index)
        all_pred.append(pred)
        rows.append({"origin": series.index[o - 1], **score(test.values, pred.values)})
    per_fold = pd.DataFrame(rows).set_index("origin")
    preds = pd.concat(all_pred)
    agg = {k: float(per_fold[k].mean()) for k in ("MAE", "RMSE", "MAPE")}
    return BacktestResult(name, horizon, len(per_fold), agg, per_fold, preds)
