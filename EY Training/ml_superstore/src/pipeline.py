"""End-to-end runner: raw CSV -> cleaned -> features -> forecast + models -> report.

Run from the project root::

    python -m src.pipeline                 # full run, writes outputs/
    python -m src.pipeline --quiet         # same, less console noise

Every stage reuses the same modules the notebooks use, so this script and
notebooks 01-05 can never drift apart.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import pandas as pd

from .config import MODEL_DIR, RAW_CSV, TABLE_DIR
from .data_loader import data_quality_report, load_clean
from .features import build_feature_frame, monthly_sales, rfm_table
from .forecasting import HoltWinters, Sarima, rolling_backtest, score
from .modeling import (
    build_xy,
    classifiers,
    eval_classification,
    eval_regression,
    best_f1_threshold,
    regressors,
)

REPORT_JSON = TABLE_DIR / "pipeline_report.json"
REPORT_MD = TABLE_DIR.parent / "REPORT.md"


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(f"[{datetime.now():%H:%M:%S}] {msg}")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_data(quiet: bool) -> pd.DataFrame:
    _log("load + clean", quiet)
    df = load_clean()
    dq = data_quality_report(df)
    dq.to_csv(TABLE_DIR / "pipeline_data_quality.csv", index=False)
    return df


def stage_features(df: pd.DataFrame, quiet: bool) -> dict:
    _log("feature engineering", quiet)
    feat = build_feature_frame(df)
    rfm = rfm_table(df)
    rfm.to_csv(TABLE_DIR / "pipeline_rfm.csv", index=False)
    engineered = [c for c in feat.columns if c not in df.columns]
    return {
        "n_rows": int(len(feat)),
        "n_engineered_features": len(engineered),
        "engineered_features": engineered,
        "n_customers": int(rfm.shape[0]),
    }


def stage_forecast(df: pd.DataFrame, quiet: bool) -> dict:
    _log("forecasting", quiet)
    y = monthly_sales(df)
    train, test = y.iloc[:-12], y.iloc[-12:]

    hw = HoltWinters(trend="add", seasonal="add").fit(train)
    hw_holdout = score(test.values, hw.predict(12).reindex(test.index).values)

    sar = Sarima(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit(train)
    sar_holdout = score(test.values, sar.predict(12).reindex(test.index).values)

    bt = rolling_backtest(y, lambda: HoltWinters(trend="add", seasonal="add"),
                          horizon=3, min_train=24, step=3, name="HoltWinters")

    hw_full = HoltWinters(trend="add", seasonal="add").fit(y)
    fc = hw_full.predict(12)
    ci = Sarima(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit(y).forecast_ci(12, alpha=0.20)
    future = pd.DataFrame({"holt_winters": fc, "sarima_mean": ci["mean"],
                           "lo80": ci["lower"], "hi80": ci["upper"]})
    future.to_csv(TABLE_DIR / "pipeline_forecast.csv")

    last_year = float(y[y.index.year == y.index.year.max()].sum())
    return {
        "holdout": {"HoltWinters": hw_holdout, "SARIMA": sar_holdout},
        "backtest_HoltWinters": bt.metrics,
        "next_12m_total": float(fc.sum()),
        "last_full_year_total": last_year,
        "yoy_growth_pct": (fc.sum() / last_year - 1) * 100,
    }


def stage_models(df: pd.DataFrame, quiet: bool) -> dict:
    _log("supervised models", quiet)
    d = build_xy(df, cutoff="2017-01-01")

    reg_scores = {}
    best_reg, best_reg_name, best_reg_mae = None, None, float("inf")
    for name, pipe in regressors().items():
        pipe.fit(d["X_train"], d["yreg_train"])
        s = eval_regression(d["yreg_test"], pipe.predict(d["X_test"]))
        reg_scores[name] = s
        if s["MAE"] < best_reg_mae:
            best_reg, best_reg_name, best_reg_mae = pipe, name, s["MAE"]

    clf_scores = {}
    best_clf, best_clf_name, best_clf_ap = None, None, -1.0
    best_proba = None
    for name, pipe in classifiers().items():
        pipe.fit(d["X_train"], d["yclf_train"])
        proba = pipe.predict_proba(d["X_test"])[:, 1]
        s = eval_classification(d["yclf_test"], proba)
        clf_scores[name] = s
        if s["PR_AUC"] > best_clf_ap:
            best_clf, best_clf_name, best_clf_ap, best_proba = pipe, name, s["PR_AUC"], proba

    t_best, f1_best = best_f1_threshold(d["yclf_test"], best_proba)

    joblib.dump(best_reg, MODEL_DIR / "pipeline_profit_regressor.joblib")
    joblib.dump(best_clf, MODEL_DIR / "pipeline_loss_classifier.joblib")

    return {
        "regression": reg_scores,
        "classification": clf_scores,
        "best_regressor": best_reg_name,
        "best_classifier": best_clf_name,
        "tuned_threshold": {"threshold": float(t_best), "F1": float(f1_best)},
        "n_train": int(d["X_train"].shape[0]),
        "n_test": int(d["X_test"].shape[0]),
    }


# ---------------------------------------------------------------------------
# orchestration + report
# ---------------------------------------------------------------------------
def run_pipeline(quiet: bool = False) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    df = stage_data(quiet)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": RAW_CSV.name,
        "n_order_lines": int(len(df)),
        "date_range": [str(df["Order Date"].min().date()), str(df["Order Date"].max().date())],
        "features": stage_features(df, quiet),
        "forecast": stage_forecast(df, quiet),
        "models": stage_models(df, quiet),
    }
    report["runtime_sec"] = round(time.time() - t0, 1)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report)
    _log(f"done in {report['runtime_sec']}s -> {REPORT_JSON.name}, {REPORT_MD.name}", quiet)
    return report


def _write_markdown(r: dict) -> None:
    f = r["forecast"]
    m = r["models"]
    reg = m["regression"][m["best_regressor"]]
    clf = m["classification"][m["best_classifier"]]
    lines = [
        "# Superstore ML — pipeline report",
        "",
        f"*Generated {r['generated_utc']} · runtime {r['runtime_sec']}s*",
        "",
        f"- **Source**: `{r['source_csv']}`",
        f"- **Order lines**: {r['n_order_lines']:,}  ({r['date_range'][0]} → {r['date_range'][1]})",
        f"- **Engineered features**: {r['features']['n_engineered_features']} "
        f"· customers {r['features']['n_customers']}",
        "",
        "## Forecast (monthly sales)",
        "",
        f"- Holt-Winters hold-out MAE **{f['holdout']['HoltWinters']['MAE']:,.0f}** "
        f"(MAPE {f['holdout']['HoltWinters']['MAPE']:.1f}%)",
        f"- SARIMA hold-out MAE {f['holdout']['SARIMA']['MAE']:,.0f}",
        f"- Holt-Winters rolling-backtest MAE {f['backtest_HoltWinters']['MAE']:,.0f}",
        f"- Next 12 months projected **{f['next_12m_total']:,.0f}** "
        f"vs last full year {f['last_full_year_total']:,.0f} "
        f"(**{f['yoy_growth_pct']:+.1f}%**)",
        "",
        "## Supervised models (train ≤2016 / test 2017)",
        "",
        f"- Best profit regressor: **{m['best_regressor']}** — "
        f"MAE {reg['MAE']:,.1f}, RMSE {reg['RMSE']:,.1f}, R² {reg['R2']:.3f}",
        f"- Best loss classifier: **{m['best_classifier']}** — "
        f"ROC-AUC {clf['ROC_AUC']:.3f}, PR-AUC {clf['PR_AUC']:.3f}, "
        f"F1 {clf['F1']:.3f} (tuned threshold {m['tuned_threshold']['threshold']:.2f} "
        f"→ F1 {m['tuned_threshold']['F1']:.3f})",
        "",
        "Artefacts: `outputs/tables/pipeline_*.csv`, `outputs/tables/pipeline_report.json`, "
        "`outputs/models/pipeline_*.joblib`.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Superstore ML pipeline end to end.")
    ap.add_argument("--quiet", action="store_true", help="suppress per-stage logging")
    args = ap.parse_args()
    run_pipeline(quiet=args.quiet)


if __name__ == "__main__":
    main()
