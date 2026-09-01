# Superstore — end-to-end ML

An end-to-end machine-learning walk-through on the **Sample - Superstore** retail
dataset (`../data/Sample - Superstore.csv`, 9,994 order lines, 2014-2017, US).

## Layout

```
ml_superstore/
├── src/                  reusable code
│   ├── config.py         paths + constants
│   ├── data_loader.py    load / clean / data-quality report
│   ├── plotting.py       figure-saving helpers
│   ├── features.py       feature engineering (date parts, RFM, leakage-safe history)
│   ├── forecasting.py    time-series models + rolling backtest
│   ├── modeling.py       preprocessing, model zoos, time split, evaluation
│   └── pipeline.py       one-shot end-to-end runner
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_forecasting.ipynb
│   ├── 05_supervised_models.ipynb
│   └── 06_end_to_end.ipynb
└── outputs/
    ├── REPORT.md          machine-written run summary
    ├── figures/           PNGs produced by the notebooks
    ├── tables/            CSV / JSON metrics and summaries
    └── models/            pickled estimators (git-ignored)
```

## Setup

```bash
python -m pip install -r "EY Training/ml_superstore/requirements.txt"
```

Run the whole thing (from `EY Training/ml_superstore/`):

```bash
python -m src.pipeline            # load -> features -> forecast -> models -> REPORT.md
```

Re-execute the notebooks in order:

```bash
cd "EY Training/ml_superstore/notebooks"
for nb in 0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "$nb"
done
```

## Shipped as a stacked 5-PR series

| # | PR | contents |
|---|----|----------|
| 1 | scaffold + data overview + EDA | `data_loader`, `plotting`, notebooks 01-02 |
| 2 | feature engineering | `features.py`, notebook 03 |
| 3 | forecasting | `forecasting.py`, notebook 04 |
| 4 | supervised models | `modeling.py`, notebook 05 |
| 5 | end-to-end | `pipeline.py`, notebook 06, this summary |

## Headline results

*Full machine-written run in [`outputs/REPORT.md`](outputs/REPORT.md).*

**Forecast — monthly sales** (48 months, hold out last 12)

| model | hold-out MAE | rolling-backtest MAE |
|-------|-------------:|---------------------:|
| Holt-Winters (add/add) | **11,456** | 12,314 |
| SARIMA(1,1,1)(1,1,0,12) | 13,411 | **11,612** |
| Seasonal-naive baseline | 15,468 | 13,994 |

Next 12 months projected ≈ **+25%** vs the last full year; Office Supplies grows
fastest, Technology roughly flat.

**Supervised — line level, train ≤2016 / test 2017**

| task | best model | score |
|------|-----------|-------|
| `Profit` regression | RandomForest | MAE **$21.4**, R² 0.67 (GradBoost R² **0.78**) |
| `is_loss` classification | RandomForest | ROC-AUC **0.99**, PR-AUC **0.96**, F1 0.87 |

`Discount` / `discount_bucket` dominate both models, then `cust_prior_loss_rate`
and `Sub-Category` — discounting past ~0.3 is what turns lines loss-making.
