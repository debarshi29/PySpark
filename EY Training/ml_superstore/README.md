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
│   ├── features.py       feature engineering            (PR: feature engineering)
│   ├── forecasting.py    time-series models + backtest  (PR: forecasting)
│   └── pipeline.py       one-shot end-to-end runner      (PR: end-to-end)
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_forecasting.ipynb
│   └── 05_supervised_models.ipynb
└── outputs/
    ├── figures/          PNGs produced by the notebooks
    ├── tables/           CSV / JSON metrics and summaries
    └── models/           pickled estimators (git-ignored)
```

## Setup

```bash
python -m pip install -r "EY Training/ml_superstore/requirements.txt"
jupyter nbconvert --to notebook --execute --inplace \
    "EY Training/ml_superstore/notebooks/01_data_overview.ipynb"
```

## Roadmap (shipped as separate PRs)

1. **Scaffold + data overview + EDA** — data loading, quality checks, univariate
   and bivariate exploration, time trends.
2. **Feature engineering** — date parts, RFM, order-level rollups, target
   definitions, encoders; writes a processed parquet/CSV.
3. **Forecasting** — monthly sales time series, Holt-Winters / SARIMA / lag
   regression, rolling backtest, 12-month forecast.
4. **Supervised models** — profit regression and profitable-order classification,
   model comparison, permutation importance.
5. **End-to-end** — `src/pipeline.py` chaining every stage + summary report.
