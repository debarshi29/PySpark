# Superstore ML — pipeline report

*Generated 2026-09-01T14:42:53+00:00 · runtime 13.7s*

- **Source**: `Sample - Superstore.csv`
- **Order lines**: 9,994  (2014-01-03 → 2017-12-30)
- **Engineered features**: 26 · customers 793

## Forecast (monthly sales)

- Holt-Winters hold-out MAE **11,456** (MAPE 22.6%)
- SARIMA hold-out MAE 13,411
- Holt-Winters rolling-backtest MAE 12,314
- Next 12 months projected **914,903** vs last full year 733,215 (**+24.8%**)

## Supervised models (train ≤2016 / test 2017)

- Best profit regressor: **RandomForest** — MAE 21.4, RMSE 139.2, R² 0.668
- Best loss classifier: **RandomForest** — ROC-AUC 0.988, PR-AUC 0.956, F1 0.865 (tuned threshold 0.50 → F1 0.865)

Artefacts: `outputs/tables/pipeline_*.csv`, `outputs/tables/pipeline_report.json`, `outputs/models/pipeline_*.joblib`.
