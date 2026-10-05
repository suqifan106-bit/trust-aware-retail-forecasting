# Trust-Aware Explainable Retail Sales Forecasting

Code for the paper

> **A Trust-Aware Explainable Framework for Retail Sales Forecasting: Auditing Explanations and Routing Forecasts for Review**
> Qifan Su, Siti Nurulain Mohd Rum. *IEEE Access* (preparing to submit), 2026.

The pipeline links four things that are usually done separately in retail forecasting studies:

1. **Forecasting**: five models (OLS, ridge, random forest, XGBoost, LightGBM) and three naive baselines on the full Walmart store-sales data. Hyper-parameters are tuned once on data that precedes the earliest forecast origin and are then frozen.
2. **Explanation**: exact TreeSHAP for every test forecast, plus a calibration window inside each training period.
3. **Explanation audit**: three functionally grounded indicators.
   - *Permutation reliance*, with a random-feature control, grouped permutation and within-series permutation.
   - *Rank stability*, under test resampling, retraining seeds and forecast origins.
   - *Plausibility*, against a literature-derived set of retail demand drivers, with a sign check.
4. **Budgeted decision review**: review triggers calibrated on out-of-sample residuals.
   - T1 series instability, T2 holiday week, T3 explanation disagreement and T4 input anomaly.
   - Each trigger is evaluated as a detector of large-error forecasts at fixed review budgets.

Every number and figure in the paper is produced by the scripts in this repository.

## Repository layout

```
.
├── common/                     shared modules
│   ├── config.py               all experiment settings (origins, feature sets, search budgets, seeds)
│   ├── data.py                 merge, calendar/holiday flags, leakage-safe lag features, splits, preprocessing
│   ├── metrics.py              MAE, RMSE, R2, WMAE, SMAPE, MASE (per series) and pooled MASE
│   ├── models.py               model factory and random-search spaces
│   ├── shap_utils.py           SHAP helpers incl. row-chunked parallel TreeSHAP
│   ├── plotting.py             figure style
│   └── io.py                   result folders, logging, run manifests
├── e00_build_data.py           E0  dataset, feature dictionary, EDA
├── e01_baselines.py            E1  naive-1, seasonal naive, series mean
├── e02_benchmark_tuning.py     E2  date-block CV tuning before the first origin, main-origin benchmark, grouped ablation
├── e03_rolling_origin.py       E3  four rolling origins with frozen configurations + calibration-window fits
├── e04_stats_bootstrap.py      E4  paired tests, block bootstrap, paired difference bootstrap, error quantiles
├── e05_segment_errors.py       E5  error by holiday, store type/size, sales quintile, department
├── e06_shap_global.py          E6  global SHAP for all test rows and calibration windows, dependence plots
├── e07_shap_local.py           E7  local SHAP cases (10/50/90th percentile, largest error)
├── e08_fidelity.py             E8  permutation reliance, deletion curves, grouped/within-series permutation
├── e09_stability.py            E9  rank stability (resampling, seeds, origins)
├── e10_plausibility.py         E10 plausibility against a cited reference set + sign check
├── e11_decision_review.py      E11 calibrated tiers, triggers T1-T4, T3 increment, recall at fixed budgets
├── e12_ablation.py             E12 feature and routing ablation tables
├── e13_thesis_replication.py   E13 replication of the earlier sampled configuration + naive baselines
├── run_all.sh                  runs E0-E13 in order
├── smoke_test.sh               5-store, small-budget end-to-end check
├── requirements.txt            pinned versions used for the paper
└── data/raw/                   put the Walmart CSV files here (see data/raw/README.md)
```

## Installation

Python 3.12 was used. A fresh environment is recommended.

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n tarf python=3.12
pip install -r requirements.txt
```

## Data

The data are the Kaggle competition files from **Walmart Recruiting – Store Sales Forecasting**. They are not redistributed here. Download `train.csv`, `features.csv` and `stores.csv` and place them in `data/raw/`. See [`data/raw/README.md`](data/raw/README.md) for details. The full dataset has 421,570 rows in `train.csv`.

## Quick start

```bash
# 1. end-to-end check on 5 stores with tiny budgets (a few minutes on a multi-core machine)
N_JOBS=8 bash smoke_test.sh            # results in results/smoke/

# 2. full experiment
N_JOBS=24 bash run_all.sh main         # results in results/main/
```

Each script can also be re-run on its own once its inputs exist, for example `RUN_ID=main python e06_shap_global.py`. Dependencies: E1 and E2 need E0's modelling table; E13 needs only the raw CSV files; E3 needs E1 and E2; E4 needs E1 and E2; E5 and E6 need E3; E7 needs E6; E8 and E10 need E2; E9 needs E2 and E6; E11 needs E3 and E6; E12 needs E2 and E11.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `RUN_ID` | `main` | name of the output folder under `results/` |
| `RESULTS_DIR` | `./results` | root folder for outputs |
| `DATA_RAW` | `./data/raw` | location of the Kaggle CSV files |
| `N_JOBS` | `24` | maximum parallel threads/processes (models, bootstrap, SHAP workers capped at 16) |
| `SMOKE` | `0` | `1` = 5 stores and small budgets |

### Runtime

The full run took **72 minutes** on an AMD Ryzen 9 7950X with `N_JOBS=24` (64 GB RAM). The most expensive steps are E6 (TreeSHAP for about 155,000 forecasts, 37 min), E2 (tuning, 15 min) and E9 (ten retraining seeds, 11 min). The smoke test took about 7 minutes with `N_JOBS=8`.

## Outputs

`results/<RUN_ID>/` contains:

| Folder | Content |
|---|---|
| `tables/` | all result tables (CSV) and `best_model.json` |
| `figures/` | every figure as PDF and PNG |
| `predictions/` | modelling table (`full_features.parquet`) and test/calibration predictions (parquet) |
| `shap/` | SHAP matrices per origin and calibration window (`.npz`) |
| `models/` | fitted models (`.joblib`, large; not needed to reproduce tables) |
| `logs/`, `manifests/` | per-script logs and JSON manifests (timing, library versions, configuration) |

Main tables used in the paper:

| Paper item | Table file |
|---|---|
| Benchmark at the main origin | `model_metrics_main.csv`, `baseline_metrics.csv`, `best_hyperparameters.csv` |
| Grouped feature ablation | `ablation_features_main_origin.csv` |
| Rolling-origin results | `rolling_origin_metrics.csv`, `rolling_origin_summary.csv` |
| Paired bootstrap / tests | `bootstrap_paired_difference.csv`, `paired_tests.csv`, `abs_error_quantiles.csv` |
| Conditional errors | `error_by_segment.csv`, `error_by_dept.csv` |
| Global / local SHAP | `shap_global_importance.csv`, `shap_importance_by_origin.csv`, `local_shap_cases.csv` |
| Permutation reliance | `fidelity_permutation.csv`, `fidelity_grouped_permutation.csv`, `fidelity_deletion_auc.csv` |
| Stability / plausibility | `stability_jaccard.csv`, `plausibility_scores.csv`, `plausibility_sign_consistency.csv` |
| Decision review | `review_trigger_evaluation.csv`, `review_trigger_T3_increment.csv`, `review_recall_curves.csv`, `review_trigger_T1_sweep.csv`, `decision_tiers.csv` |
| Earlier configuration | `thesis_setting_replication.csv` |

## Protocol in brief

- **Task**: one-week-ahead forecasts of weekly sales for each store and department.
- **Origins**: four rolling origins (2011-10-07, 2012-01-06, 2012-04-13, 2012-07-13), each with a 13-week test window. The main origin is 2012-04-13.
- **Feature sets**:
  - Set A: 17 inputs with no sales history.
  - Set B: set A plus 9 sales-history features, 5 markdown-recorded indicators and 4 named holiday flags.
  - Ablation sets: A + history, and B − history.
- **Lag features**: lags are matched on exact dates within a store-department series. Rolling statistics use only earlier observations.
- **Tuning**:
  - Random search on rows dated before 2011-10-07, using 3 expanding date-block folds. No week is split across a fold boundary.
  - Preprocessing is fitted inside each training fold. The objective is WMAE.
  - The chosen configuration is frozen for all origins. The engine is the model with the lowest CV WMAE on set B.
- **Calibration window**: the last 13 training weeks of each origin, predicted by a model trained before them.
  - These predictions set the tier thresholds, the large-error threshold (3 × the tier's calibration MAE) and the T1 reference.
  - The SHAP ranking of the window is the T3 reference.
- **Metrics**: WMAE (holiday weight 5), MAE, RMSE, R², SMAPE, MASE per series, and pooled MASE.
- **Uncertainty**: paired block bootstrap of WMAE differences, blocking by series or by week.
- **Seeds**: global seed 42. The retraining experiment uses seeds 42–51.

## Headline results

These come from the paper's full run.

- **Engine**: random forest, selected by CV. Main-origin WMAE is 1,367, against 1,710 for a last-week naive forecast.
- **Tree ensembles**: within 7 % of each other across the four origins. In the holiday-season window, seasonal naive beats every learned model.
- **Source of the gain**: sales-history features account for all of it, reducing WMAE by 25–47 % within each tree model. The indicator and holiday flags change WMAE by less than 2 %.
- **Explanation audit**:
  - Permuting the top SHAP feature raises WMAE by 963 %, against 5 % for a random feature.
  - Within-series permutation of the history group raises WMAE by 33 %.
  - The top-10 features are identical across 10 retraining seeds. Their Jaccard similarity across origins is 0.77.
- **Decision review**:
  - At a 25 % review budget, trigger-based ranking recovers 54 % of large-error forecasts, against 27 % for ranking by predicted sales.
  - The explanation-disagreement trigger has 28 % precision, but adds little recall beyond T1.
  - The input-anomaly trigger does not help.

## Notes and limitations

- SHAP values describe the fitted model, not causal demand effects.
- Markdown variables are only provided from November 2011. Missing values are set to zero and paired with "recorded" indicators.
- The per-series MASE is unstable for low-volume series. Both MASE variants are reported.
- `e13_thesis_replication.py` reproduces an earlier sampled configuration. It uses an 80,000-row random sample, standardised set-A inputs and a 5-tree random forest. The script is kept for transparency.

## Citation

```bibtex
@article{su2026trustaware,
  author  = {Su, Qifan and Mohd Rum, Siti Nurulain},
  title   = {A Trust-Aware Explainable Framework for Retail Sales Forecasting: Auditing Explanations and Routing Forecasts for Review},
  journal = {IEEE Access},
  year    = {2026},
  note    = {under review}
}
```

## License

Released under the MIT License (see `LICENSE`). The Walmart data are subject to the Kaggle competition rules and are not covered by this license.
