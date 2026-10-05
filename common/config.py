"""Global configuration for the supplementary experiments (E0-E13).

Everything can be overridden through environment variables so that the same
code runs as a quick smoke test (SMOKE=1) or as the full experiment.
"""
import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.getenv("DATA_RAW", PIPELINE_DIR / "data" / "raw"))
RUN_ID = os.getenv("RUN_ID", "main")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PIPELINE_DIR / "results")) / RUN_ID

SMOKE = os.getenv("SMOKE", "0") == "1"
SEED = 42
# CPU budget: the local machine has 32 threads; the owner asked for <= 80 % load.
N_JOBS = int(os.getenv("N_JOBS", "24"))

TARGET = "Weekly_Sales"
KEYS = ["Store", "Dept", "Date"]

# Forecast origins (first test week) and test window length in weeks.
MAIN_ORIGIN = "2012-04-13"          # same cut-off as the thesis
ORIGINS = ["2011-10-07", "2012-01-06", "2012-04-13", "2012-07-13"]
TEST_WEEKS = 13
# Hyper-parameters are tuned ONCE on data strictly before the earliest origin and then frozen,
# so that every rolling origin (including the earliest) is a forward-looking evaluation.
TUNING_CUTOFF = ORIGINS[0]
# Calibration window (last CAL_WEEKS weeks of each origin's training period): out-of-sample residuals
# from a model trained before the window are used for tier thresholds, large-error thresholds,
# the T1 reference error and the global SHAP reference of trigger T3.
CAL_WEEKS = 13

# ---- feature sets --------------------------------------------------------
FEATURES_A = [  # thesis feature set (17 inputs)
    "Store", "Dept", "Type", "Size", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "IsHoliday", "Year", "Month", "Week",
]
LAG_FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_52",
    "roll_mean_4", "roll_std_4", "roll_mean_12", "lag_1_available",
]
EXTRA_FEATURES = [
    "MarkDown1_recorded", "MarkDown2_recorded", "MarkDown3_recorded",
    "MarkDown4_recorded", "MarkDown5_recorded",
    "is_superbowl", "is_laborday", "is_thanksgiving", "is_christmas",
]
FEATURES_B = FEATURES_A + LAG_FEATURES + EXTRA_FEATURES
# Grouped ablation sets: A_lag = history features only added; B_nolag = B without history features.
FEATURES_A_LAG = FEATURES_A + LAG_FEATURES
FEATURES_B_NOLAG = FEATURES_A + EXTRA_FEATURES
FEATURE_SETS = {"A": FEATURES_A, "B": FEATURES_B}
ABLATION_SETS = {"A_lag": FEATURES_A_LAG, "B_nolag": FEATURES_B_NOLAG}
# Feature groups for grouped permutation (E8)
FEATURE_GROUPS = {
    "history": LAG_FEATURES,
    "calendar": ["Year", "Month", "Week", "IsHoliday", "is_superbowl", "is_laborday", "is_thanksgiving", "is_christmas"],
    "promotion": [f"MarkDown{i}" for i in range(1, 6)] + [f"MarkDown{i}_recorded" for i in range(1, 6)],
    "store": ["Store", "Dept", "Type_A", "Type_B", "Type_C", "Size"],
    "macro": ["Temperature", "Fuel_Price", "CPI", "Unemployment"],
}

MODELS = ["linear", "ridge", "rf", "xgb", "lgbm"]
MODEL_LABELS = {
    "linear": "Linear Regression", "ridge": "Ridge", "rf": "Random Forest",
    "xgb": "XGBoost", "lgbm": "LightGBM",
    "naive1": "Naive (last week)", "snaive": "Seasonal naive (lag-52)",
    "series_mean": "Series mean",
}

# ---- experiment sizes (full run) -----------------------------------------
N_ITER = {"ridge": 20, "rf": 15, "xgb": 25, "lgbm": 25}
CV_SPLITS = 3
N_BOOT = 1000
N_PERM_REPEATS = 20
PERM_KS = [1, 3, 5, 10]
N_SEEDS = 10
N_STAB_BOOT = 30
MAX_SHAP_ROWS = 40000        # explain every test row at every origin (<= 38,768)
CAL_SHAP_ROWS = 8000         # calibration-window SHAP is only a ranking reference -> capped
SHAP_BACKGROUND = 1000
REVIEW_BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
T1_MULTIPLIERS = [1.5, 2.0, 3.0, 4.0]
FIDELITY_SHAP_ROWS = 5000
SMOKE_STORES = None

if SMOKE:
    N_JOBS = min(N_JOBS, int(os.getenv("N_JOBS", "2")))
    N_ITER = {"ridge": 3, "rf": 2, "xgb": 2, "lgbm": 2}
    CV_SPLITS = 2
    N_BOOT = 30
    N_PERM_REPEATS = 3
    N_SEEDS = 3
    N_STAB_BOOT = 5
    MAX_SHAP_ROWS = 3000
    CAL_SHAP_ROWS = 1000
    SHAP_BACKGROUND = 200
    FIDELITY_SHAP_ROWS = 400
    SMOKE_STORES = [1, 2, 3, 4, 5]

# Walmart competition holiday weeks (Friday dates of the sales weeks).
HOLIDAY_WEEKS = {
    "is_superbowl": ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "is_laborday": ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "is_thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "is_christmas": ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}
HOLIDAY_WEIGHT = 5
