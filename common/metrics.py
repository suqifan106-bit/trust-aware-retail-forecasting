"""Evaluation metrics.

All functions take numpy arrays or pandas Series. MASE is computed per
Store-Dept series with the in-sample (training period) one-step naive MAE as
the scale, then averaged over series (Hyndman & Koehler, 2006).
"""
import numpy as np
import pandas as pd

from . import config as C


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def wmae(y, p, is_holiday):
    w = np.where(np.asarray(is_holiday).astype(bool), C.HOLIDAY_WEIGHT, 1.0)
    return float(np.sum(w * np.abs(np.asarray(y) - np.asarray(p))) / np.sum(w))


def smape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    term = np.where(denom > 0, 2 * np.abs(y - p) / np.where(denom > 0, denom, 1), 0.0)
    return float(100 * np.mean(term))


def series_scales(train_df):
    """In-sample naive-1 MAE per Store-Dept series (the MASE denominator)."""
    d = train_df.sort_values(C.KEYS)
    diff = d.groupby(["Store", "Dept"])[C.TARGET].diff().abs()
    scale = diff.groupby([d["Store"], d["Dept"]]).mean().rename("scale")
    return scale.reset_index()


def mase(test_df, pred, scales):
    """Mean over series of (test MAE of series / in-sample naive MAE of series)."""
    d = test_df[["Store", "Dept", C.TARGET]].copy()
    d["abs_err"] = np.abs(d[C.TARGET].values - np.asarray(pred))
    per = d.groupby(["Store", "Dept"])["abs_err"].mean().rename("test_mae").reset_index()
    per = per.merge(scales, on=["Store", "Dept"], how="left")
    per = per[(per["scale"] > 0) & per["scale"].notna()]
    return float(np.mean(per["test_mae"] / per["scale"])) if len(per) else float("nan")


def mase_pooled(y, pred, scales):
    """Pooled variant: overall MAE divided by the mean in-sample naive MAE over series (less sensitive to tiny series)."""
    s = scales.loc[scales["scale"] > 0, "scale"].mean()
    return float(mae(y, pred) / s) if s and np.isfinite(s) else float("nan")


def all_metrics(test_df, pred, scales):
    y = test_df[C.TARGET].values
    return {
        "MAE": mae(y, pred),
        "RMSE": rmse(y, pred),
        "R2": r2(y, pred),
        "WMAE": wmae(y, pred, test_df["IsHoliday"].values),
        "SMAPE": smape(y, pred),
        "MASE": mase(test_df, pred, scales),
        "MASE_pooled": mase_pooled(y, pred, scales),
    }


METRIC_NAMES = ["MAE", "RMSE", "R2", "WMAE", "SMAPE", "MASE", "MASE_pooled"]
