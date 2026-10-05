"""E1: naive baselines on the main split (and all rolling origins)."""
import numpy as np
import pandas as pd

from common import config as C
from common.data import load_dataset, split_at_origin
from common.io import Manifest, get_logger, save_predictions, save_table
from common.metrics import METRIC_NAMES, all_metrics, series_scales


def baseline_predictions(train, test):
    series_mean = train.groupby(["Store", "Dept"])[C.TARGET].mean().rename("series_mean").reset_index()
    t = test.merge(series_mean, on=["Store", "Dept"], how="left")
    gmean = train[C.TARGET].mean()
    t["series_mean"] = t["series_mean"].fillna(gmean)
    t["naive1"] = t["lag_1"].fillna(t["series_mean"])
    t["snaive"] = t["lag_52"].fillna(t["naive1"])
    return t


def run_origin(origin, df, log):
    train, test = split_at_origin(df, origin)
    scales = series_scales(train)
    t = baseline_predictions(train, test)
    rows = []
    for b in ["naive1", "snaive", "series_mean"]:
        met = all_metrics(test, t[b].values, scales)
        rows.append({"origin": origin, "model": b, **met})
        log.info("%s %s %s", origin, b, {k: round(v, 3) for k, v in met.items()})
    pred = t[C.KEYS + ["IsHoliday", C.TARGET, "naive1", "snaive", "series_mean"]]
    save_predictions(pred, f"baselines_{origin}")
    return rows


def main():
    log = get_logger("e01_baselines")
    with Manifest("e01_baselines"):
        df = load_dataset()
        rows = []
        for o in C.ORIGINS:
            rows += run_origin(o, df, log)
        res = pd.DataFrame(rows)
        save_table(res, "baseline_metrics_all_origins")
        save_table(res[res["origin"] == C.MAIN_ORIGIN].drop(columns="origin"), "baseline_metrics")
        log.info("E1 done")


if __name__ == "__main__":
    main()
