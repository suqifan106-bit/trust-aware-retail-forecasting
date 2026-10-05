"""E13: replicate the thesis setting (80,000-row sample, seed 42, original hyper-parameters) with the new code."""
import pandas as pd

from common import config as C
from common.data import Preprocessor, build_full_dataset, split_at_origin
from common.io import Manifest, get_logger, save_table
from common.metrics import all_metrics, series_scales
from common.models import THESIS_PARAMS, make_model


def main():
    log = get_logger("e13_thesis_replication")
    with Manifest("e13_thesis_replication"):
        df = build_full_dataset()
        if C.SMOKE_STORES:
            df = df[df["Store"].isin(C.SMOKE_STORES)]
        n = min(80000, len(df))
        sample = df.sample(n=n, random_state=C.SEED).sort_values("Date").reset_index(drop=True)
        # thesis split: 20 % of unique dates -> first test date
        dates = sorted(sample["Date"].unique()); split_date = dates[int(len(dates) * 0.8)]
        train = sample[sample["Date"] < split_date].reset_index(drop=True); test = sample[sample["Date"] >= split_date].reset_index(drop=True)
        scales = series_scales(train)
        log.info("sample=%d split=%s train=%d test=%d", n, pd.Timestamp(split_date).date(), len(train), len(test))
        rows = []
        for name, params, label in [("linear", {}, "LinearRegression (true OLS)"), ("ridge", THESIS_PARAMS["ridge"], "Ridge alpha=10 (thesis 'Linear Regression')"),
                                    ("rf", THESIS_PARAMS["rf"], "RF thesis params (5 trees)"), ("xgb", THESIS_PARAMS["xgb"], "XGB thesis params (45 rounds)")]:
            pre = Preprocessor(C.FEATURES_A, scale=True)  # thesis standardised all numeric inputs
            xtr = pre.fit_transform(train); xte = pre.transform(test)
            m = make_model(name, params); m.fit(xtr, train[C.TARGET])
            met = all_metrics(test, m.predict(xte), scales)
            rows.append({"configuration": label, "split_date": pd.Timestamp(split_date).date(), "train_rows": len(train), "test_rows": len(test), **met})
            log.info("%s %s", label, {k: round(v, 4) for k, v in met.items()})
        # naive baselines on the SAME thesis-style test window (direct comparison, not only scaled error)
        series_mean = train.groupby(["Store", "Dept"])[C.TARGET].mean().rename("smean").reset_index()
        t = test.merge(series_mean, on=["Store", "Dept"], how="left"); t["smean"] = t["smean"].fillna(train[C.TARGET].mean())
        for bname, col in [("naive-1 (last week, same window)", t["lag_1"].fillna(t["smean"])), ("seasonal naive (lag-52, same window)", t["lag_52"].fillna(t["lag_1"]).fillna(t["smean"]))]:
            met = all_metrics(test, col.values, scales)
            rows.append({"configuration": bname, "split_date": pd.Timestamp(split_date).date(), "train_rows": len(train), "test_rows": len(test), **met})
            log.info("%s %s", bname, {k: round(v, 4) for k, v in met.items()})
        res = pd.DataFrame(rows)
        res["thesis_reported_WMAE"] = [None, 14489.94, 5592.60, 7609.90, None, None]
        res["thesis_reported_R2"] = [None, 0.0925, 0.8298, 0.7272, None, None]
        res["note"] = "MASE here uses the per-series definition; the thesis value (0.2792 for RF) used a mis-specified pooled scale."
        save_table(res, "thesis_setting_replication")
        log.info("E13 done")


if __name__ == "__main__":
    main()
