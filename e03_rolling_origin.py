"""E3: rolling-origin evaluation with frozen hyper-parameters + calibration fits.

For every origin o:
  * all five models (set B, parameters frozen in E2 on data before the earliest origin) are trained on
    rows dated < o and evaluated on the 13-week window starting at o;
  * a CALIBRATION fit of the engine is trained on rows dated < (o - CAL_WEEKS weeks) and predicts the
    last CAL_WEEKS weeks of the training period. These out-of-sample residuals feed E11 (tier thresholds,
    large-error thresholds, T1 reference error) and E6 (global SHAP reference for T3).
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from common import config as C
from common.data import Preprocessor, load_dataset, split_at_origin
from common.io import Manifest, get_logger, load_table, save_fig, save_predictions, save_table
from common.metrics import METRIC_NAMES, all_metrics, series_scales
from common.models import make_model, needs_scaling
from common.plotting import new_fig


def calibration_split(df, origin, cal_weeks=C.CAL_WEEKS):
    """Training rows before the origin are split into (pre-calibration train, calibration window)."""
    train_all = df[df["Date"] < pd.Timestamp(origin)]
    dates = sorted(train_all["Date"].unique())
    cal_dates = dates[-cal_weeks:]
    cal = train_all[train_all["Date"].isin(cal_dates)].reset_index(drop=True)
    pre = train_all[~train_all["Date"].isin(cal_dates)].reset_index(drop=True)
    return pre, cal


def main():
    log = get_logger("e03_rolling_origin")
    with Manifest("e03_rolling_origin"):
        df = load_dataset()
        best = load_table("best_hyperparameters")
        best = best[best["feature_set"] == "B"].set_index("model")["params"].map(json.loads).to_dict()
        engine = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        rows, cal_rows = [], []
        for origin in C.ORIGINS:
            train, test = split_at_origin(df, origin)
            scales = series_scales(train)
            log.info("origin %s train=%d test=%d weeks=%d", origin, len(train), len(test), test["Date"].nunique())
            for name in C.MODELS:
                t0 = time.time()
                pre = Preprocessor(C.FEATURES_B, scale=needs_scaling(name))
                xtr = pre.fit_transform(train); xte = pre.transform(test)
                model = make_model(name, best.get(name, {})); model.fit(xtr, train[C.TARGET])
                pred = model.predict(xte)
                met = all_metrics(test, pred, scales)
                rows.append({"origin": origin, "model": name, "n_test": len(test), **met, "fit_seconds": time.time() - t0})
                save_predictions(pd.concat([test[C.KEYS + ["IsHoliday", C.TARGET]].reset_index(drop=True), pd.Series(pred, name="pred")], axis=1), f"{name}_B_{origin}")
                if name == engine:
                    save_predictions(pd.concat([train[C.KEYS + ["IsHoliday", C.TARGET]].reset_index(drop=True), pd.Series(model.predict(xtr), name="pred")], axis=1), f"{name}_B_{origin}_train")
                    joblib.dump({"model": model, "pre": pre, "name": name, "origin": origin}, C.RESULTS_DIR / "models" / f"{name}_B_{origin}.joblib", compress=3)
                log.info("  %s %s", name, {k: round(v, 4) for k, v in met.items()})
            # calibration fit of the engine
            t0 = time.time()
            pre_df, cal = calibration_split(df, origin)
            pre = Preprocessor(C.FEATURES_B, scale=needs_scaling(engine))
            xpre = pre.fit_transform(pre_df); xcal = pre.transform(cal)
            cmodel = make_model(engine, best.get(engine, {})); cmodel.fit(xpre, pre_df[C.TARGET])
            cpred = cmodel.predict(xcal)
            cmet = all_metrics(cal, cpred, series_scales(pre_df))
            cal_rows.append({"origin": origin, "cal_start": cal["Date"].min().date(), "cal_end": cal["Date"].max().date(),
                             "pre_rows": len(pre_df), "cal_rows": len(cal), **cmet, "fit_seconds": time.time() - t0})
            save_predictions(pd.concat([cal[C.KEYS + ["IsHoliday", C.TARGET]].reset_index(drop=True), pd.Series(cpred, name="pred")], axis=1), f"{engine}_B_{origin}_calibration")
            joblib.dump({"model": cmodel, "pre": pre, "name": engine, "origin": origin, "calibration": True}, C.RESULTS_DIR / "models" / f"{engine}_B_{origin}_calibration.joblib", compress=3)
            log.info("  calibration window %s..%s %s", cal["Date"].min().date(), cal["Date"].max().date(), {k: round(v, 3) for k, v in cmet.items()})
        res = pd.DataFrame(rows)
        base = load_table("baseline_metrics_all_origins")
        res_all = pd.concat([res, base.assign(n_test=np.nan, fit_seconds=np.nan)], ignore_index=True)
        save_table(res_all, "rolling_origin_metrics")
        save_table(pd.DataFrame(cal_rows), "calibration_window_metrics")
        summ = res_all.groupby("model")[METRIC_NAMES].agg(["mean", "std"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ["mean_rank_WMAE"] = res_all.groupby("origin")["WMAE"].rank().groupby(res_all["model"]).mean()
        summ = summ.sort_values("WMAE_mean").reset_index()
        save_table(summ, "rolling_origin_summary")
        fig, ax = new_fig(3.5, 2.6)
        sty = {"linear": ("#9db5cc", "-"), "ridge": ("#5a5a5a", ":"), "rf": ("#4f9d69", "-"), "xgb": ("#c0504d", "-"), "lgbm": ("#7f5fa0", "-"), "naive1": ("#e08a1e", "--"), "snaive": ("#8c564b", "--")}
        for mname in C.MODELS + ["naive1", "snaive"]:
            d = res_all[res_all["model"] == mname].sort_values("origin")
            ax.plot(d["origin"], d["WMAE"], marker="o", ms=3, lw=1.1, color=sty[mname][0], ls=sty[mname][1], label=C.MODEL_LABELS[mname])
        ax.set_ylabel("WMAE (13-week test window)"); ax.set_xlabel("Forecast origin"); ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper right")
        save_fig(fig, "rolling_origin_wmae")
        log.info("E3 done")


if __name__ == "__main__":
    main()
