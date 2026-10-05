"""E6: global SHAP explanation of the best model at every origin + dependence plots.

Saves the SHAP matrix (and explained row keys) per origin for reuse by E7, E9 and E11.
"""
import json

import joblib
import numpy as np
import pandas as pd

from common import config as C
from common.data import load_dataset, split_at_origin
from common.io import Manifest, get_logger, save_fig, save_table
from common.plotting import new_fig
from common.shap_utils import global_importance, make_explainer, shap_matrix, shap_matrix_parallel


def explain_origin(best, origin, df, log):
    bundle = joblib.load(C.RESULTS_DIR / "models" / f"{best}_B_{origin}.joblib")
    model, pre = bundle["model"], bundle["pre"]
    train, test = split_at_origin(df, origin)
    if len(test) > C.MAX_SHAP_ROWS:
        test = test.sample(C.MAX_SHAP_ROWS, random_state=C.SEED).sort_values(C.KEYS).reset_index(drop=True)
    x = pre.transform(test)
    bg = pre.transform(train.sample(min(C.SHAP_BACKGROUND, len(train)), random_state=C.SEED))
    expl = make_explainer(best, model, bg)
    sv = shap_matrix_parallel(best, model, x, bg)
    base_value = float(np.ravel(expl.expected_value)[0]) if hasattr(expl, "expected_value") else float("nan")
    np.savez_compressed(C.RESULTS_DIR / "shap" / f"shap_{best}_B_{origin}.npz", shap_values=sv, base_value=base_value,
                        columns=np.array(x.columns), store=test["Store"].values, dept=test["Dept"].values,
                        date=test["Date"].astype(str).values, x=x.values, y=test[C.TARGET].values, pred=model.predict(x))
    imp = global_importance(sv, x.columns).assign(origin=origin)
    log.info("origin %s explained %d rows; top5=%s", origin, len(x), list(imp["feature"].head(5)))
    return imp, sv, x, test


def main():
    log = get_logger("e06_shap_global")
    with Manifest("e06_shap_global"):
        import shap
        best = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        df = load_dataset()
        imps = []
        for o in C.ORIGINS:
            imp, sv, x, test = explain_origin(best, o, df, log)
            imps.append(imp)
            if o == C.MAIN_ORIGIN:
                main_sv, main_x = sv, x
        # calibration-window SHAP reference (training-period information only) for trigger T3 and for reporting
        cal_imps = []
        for o in C.ORIGINS:
            cb = joblib.load(C.RESULTS_DIR / "models" / f"{best}_B_{o}_calibration.joblib")
            cal_pred = pd.read_parquet(C.RESULTS_DIR / "predictions" / f"{best}_B_{o}_calibration.parquet")
            cal_rows = df.merge(cal_pred[C.KEYS], on=C.KEYS, how="inner")
            if len(cal_rows) > C.CAL_SHAP_ROWS:
                cal_rows = cal_rows.sample(C.CAL_SHAP_ROWS, random_state=C.SEED)
            xc = cb["pre"].transform(cal_rows)
            svc = shap_matrix_parallel(best, cb["model"], xc)
            cal_imps.append(global_importance(svc, xc.columns).assign(origin=o))
            np.savez_compressed(C.RESULTS_DIR / "shap" / f"shap_{best}_B_{o}_calibration.npz", shap_values=svc, columns=np.array(xc.columns),
                                store=cal_rows["Store"].values, dept=cal_rows["Dept"].values, date=cal_rows["Date"].astype(str).values)
            log.info("origin %s calibration reference top5=%s", o, list(cal_imps[-1]["feature"].head(5)))
        save_table(pd.concat(cal_imps, ignore_index=True), "shap_importance_calibration_by_origin")
        by_origin = pd.concat(imps, ignore_index=True)
        save_table(by_origin, "shap_importance_by_origin")
        piv = by_origin.pivot(index="feature", columns="origin", values="mean_abs_shap")
        summ = pd.DataFrame({"mean_abs_shap_mean": piv.mean(axis=1), "mean_abs_shap_std": piv.std(axis=1),
                             "mean_abs_shap_main": piv[C.MAIN_ORIGIN]}).sort_values("mean_abs_shap_main", ascending=False).reset_index()
        summ["rank_main"] = np.arange(1, len(summ) + 1)
        save_table(summ, "shap_global_importance")
        log.info("\n%s", summ.head(15).to_string())

        # bar plot with error bars
        top = summ.head(15)
        fig, ax = new_fig(3.5, 3.2)
        ax.barh(top["feature"][::-1], top["mean_abs_shap_mean"][::-1], xerr=top["mean_abs_shap_std"][::-1], color="#1f4e79", error_kw={"lw": 0.7})
        ax.set_xlabel("mean |SHAP| (mean ± sd over 4 origins)")
        save_fig(fig, "shap_bar")
        # beeswarm (main origin)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.5, 4))
        shap.summary_plot(main_sv, main_x, max_display=15, show=False, plot_size=None)
        fig = plt.gcf(); save_fig(fig, "shap_summary")
        # dependence plots for top features (main origin)
        for feat in [f for f in ["Dept", "Size", "lag_1", "lag_52", "Week", "roll_mean_4"] if f in main_x.columns][:6]:
            j = list(main_x.columns).index(feat)
            fig, ax = new_fig(3.3, 2.4)
            if feat == "Dept":
                d = pd.DataFrame({"Dept": main_x[feat].astype(int), "shap": main_sv[:, j]})
                med = d.groupby("Dept")["shap"].median().sort_values()
                ax.bar(range(len(med)), med.values, color=np.where(med.values >= 0, "#1f4e79", "#c0504d"), width=1.0)
                ax.set_xlabel("Department (sorted by median SHAP)"); ax.set_ylabel("SHAP value")
            else:
                ax.scatter(main_x[feat], main_sv[:, j], s=3, alpha=0.3, color="#1f4e79")
                ax.set_xlabel(feat); ax.set_ylabel("SHAP value")
            save_fig(fig, f"shap_dependence_{feat}")
        log.info("E6 done")


if __name__ == "__main__":
    main()
