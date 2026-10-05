"""E4: paired significance tests with effect sizes and block (Store-Dept) bootstrap CIs.

Uses the main-origin test predictions of all tuned models (feature set B) plus baselines.
Bootstrap is done on per-series aggregates so that every resample is exact and fast:
all six metrics are functions of per-series sums.
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

from common import config as C
from common.data import load_dataset, split_at_origin
from common.io import Manifest, get_logger, load_predictions, save_table
from common.metrics import METRIC_NAMES, all_metrics, series_scales


def cliffs_delta(a, b):
    """Cliff's delta between two samples via the Mann-Whitney U statistic."""
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); n = len(p)
    adj = np.empty(n); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def per_series_aggregates(pred_df, scales):
    d = pred_df.copy()
    e = d[C.TARGET] - d["pred"]; w = np.where(d["IsHoliday"].astype(bool), C.HOLIDAY_WEIGHT, 1.0)
    denom = (d[C.TARGET].abs() + d["pred"].abs()).replace(0, np.nan)
    d["ae"] = e.abs(); d["wae"] = w * e.abs(); d["w"] = w; d["se"] = e ** 2
    d["y"] = d[C.TARGET]; d["y2"] = d[C.TARGET] ** 2; d["sm"] = (2 * e.abs() / denom).fillna(0)
    g = d.groupby(["Store", "Dept"]).agg(n=("ae", "size"), ae=("ae", "sum"), wae=("wae", "sum"), w=("w", "sum"),
                                         se=("se", "sum"), y=("y", "sum"), y2=("y2", "sum"), sm=("sm", "sum")).reset_index()
    g = g.merge(scales, on=["Store", "Dept"], how="left")
    g["mase_s"] = (g["ae"] / g["n"]) / g["scale"]
    g.loc[~(g["scale"] > 0), "mase_s"] = np.nan
    return g


def metrics_from_aggregates(g):
    n = g["n"].sum(); y_mean = g["y"].sum() / n
    ss_tot = g["y2"].sum() - n * y_mean ** 2
    return {"MAE": g["ae"].sum() / n, "RMSE": np.sqrt(g["se"].sum() / n), "R2": 1 - g["se"].sum() / ss_tot,
            "WMAE": g["wae"].sum() / g["w"].sum(), "SMAPE": 100 * g["sm"].sum() / n, "MASE": g["mase_s"].mean(),
            "MASE_pooled": (g["ae"].sum() / n) / g.loc[g["scale"] > 0, "scale"].mean()}


def block_bootstrap(aggs, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(next(iter(aggs.values())))
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]

    def one(i):
        return {m: metrics_from_aggregates(g.iloc[i]) for m, g in aggs.items()}
    out = Parallel(n_jobs=min(C.N_JOBS, 8))(delayed(one)(i) for i in idx)
    rows = []
    for m in aggs:
        arr = pd.DataFrame([o[m] for o in out])
        for met in METRIC_NAMES:
            rows.append({"model": m, "metric": met, "boot_mean": arr[met].mean(), "boot_std": arr[met].std(),
                         "ci_lower_2.5": arr[met].quantile(0.025), "ci_upper_97.5": arr[met].quantile(0.975), "B": n_boot})
    return pd.DataFrame(rows)


def main():
    log = get_logger("e04_stats_bootstrap")
    with Manifest("e04_stats_bootstrap"):
        df = load_dataset()
        train, test = split_at_origin(df, C.MAIN_ORIGIN)
        scales = series_scales(train)
        preds = {m: load_predictions(f"{m}_B_main") for m in C.MODELS}
        base = load_predictions(f"baselines_{C.MAIN_ORIGIN}")
        for b in ["naive1", "snaive", "series_mean"]:
            preds[b] = base[C.KEYS + ["IsHoliday", C.TARGET]].assign(pred=base[b].values)
        best = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        errs = {m: (p[C.TARGET] - p["pred"]).abs().values for m, p in preds.items()}

        # paired tests: best vs every other
        rows = []
        for m in preds:
            if m == best:
                continue
            w = wilcoxon(errs[best], errs[m], alternative="two-sided", zero_method="wilcox", method="approx")
            z = getattr(w, "zstatistic", np.nan)
            rows.append({"test": "Wilcoxon signed-rank (two-sided)", "comparison": f"{best} vs {m}", "n": len(errs[best]),
                         "statistic_W": float(w.statistic), "z": float(z) if z is not None else np.nan, "p_value": float(w.pvalue),
                         "cliffs_delta": cliffs_delta(errs[best], errs[m]),
                         "median_abs_err_best": float(np.median(errs[best])), "median_abs_err_other": float(np.median(errs[m]))})
        tests = pd.DataFrame(rows)
        tests["p_holm"] = holm(tests["p_value"])
        fr = friedmanchisquare(*[errs[m] for m in C.MODELS])
        tests = pd.concat([tests, pd.DataFrame([{"test": "Friedman (5 tuned models)", "comparison": "all", "n": len(errs[best]),
                                                 "statistic_W": float(fr.statistic), "p_value": float(fr.pvalue)}])], ignore_index=True)
        save_table(tests, "paired_tests")
        log.info("\n%s", tests.to_string())

        # bootstrap
        aggs = {m: per_series_aggregates(p, scales) for m, p in preds.items()}
        block = block_bootstrap(aggs, C.N_BOOT, C.SEED)
        save_table(block, "bootstrap_ci_block")
        # row-level bootstrap (thesis style) for comparison, on the main metrics only
        rng = np.random.default_rng(C.SEED); n = len(test); rows = []
        for m, p in preds.items():
            y = p[C.TARGET].values; pr = p["pred"].values; hol = p["IsHoliday"].values
            vals = []
            for _ in range(C.N_BOOT):
                i = rng.integers(0, n, n)
                w = np.where(hol[i].astype(bool), C.HOLIDAY_WEIGHT, 1.0)
                vals.append(np.sum(w * np.abs(y[i] - pr[i])) / w.sum())
            vals = np.asarray(vals)
            rows.append({"model": m, "metric": "WMAE", "boot_mean": vals.mean(), "ci_lower_2.5": np.quantile(vals, 0.025), "ci_upper_97.5": np.quantile(vals, 0.975), "B": C.N_BOOT})
        save_table(pd.DataFrame(rows), "bootstrap_ci_row")

        # ---- paired block bootstrap of the WMAE DIFFERENCE (engine - competitor), series blocks and week blocks
        def paired_diff_ci(block_col, n_boot):
            rng = np.random.default_rng(C.SEED + 1)
            base_df = preds[best][C.KEYS + ["IsHoliday", C.TARGET]].copy()
            w = np.where(base_df["IsHoliday"].astype(bool), C.HOLIDAY_WEIGHT, 1.0)
            ae = {m: (preds[m][C.TARGET] - preds[m]["pred"]).abs().values for m in preds}
            blocks = base_df[block_col].values if block_col != "series" else (base_df["Store"].astype(str) + "_" + base_df["Dept"].astype(str)).values
            uniq, inv = np.unique(blocks, return_inverse=True)
            nb = len(uniq)
            # per-block sums of w*|e| and w
            sw = np.bincount(inv, weights=w, minlength=nb)
            swe = {m: np.bincount(inv, weights=w * ae[m], minlength=nb) for m in preds}
            out = []
            for m in preds:
                if m == best:
                    continue
                diffs = []
                for _ in range(n_boot):
                    idx = rng.integers(0, nb, nb)
                    diffs.append(swe[best][idx].sum() / sw[idx].sum() - swe[m][idx].sum() / sw[idx].sum())
                diffs = np.asarray(diffs)
                out.append({"block": block_col, "comparison": f"{best} - {m}", "n_blocks": nb, "diff_WMAE_point": float(swe[best].sum() / sw.sum() - swe[m].sum() / sw.sum()),
                            "boot_mean": diffs.mean(), "ci_lower_2.5": np.quantile(diffs, 0.025), "ci_upper_97.5": np.quantile(diffs, 0.975),
                            "excludes_zero": bool(np.quantile(diffs, 0.025) > 0 or np.quantile(diffs, 0.975) < 0), "B": n_boot})
            return out
        pd_rows = paired_diff_ci("series", C.N_BOOT) + paired_diff_ci("Date", C.N_BOOT)
        save_table(pd.DataFrame(pd_rows), "bootstrap_paired_difference")
        log.info("\n%s", pd.DataFrame(pd_rows).to_string())

        # ---- absolute-error quantiles (tail comparison engine vs baselines)
        qrows = []
        for m in preds:
            e = (preds[m][C.TARGET] - preds[m]["pred"]).abs().values
            qrows.append({"model": m, **{f"q{int(q*100)}": float(np.quantile(e, q)) for q in [0.5, 0.75, 0.9, 0.95, 0.99]}, "mean": float(e.mean()), "max": float(e.max())})
        save_table(pd.DataFrame(qrows), "abs_error_quantiles")
        log.info("E4 done")


if __name__ == "__main__":
    main()
