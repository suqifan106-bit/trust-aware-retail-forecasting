"""E11: decision tiers and review triggers, calibrated on out-of-sample residuals, compared at equal budgets.

Design:
  * Tier thresholds, tier reference MAE (large-error threshold, T1 reference) come from the CALIBRATION window
    (out-of-sample predictions of a model trained before the window), not from in-sample training residuals.
  * T3 uses the global SHAP top-5 of the calibration window of the SAME origin (training-period information only).
  * All triggers are evaluated on the same rows (every test row has SHAP values).
  * Incremental analysis: T1+T2+T4 versus T1+T2+T3+T4; recall-review-rate curves at equal review budgets for
    several routing strategies; threshold sweep for T1 multiplier and T4 variants.
  * Per-origin tables so that main-origin and pooled numbers are never mixed.
Triggers:
  T1 series instability : naive-1 MAE of the series in its last 8 training weeks > m x calibration MAE of its tier (m=2 default)
  T2 holiday week
  T3 explanation disagreement : locally most important |SHAP| feature not in the calibration global top-5
  T4 input anomaly : lag_1 missing, or a RECORDED markdown above the training 99th percentile of recorded markdowns
"""
import json

import numpy as np
import pandas as pd

from common import config as C
from common.data import load_dataset, split_at_origin
from common.io import Manifest, get_logger, load_predictions, load_table, save_fig, save_table
from common.metrics import wmae
from common.plotting import new_fig
from common.shap_utils import top_k

MD = [f"MarkDown{i}" for i in range(1, 6)]
MDR = [f"MarkDown{i}_recorded" for i in range(1, 6)]


def detector_stats(sub, flag, label, extra=None):
    f, u = sub[flag == 1], sub[flag == 0]
    le = sub["large_error"].sum()
    row = {"rule": label, "rows_evaluated": len(sub), "flag_rate": float(flag.mean()),
           "MAE_flagged": f["ae"].mean() if len(f) else np.nan, "MAE_unflagged": u["ae"].mean() if len(u) else np.nan,
           "MAE_ratio": (f["ae"].mean() / u["ae"].mean()) if len(f) and len(u) else np.nan,
           "WMAE_flagged": wmae(f[C.TARGET], f["pred"], f["IsHoliday"]) if len(f) else np.nan,
           "WMAE_unflagged": wmae(u[C.TARGET], u["pred"], u["IsHoliday"]) if len(u) else np.nan,
           "large_error_recall": f["large_error"].sum() / le if le else np.nan,
           "large_error_precision": f["large_error"].mean() if len(f) else np.nan,
           "large_error_base_rate": sub["large_error"].mean()}
    row.update(extra or {})
    return row


def recall_at_budgets(sub, score, label):
    """Rank rows by score (desc), review the top b fraction, report recall of large-error rows and MAE captured."""
    order = np.argsort(-score, kind="stable")
    le = sub["large_error"].values[order]; ae = sub["ae"].values[order]
    rows = []
    for b in C.REVIEW_BUDGETS:
        k = int(round(b * len(sub)))
        rows.append({"strategy": label, "review_rate": b, "rows_reviewed": k, "large_error_recall": le[:k].sum() / max(le.sum(), 1),
                     "large_error_precision": le[:k].mean() if k else np.nan, "share_of_abs_error_captured": ae[:k].sum() / ae.sum()})
    return rows


def build_rows(origin, df, engine, t1_mult, log):
    train, test = split_at_origin(df, origin)
    cal = load_predictions(f"{engine}_B_{origin}_calibration"); te = load_predictions(f"{engine}_B_{origin}")
    lo, hi = np.percentile(cal["pred"], [25, 75])
    tier = lambda p: np.where(p < lo, "low", np.where(p > hi, "high", "normal"))
    cal["tier"] = tier(cal["pred"]); te["tier"] = tier(te["pred"])
    cal["ae"] = (cal[C.TARGET] - cal["pred"]).abs(); te["ae"] = (te[C.TARGET] - te["pred"]).abs()
    tier_mae = cal.groupby("tier")["ae"].mean()
    te["tier_mae_cal"] = te["tier"].map(tier_mae)
    te["large_error"] = (te["ae"] > 3 * te["tier_mae_cal"]).astype(int)
    # T1
    last8 = train.sort_values(C.KEYS).groupby(["Store", "Dept"]).tail(8)
    inst = last8.groupby(["Store", "Dept"])[C.TARGET].apply(lambda s: s.diff().abs().mean()).rename("recent_naive_mae").reset_index()
    te = te.merge(inst, on=["Store", "Dept"], how="left")
    te["T1_ratio"] = (te["recent_naive_mae"] / te["tier_mae_cal"]).fillna(0)
    te["T1"] = (te["T1_ratio"] > t1_mult).astype(int)
    # T2
    te["T2"] = te["IsHoliday"].astype(int)
    # T3 with calibration reference of the same origin
    cal_imp = load_table("shap_importance_calibration_by_origin"); cal_imp = cal_imp[cal_imp["origin"] == origin].sort_values("mean_abs_shap", ascending=False)
    ref5 = set(top_k(cal_imp, 5))
    z = np.load(C.RESULTS_DIR / "shap" / f"shap_{engine}_B_{origin}.npz", allow_pickle=True)
    cols = list(z["columns"]); sv = z["shap_values"]
    top1 = [cols[int(np.argmax(np.abs(r)))] for r in sv]
    top3 = [set(cols[j] for j in np.argsort(np.abs(r))[::-1][:3]) for r in sv]
    r3 = pd.DataFrame({"Store": z["store"], "Dept": z["dept"], "Date": pd.to_datetime(z["date"]),
                       "T3": [int(t not in ref5) for t in top1], "T3_score": [1 - len(t & ref5) / 3 for t in top3]})
    te = te.merge(r3, on=C.KEYS, how="left")
    te["T3_available"] = te["T3"].notna().astype(int); te["T3"] = te["T3"].fillna(0).astype(int); te["T3_score"] = te["T3_score"].fillna(0)
    # T4
    feat = test[C.KEYS + ["lag_1"] + MD + MDR]
    te = te.merge(feat, on=C.KEYS, how="left")
    rec_train = train[MD].where(train[MDR].values == 1)
    p99 = rec_train.quantile(0.99).fillna(np.inf)
    md_anom = ((te[MD] > p99.values) & (te[MDR].values == 1)).any(axis=1)
    te["T4_lag_missing"] = te["lag_1"].isna().astype(int)
    te["T4_md_anom"] = md_anom.astype(int)
    te["T4"] = ((te["T4_lag_missing"] == 1) | (te["T4_md_anom"] == 1)).astype(int)
    te["T4_run01"] = ((te["lag_1"].isna()) | (te[MD] > train[MD].quantile(0.99).values).any(axis=1)).astype(int)  # earlier definition that ignores when markdown recording starts (reported for reference)
    te["any_no_T3"] = te[["T1", "T2", "T4"]].max(axis=1)
    te["any"] = te[["T1", "T2", "T3", "T4"]].max(axis=1)
    te["origin"] = origin
    te["tier_lo"], te["tier_hi"] = lo, hi
    log.info("origin %s: tiers lo=%.0f hi=%.0f cal tier MAE=%s T3 ref=%s", origin, lo, hi, {k: round(v) for k, v in tier_mae.items()}, sorted(ref5))
    return te


def main():
    log = get_logger("e11_decision_review")
    with Manifest("e11_decision_review"):
        engine = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        df = load_dataset()
        parts = [build_rows(o, df, engine, 2.0, log) for o in C.ORIGINS]
        d = pd.concat(parts, ignore_index=True)
        d.drop(columns=MD + MDR).to_parquet(C.RESULTS_DIR / "predictions" / "decision_review_rows.parquet", index=False)

        # tiers per origin
        tier_rows = []
        for (o, t), g in d.groupby(["origin", "tier"]):
            tier_rows.append({"origin": o, "tier": t, "n": len(g), "actual_mean": g[C.TARGET].mean(), "actual_median": g[C.TARGET].median(), "pred_mean": g["pred"].mean(),
                              "MAE": g["ae"].mean(), "WMAE": wmae(g[C.TARGET], g["pred"], g["IsHoliday"]), "large_error_rate": g["large_error"].mean(),
                              "threshold_low": g["tier_lo"].iloc[0], "threshold_high": g["tier_hi"].iloc[0], "tier_mae_calibration": g["tier_mae_cal"].iloc[0]})
        save_table(pd.DataFrame(tier_rows), "decision_tiers")

        # binary trigger evaluation: main origin, pooled, and per origin — same rows for every rule
        def eval_block(sub, scope):
            out = []
            for rule in ["T1", "T2", "T3", "T4", "T4_lag_missing", "T4_md_anom", "any_no_T3", "any"]:
                out.append(detector_stats(sub, sub[rule], rule, {"scope": scope}))
            out.append(detector_stats(sub, (sub["tier"] == "high").astype(int), "tier_only_high", {"scope": scope}))
            out.append(detector_stats(sub, sub["T4_run01"], "T4_run01_definition", {"scope": scope}))
            return out
        ev = eval_block(d[d["origin"] == C.MAIN_ORIGIN], "main_origin") + eval_block(d, "pooled_4_origins")
        for o in C.ORIGINS:
            ev += eval_block(d[d["origin"] == o], f"origin_{o}")
        ev = pd.DataFrame(ev); save_table(ev, "review_trigger_evaluation")
        log.info("\n%s", ev[ev["scope"].isin(["main_origin", "pooled_4_origins"])][["scope", "rule", "flag_rate", "large_error_recall", "large_error_precision", "MAE_ratio"]].to_string())

        # incremental value of T3 (same rows, binary rules)
        inc = []
        for scope, sub in [("main_origin", d[d["origin"] == C.MAIN_ORIGIN]), ("pooled_4_origins", d)]:
            a = detector_stats(sub, sub["any_no_T3"], "T1+T2+T4"); b = detector_stats(sub, sub["any"], "T1+T2+T3+T4")
            only3 = sub[(sub["T3"] == 1) & (sub["any_no_T3"] == 0)]
            inc.append({"scope": scope, "flag_rate_without_T3": a["flag_rate"], "flag_rate_with_T3": b["flag_rate"], "recall_without_T3": a["large_error_recall"], "recall_with_T3": b["large_error_recall"],
                        "precision_without_T3": a["large_error_precision"], "precision_with_T3": b["large_error_precision"],
                        "rows_added_by_T3": len(only3), "large_errors_added_by_T3": int(only3["large_error"].sum()), "precision_of_rows_added_by_T3": only3["large_error"].mean() if len(only3) else np.nan})
        save_table(pd.DataFrame(inc), "review_trigger_T3_increment")
        log.info("\n%s", pd.DataFrame(inc).to_string())

        # recall-review-rate curves at equal budgets (continuous scores)
        curves = []
        for scope, sub in [("main_origin", d[d["origin"] == C.MAIN_ORIGIN].reset_index(drop=True)), ("pooled_4_origins", d.reset_index(drop=True))]:
            rng = np.random.default_rng(C.SEED)
            strategies = {
                "random": rng.random(len(sub)),
                "tier_only (predicted sales)": sub["pred"].values,
                "T1 only (instability ratio)": sub["T1_ratio"].values,
                "T1+T2+T4 score": sub["T1_ratio"].values + 2.0 * sub["T2"].values + 2.0 * sub["T4"].values,
                "T1+T2+T3+T4 score": sub["T1_ratio"].values + 2.0 * sub["T2"].values + 2.0 * sub["T4"].values + 2.0 * sub["T3_score"].values,
                "T3 only (disagreement score)": sub["T3_score"].values + 1e-6 * rng.random(len(sub)),
            }
            for lab, sc in strategies.items():
                for r in recall_at_budgets(sub, np.asarray(sc, float), lab):
                    r["scope"] = scope; curves.append(r)
        curves = pd.DataFrame(curves); save_table(curves, "review_recall_curves")
        fig, ax = new_fig(3.5, 2.6)
        for lab, c in [("random", "#aaaaaa"), ("tier_only (predicted sales)", "#e08a1e"), ("T1 only (instability ratio)", "#4f9d69"), ("T1+T2+T4 score", "#1f4e79"), ("T1+T2+T3+T4 score", "#c0504d"), ("T3 only (disagreement score)", "#7f5fa0")]:
            g = curves[(curves["scope"] == "pooled_4_origins") & (curves["strategy"] == lab)]
            ax.plot(g["review_rate"], g["large_error_recall"], marker="o", ms=3, lw=1.1, color=c, label=lab)
        ax.set_xlabel("Review rate (share of forecasts reviewed)"); ax.set_ylabel("Recall of large-error forecasts"); ax.legend(frameon=False, fontsize=6)
        save_fig(fig, "review_recall_curves")

        # T1 multiplier sweep (binary rules, pooled and main)
        sweep = []
        for m in C.T1_MULTIPLIERS:
            for scope, sub in [("main_origin", d[d["origin"] == C.MAIN_ORIGIN]), ("pooled_4_origins", d)]:
                t1 = (sub["T1_ratio"] > m).astype(int)
                anyf = np.maximum.reduce([t1.values, sub["T2"].values, sub["T3"].values, sub["T4"].values])
                sweep.append({**detector_stats(sub, t1, "T1", {"scope": scope, "t1_multiplier": m})})
                sweep.append({**detector_stats(sub, pd.Series(anyf, index=sub.index), "any", {"scope": scope, "t1_multiplier": m})})
        save_table(pd.DataFrame(sweep), "review_trigger_T1_sweep")

        by_origin = ev[(ev["rule"] == "any") & ev["scope"].str.startswith("origin_")][["scope", "flag_rate", "large_error_recall", "large_error_precision"]]
        save_table(by_origin, "review_trigger_by_origin")
        fig, ax = new_fig(3.5, 2.5)
        res = ev[(ev["scope"] == "pooled_4_origins") & ev["rule"].isin(["T1", "T2", "T3", "T4", "any_no_T3", "any", "tier_only_high"])]
        xs = np.arange(len(res)); w = 0.38
        ax.bar(xs - w / 2, res["MAE_flagged"], w, color="#c0504d", label="flagged"); ax.bar(xs + w / 2, res["MAE_unflagged"], w, color="#1f4e79", label="not flagged")
        ax.set_xticks(xs); ax.set_xticklabels(res["rule"], rotation=25, ha="right"); ax.set_ylabel("MAE"); ax.legend(frameon=False)
        save_fig(fig, "flagged_vs_unflagged_error")
        log.info("E11 done")


if __name__ == "__main__":
    main()
