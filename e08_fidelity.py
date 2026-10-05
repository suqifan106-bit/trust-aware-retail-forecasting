"""E8: explanation fidelity as a continuous, controlled metric (all five tuned models, set B).

Fidelity@k = mean ΔWMAE% after permuting the top-k SHAP features
             − mean ΔWMAE% after permuting k random non-top-k features.
Also a deletion curve (cumulative permutation in SHAP order vs random order).
"""
import json

import joblib
import numpy as np
import pandas as pd

from common import config as C
from common.data import load_dataset, split_at_origin
from common.io import Manifest, get_logger, save_fig, save_table
from common.metrics import wmae
from common.plotting import new_fig
from common.shap_utils import global_importance, make_explainer, shap_matrix, shap_matrix_parallel


def permuted_wmae(model, x, y, hol, feats, rng):
    xp = x.copy()
    for f in feats:
        xp[f] = rng.permutation(xp[f].values)
    return wmae(y, model.predict(xp), hol)


def main():
    log = get_logger("e08_fidelity")
    with Manifest("e08_fidelity"):
        df = load_dataset()
        train, test = split_at_origin(df, C.MAIN_ORIGIN)
        y, hol = test[C.TARGET].values, test["IsHoliday"].values
        rows, curve_rows = [], []
        for name in C.MODELS:
            b = joblib.load(C.RESULTS_DIR / "models" / f"{name}_B_main.joblib")
            model, pre = b["model"], b["pre"]
            x = pre.transform(test); xtr = pre.transform(train)
            base = wmae(y, model.predict(x), hol)
            sub = x.sample(min(C.FIDELITY_SHAP_ROWS, len(x)), random_state=C.SEED)
            bg = xtr.sample(min(C.SHAP_BACKGROUND, len(xtr)), random_state=C.SEED)
            imp = global_importance(shap_matrix_parallel(name, model, sub, bg), x.columns)
            ranking = list(imp["feature"]); p = len(ranking)
            log.info("%s base WMAE=%.2f top5=%s", name, base, ranking[:5])
            rng = np.random.default_rng(C.SEED)
            for k in C.PERM_KS:
                top = ranking[:k]; others = ranking[k:]
                d_top = [(permuted_wmae(model, x, y, hol, top, rng) - base) / base * 100 for _ in range(C.N_PERM_REPEATS)]
                d_rnd = [(permuted_wmae(model, x, y, hol, list(rng.choice(others, size=min(k, len(others)), replace=False)), rng) - base) / base * 100
                         for _ in range(C.N_PERM_REPEATS)]
                rows.append({"model": name, "k": k, "base_WMAE": base, "delta_top_mean": np.mean(d_top), "delta_top_std": np.std(d_top),
                             "delta_random_mean": np.mean(d_rnd), "delta_random_std": np.std(d_rnd),
                             "fidelity_at_k": np.mean(d_top) - np.mean(d_rnd), "repeats": C.N_PERM_REPEATS, "top_features": "|".join(top)})
            # deletion curves (cumulative), 3 repeats each
            for mode in ["shap_order", "random_order"]:
                for r in range(3):
                    order = ranking if mode == "shap_order" else list(rng.permutation(ranking))
                    xp = x.copy()
                    for i, f in enumerate(order, 1):
                        xp[f] = rng.permutation(xp[f].values)
                        if i in (1, 2, 3, 5, 8, 12, 16, 20, p) or i == p:
                            curve_rows.append({"model": name, "mode": mode, "repeat": r, "n_removed": i,
                                               "delta_WMAE_pct": (wmae(y, model.predict(xp), hol) - base) / base * 100})
        res = pd.DataFrame(rows); save_table(res, "fidelity_permutation")
        # ---- grouped permutation (whole feature groups) and within-series permutation (shuffle time order
        # inside each Store-Dept series), which keep cross-sectional structure more realistic than global shuffling.
        grp_rows = []
        keys = test[["Store", "Dept"]].reset_index(drop=True)
        for name in C.MODELS:
            b = joblib.load(C.RESULTS_DIR / "models" / f"{name}_B_main.joblib"); model, pre = b["model"], b["pre"]
            x = pre.transform(test).reset_index(drop=True); base = wmae(y, model.predict(x), hol)
            rng = np.random.default_rng(C.SEED)
            for gname, gfeats in C.FEATURE_GROUPS.items():
                feats = [f for f in gfeats if f in x.columns]
                d_glob, d_ser = [], []
                for r in range(max(3, C.N_PERM_REPEATS // 4)):
                    d_glob.append((permuted_wmae(model, x, y, hol, feats, rng) - base) / base * 100)
                    xp = x.copy()
                    perm = keys.groupby(["Store", "Dept"]).indices
                    for f in feats:
                        col = xp[f].values.copy()
                        for idx in perm.values():
                            if len(idx) > 1:
                                col[idx] = col[rng.permutation(idx)]
                        xp[f] = col
                    d_ser.append((wmae(y, model.predict(xp), hol) - base) / base * 100)
                grp_rows.append({"model": name, "group": gname, "n_features": len(feats), "delta_global_perm_mean": np.mean(d_glob), "delta_global_perm_std": np.std(d_glob),
                                 "delta_within_series_perm_mean": np.mean(d_ser), "delta_within_series_perm_std": np.std(d_ser)})
                log.info("%s group %s global=%.1f%% within-series=%.1f%%", name, gname, np.mean(d_glob), np.mean(d_ser))
        save_table(pd.DataFrame(grp_rows), "fidelity_grouped_permutation")
        curves = pd.DataFrame(curve_rows); save_table(curves, "fidelity_deletion_curve")
        auc_rows = []
        for (mname, mode), g in curves.groupby(["model", "mode"]):
            cur = g.groupby("n_removed")["delta_WMAE_pct"].mean()
            auc_rows.append({"model": mname, "mode": mode, "auc": float(np.trapezoid(cur.values, cur.index.values))})
        save_table(pd.DataFrame(auc_rows), "fidelity_deletion_auc")
        log.info("\n%s", res[["model", "k", "delta_top_mean", "delta_random_mean", "fidelity_at_k"]].to_string())
        fig, ax = new_fig(3.5, 2.5)
        for i, name in enumerate(C.MODELS):
            d = res[res["model"] == name]
            ax.errorbar(d["k"], d["delta_top_mean"], yerr=d["delta_top_std"], marker="o", ms=3, lw=1, label=C.MODEL_LABELS[name], capsize=2)
        d = res.groupby("k")["delta_random_mean"].mean()
        ax.plot(d.index, d.values, "k--", lw=1, label="random-k control (mean)")
        ax.set_xlabel("k (top-k SHAP features permuted)"); ax.set_ylabel("ΔWMAE (%)"); ax.legend(frameon=False)
        save_fig(fig, "fidelity_permutation")
        fig, ax = new_fig(3.5, 2.5)
        best = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        for mode, c in [("shap_order", "#1f4e79"), ("random_order", "#c0504d")]:
            d = curves[(curves["model"] == best) & (curves["mode"] == mode)].groupby("n_removed")["delta_WMAE_pct"].mean()
            ax.plot(d.index, d.values, marker="o", ms=3, lw=1, color=c, label=mode.replace("_", " "))
        ax.set_xlabel("features permuted (cumulative)"); ax.set_ylabel("ΔWMAE (%)"); ax.set_title(f"Deletion curve, {C.MODEL_LABELS[best]}"); ax.legend(frameon=False)
        save_fig(fig, "fidelity_deletion_curve")
        log.info("E8 done")


if __name__ == "__main__":
    main()
