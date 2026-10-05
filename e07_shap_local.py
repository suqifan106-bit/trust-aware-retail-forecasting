"""E7: local SHAP cases (10/50/90 % predicted quantiles + largest error) with waterfall plots."""
import json

import numpy as np
import pandas as pd

from common import config as C
from common.io import Manifest, get_logger, save_fig, save_table
from common.plotting import new_fig


def waterfall(ax, base, contrib, names, pred, k=8):
    order = np.argsort(np.abs(contrib))[::-1]
    top = order[:k]; rest = contrib[order[k:]].sum()
    labels = [names[i] for i in top] + ["other features"]
    vals = list(contrib[top]) + [rest]
    cum = base; ys = []
    for v in vals:
        ys.append(cum); cum += v
    colors = ["#c0504d" if v < 0 else "#1f4e79" for v in vals]
    ax.barh(range(len(vals)), vals, left=ys, color=colors, height=0.6)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.axvline(base, color="grey", lw=0.8, ls="--"); ax.axvline(pred, color="black", lw=0.8)
    ax.set_xlabel(f"base value {base:,.0f}  →  prediction {pred:,.0f}")


def main():
    log = get_logger("e07_shap_local")
    with Manifest("e07_shap_local"):
        best = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        z = np.load(C.RESULTS_DIR / "shap" / f"shap_{best}_B_{C.MAIN_ORIGIN}.npz", allow_pickle=True)
        sv, cols, pred, y = z["shap_values"], list(z["columns"]), z["pred"], z["y"]
        base = float(z["base_value"]) if np.isfinite(z["base_value"]) else float(pred.mean() - sv.sum(1).mean())
        qs = np.quantile(pred, [0.1, 0.5, 0.9])
        idx = [int(np.argmin(np.abs(pred - q))) for q in qs] + [int(np.argmax(np.abs(y - pred)))]
        labels = ["low (10th pct)", "typical (50th pct)", "high (90th pct)", "largest error"]
        rows = []
        for n, (i, lab) in enumerate(zip(idx, labels), 1):
            for j in np.argsort(np.abs(sv[i]))[::-1]:
                rows.append({"case": n, "case_type": lab, "Store": int(z["store"][i]), "Dept": int(z["dept"][i]), "Date": str(z["date"][i]),
                             "actual": float(y[i]), "predicted": float(pred[i]), "base_value": base, "feature": cols[j],
                             "feature_value": float(z["x"][i, j]), "shap_value": float(sv[i, j])})
            fig, ax = new_fig(3.5, 2.8)
            waterfall(ax, base, sv[i], cols, float(pred[i]))
            ax.set_title(f"Case {n}: {lab} | Store {int(z['store'][i])} Dept {int(z['dept'][i])} {z['date'][i]} | actual {y[i]:,.0f}", fontsize=7)
            save_fig(fig, f"shap_waterfall_case{n}")
        save_table(pd.DataFrame(rows), "local_shap_cases")
        log.info("E7 done: cases %s", list(zip(labels, idx)))


if __name__ == "__main__":
    main()
