"""E5: conditional (segment-wise) errors of the best model across all rolling origins."""
import json

import numpy as np
import pandas as pd

from common import config as C
from common.data import load_dataset
from common.io import Manifest, get_logger, load_predictions, save_fig, save_table
from common.metrics import smape, wmae
from common.plotting import new_fig


def seg_metrics(d):
    e = (d[C.TARGET] - d["pred"]).abs()
    return pd.Series({"n": len(d), "MAE": e.mean(), "WMAE": wmae(d[C.TARGET], d["pred"], d["IsHoliday"]),
                      "SMAPE": smape(d[C.TARGET], d["pred"]), "median_AE": e.median(), "bias": (d["pred"] - d[C.TARGET]).mean()})


def main():
    log = get_logger("e05_segment_errors")
    with Manifest("e05_segment_errors"):
        best = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        df = load_dataset()
        meta = df[C.KEYS + ["Type", "Size"] + list(C.HOLIDAY_WEEKS)]
        parts = []
        for o in C.ORIGINS:
            p = load_predictions(f"{best}_B_{o}").assign(origin=o)
            parts.append(p)
        d = pd.concat(parts, ignore_index=True).merge(meta, on=C.KEYS, how="left")
        d["size_tercile"] = pd.qcut(d["Size"], 3, labels=["small", "medium", "large"])
        d["sales_quintile"] = pd.qcut(d[C.TARGET].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
        d["holiday_name"] = "none"
        for h in C.HOLIDAY_WEEKS:
            d.loc[d[h] == 1, "holiday_name"] = h.replace("is_", "")
        def by(col):
            rows = [pd.concat([pd.Series({"level": lvl}), seg_metrics(g)]) for lvl, g in d.groupby(col, observed=True)]
            s = pd.DataFrame(rows); s.insert(0, "segment", col); return s
        out = [by(col) for col in ["origin", "IsHoliday", "holiday_name", "Type", "size_tercile", "sales_quintile"]]
        dept = by("Dept").drop(columns="segment").rename(columns={"level": "Dept"}).sort_values("MAE")
        save_table(dept, "error_by_dept")
        top = pd.concat([dept.head(10).assign(group="lowest_MAE"), dept.tail(10).assign(group="highest_MAE")])
        top = top.rename(columns={"Dept": "level"}); top.insert(0, "segment", "Dept_" + top.pop("group"))
        out.append(top)
        seg = pd.concat(out, ignore_index=True)
        save_table(seg, "error_by_segment")
        log.info("\n%s", seg.to_string())
        # heatmap figure: WMAE by origin x holiday / quintile
        fig, axes = new_fig(7, 2.4)
        fig.clf(); ax1, ax2 = fig.subplots(1, 2)
        for ax, col in [(ax1, "sales_quintile"), (ax2, "holiday_name")]:
            piv = pd.Series({k: wmae(g[C.TARGET], g["pred"], g["IsHoliday"]) for k, g in d.groupby(["origin", col], observed=True)}).unstack(level=1)
            im = ax.imshow(piv.values, cmap="Reds", aspect="auto")
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
            ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index); ax.grid(False)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    ax.text(j, i, f"{piv.values[i, j]:,.0f}", ha="center", va="center", fontsize=5.5)
            ax.set_title(f"WMAE by {col}")
        save_fig(fig, "error_heatmap_segment")
        log.info("E5 done")


if __name__ == "__main__":
    main()
