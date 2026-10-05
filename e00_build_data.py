"""E0: build the full feature dataset, EDA tables and series completeness report."""
import numpy as np
import pandas as pd

from common import config as C
from common.data import build_full_dataset, dataset_path
from common.io import Manifest, get_logger, save_table
from common.plotting import new_fig
from common.io import save_fig


def main():
    log = get_logger("e00_build_data")
    with Manifest("e00_build_data") as m:
        df = build_full_dataset()
        if C.SMOKE_STORES:
            df = df[df["Store"].isin(C.SMOKE_STORES)].reset_index(drop=True)
            log.info("SMOKE: keeping stores %s -> %d rows", C.SMOKE_STORES, len(df))
        df.to_parquet(dataset_path(), index=False)
        log.info("dataset rows=%d cols=%d", *df.shape)

        # --- feature dictionary
        rows = []
        for f in C.FEATURES_B:
            src = ("lag" if f in C.LAG_FEATURES else "extra" if f in C.EXTRA_FEATURES else "thesis")
            rows.append({"feature": f, "in_set_A": f in C.FEATURES_A, "in_set_B": True, "group": src,
                         "dtype": str(df[f].dtype), "missing_ratio": float(df[f].isna().mean())})
        save_table(pd.DataFrame(rows), "feature_dictionary")

        # --- series completeness
        g = df.groupby(["Store", "Dept"])
        comp = pd.DataFrame({"n_weeks": g.size(), "first": g["Date"].min(), "last": g["Date"].max()})
        comp["expected_weeks"] = ((comp["last"] - comp["first"]).dt.days // 7 + 1)
        comp["missing_weeks"] = comp["expected_weeks"] - comp["n_weeks"]
        comp = comp.reset_index()
        save_table(comp, "series_completeness")
        summary = pd.DataFrame([{
            "n_series": len(comp), "mean_weeks": comp["n_weeks"].mean(), "median_weeks": comp["n_weeks"].median(),
            "series_full_143": int((comp["n_weeks"] == 143).sum()), "series_with_gaps": int((comp["missing_weeks"] > 0).sum()),
            "negative_sales_rows": int((df[C.TARGET] < 0).sum()), "zero_sales_rows": int((df[C.TARGET] == 0).sum()),
            "rows": len(df), "date_min": df["Date"].min().date(), "date_max": df["Date"].max().date(),
        }])
        save_table(summary, "dataset_summary")

        # --- EDA on the full data (one consistent basis)
        desc_cols = [C.TARGET, "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment"] + [f"MarkDown{i}" for i in range(1, 6)]
        desc = df[desc_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index().rename(columns={"index": "variable"})
        save_table(desc, "eda_descriptive_statistics")
        miss = pd.DataFrame({"variable": [f"MarkDown{i}" for i in range(1, 6)],
                             "missing_ratio_before_fill": [1 - df[f"MarkDown{i}_recorded"].mean() for i in range(1, 6)]})
        save_table(miss, "eda_missing_markdown")
        q1, q3 = df[C.TARGET].quantile([0.25, 0.75]); iqr = q3 - q1
        out = ((df[C.TARGET] < q1 - 1.5 * iqr) | (df[C.TARGET] > q3 + 1.5 * iqr))
        save_table(pd.DataFrame([{"Q1": q1, "Q3": q3, "IQR": iqr, "outlier_rows": int(out.sum()), "outlier_ratio": float(out.mean())}]), "eda_outliers_iqr")
        num = [C.TARGET, "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", "Week", "lag_1", "lag_52"]
        corr = df[num].corr()[C.TARGET].drop(C.TARGET).rename("pearson_r").reset_index().rename(columns={"index": "variable"})
        save_table(corr, "eda_correlation_with_target")
        by_type = df.groupby("Type")[C.TARGET].agg(["count", "mean", "median", "std"]).reset_index()
        by_hol = df.groupby("IsHoliday")[C.TARGET].agg(["count", "mean", "median", "std"]).reset_index()
        save_table(by_type, "eda_sales_by_store_type"); save_table(by_hol, "eda_sales_by_holiday")

        fig, ax = new_fig(3.5, 2.4)
        ax.hist(df[C.TARGET].clip(df[C.TARGET].quantile(0.01), df[C.TARGET].quantile(0.99)), bins=60, color="#1f4e79")
        ax.set_xlabel("Weekly sales (1st-99th pct clipped)"); ax.set_ylabel("Rows")
        save_fig(fig, "eda_weekly_sales_hist")
        weekly = df.groupby("Date")[C.TARGET].sum().reset_index()
        fig, ax = new_fig(7, 2.4)
        ax.plot(weekly["Date"], weekly[C.TARGET] / 1e6, color="#1f4e79", lw=1)
        for o in C.ORIGINS:
            ax.axvline(pd.Timestamp(o), color="#c0504d", ls="--", lw=0.8)
        ax.set_ylabel("Total weekly sales (M$)"); ax.set_xlabel("Week (dashed = forecast origins)")
        save_fig(fig, "eda_total_sales_timeline")
        m.add(rows=len(df), n_series=len(comp))
        log.info("E0 done")


if __name__ == "__main__":
    main()
