"""E10: retail-logic plausibility against a literature-derived reference set + sign consistency.

Reference drivers of weekly store-department sales (with sources):
  R1 recent sales history / lags ........ Soleimani et al. 2026; M5 top solutions (Makridakis et al. 2022)
  R2 product department / category ...... Yang 2023; Niu 2020; Kaggle competition write-ups
  R3 store size & store type ............ Yang 2023; Tayyab & Nasim 2024
  R4 holiday weeks ...................... Walmart competition definition (WMAE weight 5); Fildes et al. 2022
  R5 seasonality / week of year ......... Fildes et al. 2022; Hyndman & Athanasopoulos 2021
  R6 promotions (markdowns) ............. Fildes et al. 2022; Arboleda-Florez & Castro-Zuluaga 2023
  R7 macro-economic conditions .......... Niu 2020 (CPI, unemployment, fuel price)
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import config as C
from common.data import load_dataset, split_at_origin
from common.io import Manifest, get_logger, save_table
from common.shap_utils import global_importance, make_explainer, shap_matrix, shap_matrix_parallel, top_k

REFERENCE = [
    ("R1", "recent sales history (lags, rolling means)", "Soleimani et al. (2026); Makridakis et al. (2022) M5"),
    ("R2", "product department", "Yang (2023); Niu (2020)"),
    ("R3", "store size and type", "Yang (2023); Tayyab & Nasim (2024)"),
    ("R4", "holiday weeks", "Kaggle Walmart competition (WMAE weighting); Fildes et al. (2022)"),
    ("R5", "seasonality / week of year", "Fildes et al. (2022); Hyndman & Athanasopoulos (2021)"),
    ("R6", "promotions / markdowns", "Fildes et al. (2022); Arboleda-Florez & Castro-Zuluaga (2023)"),
    ("R7", "macro-economic conditions", "Niu (2020)"),
]
CATEGORY = {**{f: "R1" for f in ["lag_1", "lag_2", "lag_4", "lag_8", "lag_52", "roll_mean_4", "roll_std_4", "roll_mean_12", "lag_1_available"]},
            "Dept": "R2", "Store": "R3", "Size": "R3", "Type_A": "R3", "Type_B": "R3", "Type_C": "R3",
            "IsHoliday": "R4", "is_superbowl": "R4", "is_laborday": "R4", "is_thanksgiving": "R4", "is_christmas": "R4",
            "Week": "R5", "Month": "R5", "Year": "R5",
            **{f"MarkDown{i}": "R6" for i in range(1, 6)}, **{f"MarkDown{i}_recorded": "R6" for i in range(1, 6)},
            "CPI": "R7", "Unemployment": "R7", "Fuel_Price": "R7", "Temperature": "R7"}
# expected monotone direction of the SHAP value w.r.t. the feature value (+1 / -1)
EXPECTED_SIGN = {"lag_1": 1, "lag_4": 1, "lag_52": 1, "roll_mean_4": 1, "roll_mean_12": 1, "Size": 1,
                 "is_thanksgiving": 1, "Unemployment": -1, "MarkDown1": 1, "MarkDown5": 1}


def main():
    log = get_logger("e10_plausibility")
    with Manifest("e10_plausibility"):
        save_table(pd.DataFrame(REFERENCE, columns=["id", "driver", "sources"]), "plausibility_reference_set")
        df = load_dataset()
        train, test = split_at_origin(df, C.MAIN_ORIGIN)
        rows, sign_rows = [], []
        for name in C.MODELS:
            b = joblib.load(C.RESULTS_DIR / "models" / f"{name}_B_main.joblib")
            model, pre = b["model"], b["pre"]
            x = pre.transform(test); xtr = pre.transform(train)
            sub = x.sample(min(C.FIDELITY_SHAP_ROWS, len(x)), random_state=C.SEED)
            sv = shap_matrix_parallel(name, model, sub, xtr.sample(min(C.SHAP_BACKGROUND, len(xtr)), random_state=C.SEED))
            imp = global_importance(sv, x.columns)
            t10 = top_k(imp, 10)
            cats = {CATEGORY.get(f, "unmapped") for f in t10}
            covered = sorted(c for c in cats if c != "unmapped")
            # sign consistency
            ok, checked = 0, 0
            for f, s in EXPECTED_SIGN.items():
                if f in sub.columns:
                    j = list(sub.columns).index(f)
                    if sub[f].nunique() > 1 and np.std(sv[:, j]) > 0:
                        rho = spearmanr(sub[f].values, sv[:, j]).statistic
                        agree = int(np.sign(rho) == s)
                        sign_rows.append({"model": name, "feature": f, "expected_sign": s, "spearman_rho": rho, "agrees": agree,
                                          "rank_in_model": int(imp.index[imp["feature"] == f][0]) + 1})
                        ok += agree; checked += 1
            rows.append({"model": name, "top10": "|".join(t10), "categories_covered": "|".join(covered),
                         "coverage_ratio": len(covered) / len(REFERENCE), "unmapped_in_top10": int("unmapped" in cats),
                         "sign_checks": checked, "sign_agreement_ratio": ok / checked if checked else np.nan})
            log.info("%s coverage=%.3f sign=%d/%d top10=%s", name, len(covered) / len(REFERENCE), ok, checked, t10)
        save_table(pd.DataFrame(rows), "plausibility_scores")
        save_table(pd.DataFrame(sign_rows), "plausibility_sign_consistency")
        log.info("E10 done")


if __name__ == "__main__":
    main()
