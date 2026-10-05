"""E12: ablation tables with a single, explicit scope per table.

Table 1 (main origin only): feature-set ablation for every model (A, B minus history, A plus history, B).
Table 2 (main origin only): routing ablation for the engine, all numbers from the main-origin test window.
Table 3 (pooled four origins): routing ablation pooled, reported separately.
"""
import json

import pandas as pd

from common import config as C
from common.io import Manifest, get_logger, load_table, save_table


def main():
    log = get_logger("e12_ablation")
    with Manifest("e12_ablation"):
        engine = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        m = load_table("model_metrics_main"); ev = load_table("review_trigger_evaluation")
        piv = m.pivot(index="model", columns="feature_set", values="WMAE").reindex(C.MODELS)[["A", "B_nolag", "A_lag", "B"]]
        piv["history_effect_pct_A_to_A_lag"] = (piv["A_lag"] / piv["A"] - 1) * 100
        piv["history_effect_pct_B_nolag_to_B"] = (piv["B"] / piv["B_nolag"] - 1) * 100
        piv["indicator_effect_pct_A_to_B_nolag"] = (piv["B_nolag"] / piv["A"] - 1) * 100
        piv = piv.reset_index(); save_table(piv, "ablation_features_main_origin")
        log.info("\n%s", piv.round(2).to_string())
        rows = []
        for scope, name in [("main_origin", "ablation_routing_main_origin"), ("pooled_4_origins", "ablation_routing_pooled")]:
            e = ev[ev["scope"] == scope].set_index("rule")
            base = m[(m["model"] == engine) & (m["feature_set"] == "B")].iloc[0]
            rows = [
                {"variant": f"{C.MODEL_LABELS[engine]}, set B, no routing", "WMAE_main_origin": base["WMAE"], "flag_rate": None, "large_error_recall": None, "large_error_precision": None},
                {"variant": "+ sales tier only (high tier reviewed)", "WMAE_main_origin": base["WMAE"], "flag_rate": e.loc["tier_only_high", "flag_rate"], "large_error_recall": e.loc["tier_only_high", "large_error_recall"], "large_error_precision": e.loc["tier_only_high", "large_error_precision"]},
                {"variant": "+ triggers T1+T2+T4", "WMAE_main_origin": base["WMAE"], "flag_rate": e.loc["any_no_T3", "flag_rate"], "large_error_recall": e.loc["any_no_T3", "large_error_recall"], "large_error_precision": e.loc["any_no_T3", "large_error_precision"]},
                {"variant": "+ triggers T1+T2+T3+T4", "WMAE_main_origin": base["WMAE"], "flag_rate": e.loc["any", "flag_rate"], "large_error_recall": e.loc["any", "large_error_recall"], "large_error_precision": e.loc["any", "large_error_precision"]},
            ]
            save_table(pd.DataFrame(rows).assign(scope=scope), name)
            log.info("[%s]\n%s", scope, pd.DataFrame(rows).to_string())
        log.info("E12 done")


if __name__ == "__main__":
    main()
