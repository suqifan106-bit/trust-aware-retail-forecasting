"""E9: explanation stability of the best model under (a) test resampling, (b) retraining seeds, (c) origins."""
import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from common import config as C
from common.data import Preprocessor, load_dataset, split_at_origin
from common.io import Manifest, get_logger, load_table, save_fig, save_table
from common.models import make_model, needs_scaling
from common.plotting import new_fig
from common.shap_utils import global_importance, jaccard, make_explainer, shap_matrix, shap_matrix_parallel, top_k


def pairwise_stats(rankings, k=10):
    """rankings: list of importance DataFrames. Returns Jaccard@k and Kendall tau over all pairs."""
    js, taus = [], []
    for a, b in combinations(rankings, 2):
        js.append(jaccard(top_k(a, k), top_k(b, k)))
        ra = a.set_index("feature")["mean_abs_shap"]; rb = b.set_index("feature")["mean_abs_shap"].reindex(ra.index)
        taus.append(kendalltau(ra.values, rb.values).statistic)
    return {"jaccard_top10_mean": np.mean(js), "jaccard_top10_std": np.std(js), "jaccard_top10_min": np.min(js),
            "kendall_tau_mean": np.mean(taus), "kendall_tau_std": np.std(taus), "n_pairs": len(js)}


def main():
    log = get_logger("e09_stability")
    with Manifest("e09_stability"):
        best = json.load(open(C.RESULTS_DIR / "tables" / "best_model.json"))["best_model"]
        df = load_dataset()
        rows, rank_tables = [], []
        # (a) resampling the explained test rows
        z = np.load(C.RESULTS_DIR / "shap" / f"shap_{best}_B_{C.MAIN_ORIGIN}.npz", allow_pickle=True)
        sv, cols = z["shap_values"], list(z["columns"])
        rng = np.random.default_rng(C.SEED)
        ranks = [global_importance(sv[rng.integers(0, len(sv), len(sv))], cols) for _ in range(C.N_STAB_BOOT)]
        rows.append({"source": "a_test_resampling", "n_repeats": C.N_STAB_BOOT, **pairwise_stats(ranks)})
        # (b) retraining with different seeds
        params = load_table("best_hyperparameters"); params = json.loads(params[(params["feature_set"] == "B") & (params["model"] == best)]["params"].iloc[0])
        train, test = split_at_origin(df, C.MAIN_ORIGIN)
        pre = Preprocessor(C.FEATURES_B, scale=needs_scaling(best)); xtr = pre.fit_transform(train); xte = pre.transform(test)
        sub = xte.sample(min(2000 if not C.SMOKE else 400, len(xte)), random_state=C.SEED); bg = xtr.sample(min(C.SHAP_BACKGROUND, len(xtr)), random_state=C.SEED)
        seed_ranks = []
        for s in range(C.N_SEEDS):
            m = make_model(best, params, seed=C.SEED + s); m.fit(xtr, train[C.TARGET])
            imp = global_importance(shap_matrix_parallel(best, m, sub, bg), xte.columns)
            seed_ranks.append(imp); rank_tables.append(imp.assign(source="seed", id=s))
            log.info("seed %d top5=%s", s, top_k(imp, 5))
        rows.append({"source": "b_retraining_seeds", "n_repeats": C.N_SEEDS, **pairwise_stats(seed_ranks)})
        # (c) origins
        bo = load_table("shap_importance_by_origin")
        origin_ranks = [bo[bo["origin"] == o].sort_values("mean_abs_shap", ascending=False) for o in C.ORIGINS]
        rows.append({"source": "c_forecast_origins", "n_repeats": len(C.ORIGINS), **pairwise_stats(origin_ranks)})
        res = pd.DataFrame(rows); save_table(res, "stability_jaccard")
        save_table(pd.concat(rank_tables, ignore_index=True), "stability_rankings_by_seed")
        log.info("\n%s", res.to_string())
        # rank heatmap over origins + seeds (top 15 of main)
        main_top = top_k(origin_ranks[C.ORIGINS.index(C.MAIN_ORIGIN)], 15)
        mats = []
        for lab, r in [(f"origin {o}", r) for o, r in zip(C.ORIGINS, origin_ranks)] + [(f"seed {i}", r) for i, r in enumerate(seed_ranks)]:
            rk = r.reset_index(drop=True); rk["rank"] = np.arange(1, len(rk) + 1)
            mats.append(rk.set_index("feature")["rank"].reindex(main_top).rename(lab))
        mat = pd.concat(mats, axis=1)
        fig, ax = new_fig(5.5, 3.2)
        im = ax.imshow(mat.values, cmap="viridis_r", aspect="auto"); ax.grid(False)
        ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index); ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=60, ha="right")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{int(mat.values[i, j])}", ha="center", va="center", fontsize=5.5, color="white")
        ax.set_title("Feature rank across origins and seeds")
        save_fig(fig, "stability_rank_heatmap")
        log.info("E9 done")


if __name__ == "__main__":
    main()
