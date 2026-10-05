"""E2: unified benchmark with random-search tuning, frozen before the earliest origin.

Protocol:
  * Hyper-parameters are tuned ONLY on rows dated before C.TUNING_CUTOFF (the earliest rolling origin)
    and then frozen for every origin, so all rolling evaluations are forward-looking.
  * Cross-validation folds are formed on UNIQUE DATES (contiguous blocks of weeks), never splitting a week.
  * Preprocessing (medians / scaling) is fitted inside each training fold.
  * Two grouped-ablation feature sets are evaluated with frozen parameters of the nearest tuned set
    (A_lag uses set-B parameters, B_nolag uses set-A parameters).
Outputs (main origin): model_metrics_main, best_hyperparameters, tuning_history, best_model.json,
predictions/<model>_<set>_main(_train), models/*.joblib, figure wmae_by_model_featureset.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from common import config as C
from common.data import Preprocessor, load_dataset, split_at_origin
from common.io import Manifest, get_logger, save_fig, save_predictions, save_table
from common.metrics import all_metrics, series_scales, wmae
from common.models import make_model, needs_scaling, sample_params, sanitize
from common.plotting import new_fig


def date_folds(dates, n_splits):
    """Expanding-window folds on unique dates: fold k validates on block k+1 of (n_splits+1) blocks."""
    u = np.array(sorted(pd.unique(dates)))
    blocks = np.array_split(u, n_splits + 1)
    for k in range(1, n_splits + 1):
        train_dates = np.concatenate(blocks[:k]); val_dates = blocks[k]
        yield np.where(np.isin(dates, train_dates))[0], np.where(np.isin(dates, val_dates))[0]


def cv_score(name, params, feats, df, n_splits):
    """Mean WMAE over date-block folds; preprocessing fitted inside each training fold."""
    scores = []
    for tr, va in date_folds(df["Date"].values, n_splits):
        dtr, dva = df.iloc[tr], df.iloc[va]
        pre = Preprocessor(feats, scale=needs_scaling(name))
        xtr = pre.fit_transform(dtr); xva = pre.transform(dva)
        m = make_model(name, params); m.fit(xtr, dtr[C.TARGET])
        scores.append(wmae(dva[C.TARGET], m.predict(xva), dva["IsHoliday"]))
    return float(np.mean(scores))


def fit_eval(name, params, feats, train, test, scales, fs_name, log, cv_best=np.nan, t0=None):
    pre = Preprocessor(feats, scale=needs_scaling(name))
    xtr = pre.fit_transform(train); xte = pre.transform(test)
    model = make_model(name, params); model.fit(xtr, train[C.TARGET])
    pred_te = model.predict(xte); pred_tr = model.predict(xtr)
    met = all_metrics(test, pred_te, scales)
    tag = f"{name}_{fs_name}_main"
    save_predictions(pd.concat([test[C.KEYS + ["IsHoliday", C.TARGET]].reset_index(drop=True), pd.Series(pred_te, name="pred")], axis=1), tag)
    save_predictions(pd.concat([train[C.KEYS + ["IsHoliday", C.TARGET]].reset_index(drop=True), pd.Series(pred_tr, name="pred")], axis=1), f"{tag}_train")
    joblib.dump({"model": model, "pre": pre, "name": name, "feature_set": fs_name, "params": sanitize(params)},
                C.RESULTS_DIR / "models" / f"{tag}.joblib", compress=3)
    log.info("[%s/%s] TEST %s", fs_name, name, {k: round(v, 4) for k, v in met.items()})
    return {"feature_set": fs_name, "model": name, "cv_wmae": cv_best, **met, "fit_seconds": (time.time() - t0) if t0 else np.nan}


def main():
    log = get_logger("e02_benchmark_tuning")
    with Manifest("e02_benchmark_tuning") as man:
        df = load_dataset()
        tune_df = df[df["Date"] < pd.Timestamp(C.TUNING_CUTOFF)].reset_index(drop=True)
        train, test = split_at_origin(df, C.MAIN_ORIGIN)
        scales = series_scales(train)
        log.info("tuning rows=%d (dates < %s) | main-origin train=%d test=%d", len(tune_df), C.TUNING_CUTOFF, len(train), len(test))
        rng = np.random.default_rng(C.SEED)
        metrics_rows, best_rows, history, frozen = [], [], [], {}
        for fs_name, feats in C.FEATURE_SETS.items():
            for name in C.MODELS:
                t0 = time.time()
                cands = [{}] if name == "linear" else [sample_params(name, rng) for _ in range(C.N_ITER[name])]
                best = None
                for i, p in enumerate(cands):
                    s = cv_score(name, p, feats, tune_df, C.CV_SPLITS)
                    history.append({"feature_set": fs_name, "model": name, "iter": i, "cv_wmae": s, "params": json.dumps(sanitize(p))})
                    if best is None or s < best[0]:
                        best = (s, p)
                    log.info("[%s/%s] iter %d cv_wmae=%.2f %s", fs_name, name, i, s, sanitize(p))
                cv_best, params = best
                frozen[(fs_name, name)] = params
                best_rows.append({"feature_set": fs_name, "model": name, "cv_wmae": cv_best, "params": json.dumps(sanitize(params)),
                                  "tuning_cutoff": C.TUNING_CUTOFF, "tuning_rows": len(tune_df)})
                metrics_rows.append(fit_eval(name, params, feats, train, test, scales, fs_name, log, cv_best, t0))
        # grouped ablation sets reuse the frozen parameters of the nearest tuned set:
        # A_lag (A + history) <- set-B parameters; B_nolag (B without history) <- set-A parameters.
        nearest = {"A_lag": "B", "B_nolag": "A"}
        for fs_name, feats in C.ABLATION_SETS.items():
            for name in C.MODELS:
                t0 = time.time()
                metrics_rows.append(fit_eval(name, frozen[(nearest[fs_name], name)], feats, train, test, scales, fs_name, log, np.nan, t0))
        metrics = pd.DataFrame(metrics_rows)
        save_table(metrics, "model_metrics_main")
        save_table(pd.DataFrame(best_rows), "best_hyperparameters")
        save_table(pd.DataFrame(history), "tuning_history")
        b = metrics[metrics["feature_set"] == "B"].sort_values("cv_wmae").iloc[0]
        best_info = {"best_model": b["model"], "feature_set": "B", "cv_wmae": float(b["cv_wmae"]), "test_wmae": float(b["WMAE"]),
                     "selection_rule": f"lowest date-block CV WMAE on rows before {C.TUNING_CUTOFF}", "tuning_cutoff": C.TUNING_CUTOFF}
        with open(C.RESULTS_DIR / "tables" / "best_model.json", "w") as f:
            json.dump(best_info, f, indent=2)
        man.add(**best_info)
        log.info("engine (frozen): %s", best_info)
        fig, ax = new_fig(3.5, 2.5)
        order = ["A", "B_nolag", "A_lag", "B"]; piv = metrics.pivot(index="model", columns="feature_set", values="WMAE").reindex(C.MODELS)[order]
        xs = np.arange(len(piv)); w = 0.2; cols = ["#9db5cc", "#c9b3d6", "#7fb59a", "#1f4e79"]
        labels = ["A: no sales history", "B minus history", "A plus history", "B: full"]
        for j, c in enumerate(order):
            ax.bar(xs + (j - 1.5) * w, piv[c], w, label=labels[j], color=cols[j])
        ax.set_xticks(xs); ax.set_xticklabels([C.MODEL_LABELS[m] for m in piv.index], rotation=20, ha="right")
        ax.set_ylabel("WMAE (test, main origin)"); ax.set_yscale("log"); ax.legend(frameon=False, fontsize=6)
        save_fig(fig, "wmae_by_model_featureset")
        log.info("E2 done")


if __name__ == "__main__":
    main()
