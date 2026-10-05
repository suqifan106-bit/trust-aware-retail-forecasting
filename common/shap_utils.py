"""SHAP helpers shared by E6-E11."""
import numpy as np
import pandas as pd


def make_explainer(model_name, model, x_background=None):
    import shap
    if model_name in ("rf", "xgb", "lgbm"):
        # tree_path_dependent: exact for the fitted trees, no background needed, fast.
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, x_background)


def shap_matrix(explainer, x):
    # check_additivity is accepted by TreeExplainer only; LinearExplainer.shap_values(X) has no such argument.
    try:
        vals = explainer.shap_values(x, check_additivity=False)
    except TypeError:
        vals = explainer.shap_values(x)
    if isinstance(vals, list):
        vals = vals[0]
    return np.asarray(vals)


def _explain_chunk(model_name, model, x_chunk, x_background):
    return shap_matrix(make_explainer(model_name, model, x_background), x_chunk)


def shap_matrix_parallel(model_name, model, x, x_background=None, n_jobs=None, chunk_rows=None):
    """Row-chunked parallel SHAP (loky workers). TreeExplainer on random forests is single-threaded and
    slow; splitting rows across workers gives near-linear speed-up on a many-core machine.
    Chunk size adapts to the number of workers so that every worker is used even for small inputs."""
    import math
    from joblib import Parallel, delayed
    from . import config as C
    n_jobs = n_jobs or max(1, min(C.N_JOBS, 16))
    n = len(x)
    if chunk_rows is None:
        chunk_rows = max(100, math.ceil(n / n_jobs))
    if n_jobs == 1 or n < 200:
        return _explain_chunk(model_name, model, x, x_background)
    chunks = [x.iloc[i:i + chunk_rows] for i in range(0, n, chunk_rows)]
    parts = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=None)(delayed(_explain_chunk)(model_name, model, c, x_background) for c in chunks)
    return np.vstack(parts)


def global_importance(shap_vals, columns):
    imp = np.abs(shap_vals).mean(axis=0)
    return (pd.DataFrame({"feature": columns, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def top_k(importance_df, k):
    return list(importance_df["feature"].head(k))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0
