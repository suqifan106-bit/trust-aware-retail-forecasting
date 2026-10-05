"""Model zoo, random-search spaces and thesis configurations."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from . import config as C


def make_model(name, params=None, seed=C.SEED, n_jobs=C.N_JOBS):
    params = dict(params or {})
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0))
    if name == "rf":
        p = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3, "max_features": 0.8}
        p.update(params)
        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p)
    if name == "xgb":
        from xgboost import XGBRegressor
        p = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9,
             "colsample_bytree": 0.9, "min_child_weight": 5}
        p.update(params)
        return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=n_jobs,
                            tree_method="hist", verbosity=0, **p)
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        p = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.9,
             "subsample_freq": 1, "colsample_bytree": 0.9, "min_child_samples": 20}
        p.update(params)
        return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
    raise ValueError(name)


def needs_scaling(name):
    return name in ("linear", "ridge")


def sample_params(name, rng):
    """One random configuration from the search space of `name`."""
    ch = rng.choice
    if name == "ridge":
        return {"alpha": float(10 ** rng.uniform(-2, 3))}
    if name == "rf":
        return {"n_estimators": int(ch([100, 200, 300])), "max_depth": int(ch([8, 12, 16, 24])),
                "min_samples_leaf": int(ch([1, 3, 5, 10])), "max_features": float(ch([0.5, 0.8, 1.0]))}
    if name == "xgb":
        return {"n_estimators": int(ch([200, 400, 800])), "learning_rate": float(ch([0.03, 0.05, 0.1])),
                "max_depth": int(ch([4, 6, 8])), "subsample": float(ch([0.7, 0.9])),
                "colsample_bytree": float(ch([0.7, 0.9])), "min_child_weight": int(ch([1, 5, 10]))}
    if name == "lgbm":
        return {"n_estimators": int(ch([200, 400, 800])), "learning_rate": float(ch([0.03, 0.05, 0.1])),
                "num_leaves": int(ch([31, 63, 127, 255])), "subsample": float(ch([0.7, 0.9])),
                "colsample_bytree": float(ch([0.7, 0.9])), "min_child_samples": int(ch([10, 20, 50]))}
    return {}


def sanitize(params):
    """Make params JSON-serialisable (numpy types, None)."""
    out = {}
    for k, v in params.items():
        if v is None or isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v if v != "None" else None
    if "max_depth" in out and out["max_depth"] in ("None", "nan"):
        out["max_depth"] = None
    return out


# Thesis configurations (for E13 replication).
THESIS_PARAMS = {
    "ridge": {"alpha": 10.0},
    "rf": {"n_estimators": 5, "max_depth": 8, "min_samples_leaf": 3, "max_features": 1.0},
    "xgb": {"n_estimators": 45, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.9,
            "colsample_bytree": 0.9, "min_child_weight": 1},
}
