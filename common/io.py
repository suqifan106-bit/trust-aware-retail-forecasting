"""Result directories, logging, manifests and save helpers."""
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as C

SUBDIRS = ["tables", "figures", "predictions", "shap", "models", "logs", "manifests"]


def ensure_dirs():
    for d in SUBDIRS:
        (C.RESULTS_DIR / d).mkdir(parents=True, exist_ok=True)
    return C.RESULTS_DIR


def get_logger(script_name):
    ensure_dirs()
    log_path = C.RESULTS_DIR / "logs" / f"{script_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info("RUN_ID=%s SMOKE=%s N_JOBS=%s results=%s", C.RUN_ID, C.SMOKE, C.N_JOBS, C.RESULTS_DIR)
    return logger


def save_table(df, name, index=False):
    path = C.RESULTS_DIR / "tables" / f"{name}.csv"
    df.to_csv(path, index=index, float_format="%.6f")
    return path


def load_table(name, **kw):
    return pd.read_csv(C.RESULTS_DIR / "tables" / f"{name}.csv", **kw)


def save_predictions(df, name):
    path = C.RESULTS_DIR / "predictions" / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_predictions(name):
    return pd.read_parquet(C.RESULTS_DIR / "predictions" / f"{name}.parquet")


def save_fig(fig, name, dpi=300):
    out = C.RESULTS_DIR / "figures"
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def library_versions():
    out = {"python": platform.python_version()}
    for lib in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "shap", "matplotlib", "joblib", "pyarrow"]:
        try:
            out[lib] = __import__(lib).__version__
        except Exception:
            out[lib] = "not installed"
    return out


class Manifest:
    """Context manager that records a run manifest for one script."""

    def __init__(self, script_name, extra=None):
        self.script = script_name
        self.extra = extra or {}
        self.t0 = time.time()

    def __enter__(self):
        ensure_dirs()
        return self

    def add(self, **kw):
        self.extra.update(kw)

    def __exit__(self, exc_type, exc, tb):
        payload = {
            "script": self.script,
            "run_id": C.RUN_ID,
            "smoke": C.SMOKE,
            "started": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - self.t0, 1),
            "status": "ok" if exc is None else f"error: {exc!r}",
            "n_jobs": C.N_JOBS,
            "machine": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "libraries": library_versions(),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(C).items()
                       if k.isupper() and not k.startswith("_") and k not in ("HOLIDAY_WEEKS",)},
            **self.extra,
        }
        with open(C.RESULTS_DIR / "manifests" / f"{self.script}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return False
