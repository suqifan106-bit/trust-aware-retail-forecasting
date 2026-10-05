"""Data construction, feature engineering, splitting and preprocessing."""
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------- build
def load_raw():
    train = pd.read_csv(C.DATA_RAW / "train.csv")
    features = pd.read_csv(C.DATA_RAW / "features.csv")
    stores = pd.read_csv(C.DATA_RAW / "stores.csv")
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])
    return train, features, stores


def build_full_dataset():
    """Merge the three Kaggle files and add calendar, promotion-indicator and lag features."""
    train, features, stores = load_raw()
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left").merge(stores, on="Store", how="left")
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    md = [f"MarkDown{i}" for i in range(1, 6)]
    for c in md:
        df[f"{c}_recorded"] = df[c].notna().astype(int)
        df[c] = df[c].fillna(0.0)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    for flag, dates in C.HOLIDAY_WEEKS.items():
        df[flag] = df["Date"].isin(pd.to_datetime(dates)).astype(int)
    df = df.sort_values(C.KEYS).reset_index(drop=True)
    df = add_lag_features(df)
    return df


def add_lag_features(df):
    """Exact-date lags (Date - 7k days within the same Store-Dept) and rolling stats of past weeks.

    Rolling statistics use the previous available rows of the series (shift(1)), so they only
    ever contain information from strictly earlier dates.
    """
    base = df[C.KEYS + [C.TARGET]].copy()
    for k in [1, 2, 4, 8, 52]:
        lag = base.copy()
        lag["Date"] = lag["Date"] + pd.Timedelta(days=7 * k)
        lag = lag.rename(columns={C.TARGET: f"lag_{k}"})
        df = df.merge(lag, on=C.KEYS, how="left")
    g = df.groupby(["Store", "Dept"])[C.TARGET]
    shifted = g.shift(1)
    df["roll_mean_4"] = shifted.groupby([df["Store"], df["Dept"]]).transform(lambda s: s.rolling(4, min_periods=2).mean())
    df["roll_std_4"] = shifted.groupby([df["Store"], df["Dept"]]).transform(lambda s: s.rolling(4, min_periods=2).std())
    df["roll_mean_12"] = shifted.groupby([df["Store"], df["Dept"]]).transform(lambda s: s.rolling(12, min_periods=4).mean())
    df["lag_1_available"] = df["lag_1"].notna().astype(int)
    return df


def dataset_path():
    return C.RESULTS_DIR / "predictions" / "full_features.parquet"


def load_dataset():
    df = pd.read_parquet(dataset_path())
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# --------------------------------------------------------------------- split
def split_at_origin(df, origin, test_weeks=C.TEST_WEEKS):
    """Train = all rows before origin; test = the first `test_weeks` distinct weeks from origin on."""
    origin = pd.Timestamp(origin)
    train = df[df["Date"] < origin]
    test_dates = sorted(df.loc[df["Date"] >= origin, "Date"].unique())[:test_weeks]
    test = df[df["Date"].isin(test_dates)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# -------------------------------------------------------------- preprocess
class Preprocessor:
    """Minimal, name-preserving preprocessing: one-hot `Type`, median imputation fitted on train.

    Tree models receive unscaled inputs; `scale=True` adds standardisation for linear models.
    Feature names stay human readable (no `num__` / `cat__` prefixes).
    """

    def __init__(self, features, scale=False):
        self.features = list(features)
        self.scale = scale

    def _frame(self, df):
        x = df[self.features].copy()
        if "Type" in x.columns:
            for t in ["A", "B", "C"]:
                x[f"Type_{t}"] = (x["Type"] == t).astype(int)
            x = x.drop(columns=["Type"])
        return x.astype(float)

    def fit(self, df):
        x = self._frame(df)
        self.columns_ = list(x.columns)
        self.medians_ = x.median().fillna(0.0)  # a column can be entirely missing in early folds (e.g. lag_52)
        if self.scale:
            self.mu_ = x.fillna(self.medians_).mean()
            self.sd_ = x.fillna(self.medians_).std().replace(0, 1.0)
        return self

    def transform(self, df):
        x = self._frame(df)[self.columns_].fillna(self.medians_)
        if self.scale:
            x = (x - self.mu_) / self.sd_
        return x

    def fit_transform(self, df):
        return self.fit(df).transform(df)
