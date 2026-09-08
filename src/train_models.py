"""
train_models.py — Phases 10 & 11: baselines first, then real models.

RULES THIS SCRIPT ENFORCES
--------------------------
1. Only `train.csv` is read. `test.csv` is not opened. It stays sealed
   until Phase 18.
2. Every score comes from GroupKFold with `groups=building`, so no building
   is ever in both the fitting and the scoring half of a fold.
3. Every step that learns anything -- imputer, scaler, encoder, model -- is
   inside a Pipeline, so it refits on each fold's training rows only.
4. Baselines run first. A model that cannot beat a simple rule has not
   earned its complexity.

Usage:
    python src/train_models.py data/processed/train.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LAKH = 100_000
N_FOLDS = 5
MIN_CATEGORY_ROWS = 30

NUM_COLS = ["area", "bedroom_num", "bathroom_num", "balcony_num", "age",
            "age_is_zero", "has_balcony", "latitude", "longitude",
            "dist_nariman_km", "dist_bkc_km", "dist_cbd_km",
            "rooms_total", "bath_per_bed"]
CAT_COLS = ["locality_grouped", "property_type", "furnished"]


# ===========================================================================
# PHASE 10 — BASELINES
#
# A baseline is a deliberately simple rule. Its job is to answer:
# "did the machine learning actually add anything?"
#
# Without one, an MAE of 41 lakh is just a number. Against a baseline it
# becomes "35% better than what a person with a calculator would do".
# ===========================================================================

class GlobalMedianRate(BaseEstimator, RegressorMixin):
    """Predict the same price-per-sqft for every property.

    The dumbest possible model that is not zero. Any real model must beat
    this by a wide margin or something is badly wrong.
    """

    def fit(self, X, y):
        self.rate_ = np.median(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.rate_)


class LocalityMedianRate(BaseEstimator, RegressorMixin):
    """Predict each locality's own median price-per-sqft.

    THIS IS THE BASELINE THAT MATTERS. It is what an experienced broker does
    in their head: "Andheri is running about 18,000 a foot, the flat is 900
    feet, so roughly 1.6 crore."

    If our model cannot beat this, the whole project is an expensive way to
    reproduce a lookup table.

    Note it learns its medians in `fit`, from training rows only. Computing
    them over the whole dataset would leak -- the same mistake Phase 8
    measured at 6.85 lakh.
    """

    def fit(self, X, y):
        s = pd.Series(y, index=X.index)
        self.medians_ = s.groupby(X["locality_grouped"]).median()
        self.fallback_ = np.median(y)      # for a locality never seen in fit
        return self

    def predict(self, X):
        return X["locality_grouped"].map(self.medians_).fillna(self.fallback_).values


# ===========================================================================
# PREPROCESSING — one definition, shared by every model
# ===========================================================================

def make_preprocessor(scale_numbers: bool) -> ColumnTransformer:
    """Impute, optionally scale, and one-hot encode.

    `scale_numbers` is True for linear models and False for trees.

    Linear models need scaling because they compare coefficients across
    features: without it, `area` (values in the hundreds) and `has_balcony`
    (0 or 1) are treated as if a one-unit change means the same thing in both.
    Trees only ever ask "is this value above or below X?", so the scale of a
    column makes no difference to them.

    The imputer handles the 151 rows with no coordinates. `strategy="median"`
    rather than "mean" because a median is not dragged around by outliers.
    """
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale_numbers:
        numeric.append(("scale", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric), NUM_COLS),
        # Phase 8 decision. `infrequent_if_exist` means a locality the model
        # has never seen maps to an "infrequent" bucket instead of crashing --
        # which is exactly what the Streamlit app needs.
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=MIN_CATEGORY_ROWS,
                              sparse_output=False), CAT_COLS),
    ])


def evaluate(name, model, df, scale_numbers=False, raw=False):
    """Score one model with GroupKFold, reporting on the rupee scale.

    `raw=True` skips the preprocessor -- used for the baselines, which read
    columns directly and need no encoding.
    """
    X = df[NUM_COLS + CAT_COLS]
    y = (df.price / df.area).values          # Phase 6 target: rate per sqft
    price = df.price.values.astype(float)
    area = df.area.values.astype(float)
    groups = df.building.values              # Phase 9: never split a building

    cv = GroupKFold(n_splits=N_FOLDS)
    pred = np.zeros(len(df))
    fold_maes = []
    t0 = time.time()

    for train_idx, val_idx in cv.split(X, y, groups):
        pipe = (model if raw
                else Pipeline([("pre", make_preprocessor(scale_numbers)),
                               ("model", model)]))
        pipe.fit(X.iloc[train_idx], y[train_idx])
        p = pipe.predict(X.iloc[val_idx]) * area[val_idx]
        pred[val_idx] = p
        fold_maes.append(mean_absolute_error(price[val_idx], p) / LAKH)

    return {
        "model": name,
        "MAE": mean_absolute_error(price, pred) / LAKH,
        "RMSE": np.sqrt(mean_squared_error(price, pred)) / LAKH,
        "R2": r2_score(price, pred),
        # Spread across folds. A model with a good average but a wide spread
        # is unstable -- it got lucky on some folds. See Phase 18.
        "fold_sd": np.std(fold_maes),
        "seconds": time.time() - t0,
    }


def show(results) -> None:
    frame = pd.DataFrame(results)
    frame = frame[["model", "MAE", "RMSE", "R2", "fold_sd", "seconds"]]
    frame["MAE"] = frame.MAE.round(2)
    frame["RMSE"] = frame.RMSE.round(2)
    frame["R2"] = frame.R2.round(4)
    frame["fold_sd"] = frame.fold_sd.round(2)
    frame["seconds"] = frame.seconds.round(0).astype(int)
    print(frame.to_string(index=False))


def main(path: str) -> int:
    src = Path(path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1
    df = pd.read_csv(src)
    print(f"Training rows: {len(df):,} | buildings: {df.building.nunique():,}")
    print(f"{N_FOLDS}-fold GroupKFold, grouped by building. "
          f"test.csv is NOT opened.\n")

    print("=" * 78)
    print("PHASE 10 — BASELINES")
    print("=" * 78)
    baselines = [
        evaluate("B1 global median rate", GlobalMedianRate(), df, raw=True),
        evaluate("B2 locality median rate", LocalityMedianRate(), df, raw=True),
        evaluate("B3 linear regression", LinearRegression(), df, scale_numbers=True),
    ]
    show(baselines)
    best_base = min(baselines, key=lambda r: r["MAE"])
    print(f"\nBest baseline: {best_base['model']} at {best_base['MAE']:.2f} L")
    print("Everything below must beat this to justify its complexity.")

    print("\n" + "=" * 78)
    print("PHASE 11 — MACHINE LEARNING MODELS")
    print("=" * 78)
    models = [
        ("M1 Ridge", Ridge(alpha=1.0), True),
        ("M2 Lasso", Lasso(alpha=10.0, max_iter=5000), True),
        ("M3 Decision Tree", DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE), False),
        ("M4 Random Forest", RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1,
            random_state=RANDOM_STATE), False),
        ("M5 Extra Trees", ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1,
            random_state=RANDOM_STATE), False),
        ("M6 Gradient Boosting", GradientBoostingRegressor(
            n_estimators=200, random_state=RANDOM_STATE), False),
        ("M7 HistGradientBoosting", HistGradientBoostingRegressor(
            max_iter=300, random_state=RANDOM_STATE), False),
    ]

    try:
        from xgboost import XGBRegressor
        models.append(("M8 XGBoost", XGBRegressor(
            n_estimators=400, learning_rate=0.08, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE, tree_method="hist"), False))
    except ImportError:
        print("(xgboost not installed -- skipping M8)\n")

    results = []
    for name, model, scale in models:
        r = evaluate(name, model, df, scale_numbers=scale)
        results.append(r)
        print(f"  done {name:<26} MAE {r['MAE']:6.2f} L  [{r['seconds']:.0f}s]")

    print()
    show(results)

    best = min(results, key=lambda r: r["MAE"])
    gain = (best_base["MAE"] - best["MAE"]) / best_base["MAE"] * 100
    print(f"\nBest model : {best['model']} at {best['MAE']:.2f} L")
    print(f"Best baseline: {best_base['model']} at {best_base['MAE']:.2f} L")
    print(f"Improvement over the baseline: {gain:.1f}%")
    print("\nDo not pick the winner on MAE alone. Read the fold_sd column too "
          "(stability) and the seconds column (cost). Phase 18 decides.")

    pd.DataFrame(baselines + results).to_csv("reports/model_comparison.csv",
                                             index=False)
    print("Saved: reports/model_comparison.csv")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
