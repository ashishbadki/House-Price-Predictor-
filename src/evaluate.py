"""
evaluate.py — Phases 12 & 13: measure models honestly.

WHY THIS IS ITS OWN MODULE
--------------------------
Every later phase needs the same set of numbers. Writing them once means
Phase 16 (error analysis), Phase 18 (final choice) and Phase 22 (tests) all
score things the same way, so their results can be compared.

WHY SO MANY METRICS
-------------------
There is no single "accuracy" for a regression problem, and different metrics
crown different winners. On this data, with the same three models:

    MAE picks       XGBoost
    RMSE and R2 pick HistGradientBoosting
    MAPE, median APE, bias and hit-rate all pick Random Forest

You cannot avoid the choice by looking at one number. You have to decide which
kind of wrongness matters for your application, then pick the metric that
measures it.

Usage:
    python src/evaluate.py data/processed/train.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LAKH = 100_000
N_FOLDS = 5

NUM_COLS = ["area", "bedroom_num", "bathroom_num", "balcony_num", "age",
            "age_is_zero", "has_balcony", "latitude", "longitude",
            "dist_nariman_km", "dist_bkc_km", "dist_cbd_km",
            "rooms_total", "bath_per_bed"]
CAT_COLS = ["locality_grouped", "property_type", "furnished"]


# ===========================================================================
# THE METRICS
# ===========================================================================

def regression_report(y_true, y_pred) -> dict:
    """Every metric worth looking at, in one place.

    MAE — Mean Absolute Error.
        Average size of the mistake, in rupees, ignoring direction.
        MAE = 42 lakh means: on a typical property we are 42 lakh out.
        Easy to explain to anyone. Treats a 10 lakh error on a 50 lakh flat
        the same as a 10 lakh error on a 50 crore flat.

    RMSE — Root Mean Squared Error.
        Errors are squared before averaging, so big ones dominate. Always
        larger than MAE. Use it when one huge miss is much worse than several
        small ones. Here RMSE (145) is 3.4x MAE (42), which tells you a
        handful of properties are very badly predicted.

    R2 — proportion of the variation explained.
        1.0 is perfect. 0.0 means no better than always predicting the mean.
        Negative means worse than that. R2 = 0.82 means we account for 82%
        of the differences between properties.
        Weakness: it depends on how spread out your data is, so it is not
        comparable across datasets, and it hides the rupee amounts entirely.

    MAPE — Mean Absolute Percentage Error.
        The error as a percentage of the true price. This is usually what a
        user actually feels: being 20% off is bad whether the flat is 50 lakh
        or 5 crore. Weakness: it explodes on very cheap items, because a small
        rupee error is a huge percentage of a small price.

    Median APE — the middle percentage error.
        More robust than MAPE. Half of all predictions are better than this.
        The single most honest number to put in front of a user.

    Bias — mean(prediction - actual), keeping the sign.
        The only metric here that shows DIRECTION. MAE, RMSE, R2 and MAPE all
        throw the sign away, so a model that is always 5 lakh low looks
        identical to one that is randomly 5 lakh out. Always check this.

    Hit rate — the share of predictions within 10% / 20% of the truth.
        The most concrete statement you can make to a non-technical person:
        "4 out of 10 estimates land within 10% of the asking price."
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true * 100

    return {
        "MAE_lakh": mean_absolute_error(y_true, y_pred) / LAKH,
        "RMSE_lakh": np.sqrt(mean_squared_error(y_true, y_pred)) / LAKH,
        "R2": r2_score(y_true, y_pred),
        "MAPE_pct": ape.mean(),
        "MedAPE_pct": np.median(ape),
        "bias_lakh": (y_pred - y_true).mean() / LAKH,
        "within_10pct": (ape <= 10).mean() * 100,
        "within_20pct": (ape <= 20).mean() * 100,
    }


def why_not_accuracy() -> None:
    """Why "accuracy" does not exist for this problem.

    Accuracy is (number of correct answers) / (total answers). It needs a
    clear right and wrong.

    Predicting a price has no exact right answer. If a flat sold for
    1,20,00,000 and we predict 1,19,80,000, is that "correct"? By exact match,
    no -- and by exact match our accuracy would be 0%, forever, on every model.

    You could invent a rule ("correct if within 10%"), and that is exactly what
    `within_10pct` above is. But notice what it throws away: a prediction that
    is 11% off and one that is 300% off both count as simply "wrong". MAE and
    RMSE keep that information.

    So: report the hit rate because people understand it, but never choose a
    model on it alone.
    """


def report_by_band(y_true, y_pred, n_bands: int = 5) -> pd.DataFrame:
    """Break the error down by how expensive the property is.

    A single average hides everything. A model can look fine overall while
    being useless on the cheapest or most expensive quarter of the market --
    and those are real users.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    labels = ["cheapest 20%", "20-40%", "40-60%", "60-80%", "priciest 20%"][:n_bands]
    bands = pd.qcut(y_true, n_bands, labels=labels)

    frame = pd.DataFrame({"band": bands, "actual": y_true, "pred": y_pred})
    rows = []
    for band, g in frame.groupby("band", observed=True):
        err = g.pred - g.actual
        ape = (err.abs() / g.actual * 100)
        rows.append({
            "band": band,
            "n": len(g),
            "median_price_L": g.actual.median() / LAKH,
            "MAE_L": err.abs().mean() / LAKH,
            "MedAPE_pct": ape.median(),
            "bias_L": err.mean() / LAKH,
        })
    return pd.DataFrame(rows).round(2)


# ===========================================================================
# PHASE 13 — CROSS VALIDATION
# ===========================================================================

def make_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=30, sparse_output=False), CAT_COLS),
    ])


def cross_val_predict_grouped(model_fn, df):
    """K-fold cross-validation, done properly.

    HOW K-FOLD WORKS
        Cut the data into 5 parts. Train on 4, score on the 1 left out.
        Repeat 5 times so every row gets scored exactly once, by a model that
        never saw it. Average the 5 scores.

    WHY IT BEATS A SINGLE SPLIT
        Every row is used for training (4 times) and for scoring (once), so
        nothing is wasted. And the answer is an average of 5 numbers rather
        than one number, so it moves around far less.

    WHY GroupKFold, NOT KFold
        Phase 9: a building must not appear on both sides. GroupKFold takes
        `groups` and keeps each building whole.

    WHY THE PIPELINE IS BUILT INSIDE THE LOOP
        So the imputer and the encoder refit on each fold's training rows only.
        Fitting them once outside the loop is the Phase 8 mistake, worth 6.85
        lakh of self-deception.
    """
    X = df[NUM_COLS + CAT_COLS]
    y = (df.price / df.area).values
    price = df.price.values.astype(float)
    area = df.area.values.astype(float)
    groups = df.building.values

    pred = np.zeros(len(df))
    fold_scores = []
    for train_idx, val_idx in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        pipe = Pipeline([("pre", make_preprocessor()), ("model", model_fn())])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        p = pipe.predict(X.iloc[val_idx]) * area[val_idx]
        pred[val_idx] = p
        fold_scores.append(mean_absolute_error(price[val_idx], p) / LAKH)

    return pred, np.array(fold_scores)


def single_split_instability(df, model_fn, n_seeds: int = 10):
    """Show how much one 80/20 split can vary, using the SAME model.

    This is the argument for cross-validation, made with numbers instead of
    assertion. If a single split swings more than the gap between your
    candidate models, then a single split cannot tell them apart.
    """
    X = df[NUM_COLS + CAT_COLS]
    y = (df.price / df.area).values
    price = df.price.values.astype(float)
    area = df.area.values.astype(float)
    groups = df.building.values

    scores = []
    for seed in range(n_seeds):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                        random_state=seed).split(X, y, groups))
        pipe = Pipeline([("pre", make_preprocessor()), ("model", model_fn())])
        pipe.fit(X.iloc[tr], y[tr])
        scores.append(mean_absolute_error(
            price[te], pipe.predict(X.iloc[te]) * area[te]) / LAKH)
    return np.array(scores)


# ===========================================================================

def main(path: str) -> int:
    src = Path(path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1
    df = pd.read_csv(src)
    print(f"{len(df):,} training rows | {N_FOLDS}-fold GroupKFold by building\n")

    models = {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1,
            random_state=RANDOM_STATE),
        "HistGB": lambda: HistGradientBoostingRegressor(
            max_iter=300, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400, learning_rate=0.08, max_depth=7, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE,
            tree_method="hist")
    except ImportError:
        pass

    print("=" * 78)
    print("PHASE 12 — EVERY METRIC, SAME PREDICTIONS")
    print("=" * 78)
    all_preds, rows = {}, []
    for name, fn in models.items():
        t0 = time.time()
        pred, folds = cross_val_predict_grouped(fn, df)
        all_preds[name] = pred
        r = regression_report(df.price.values, pred)
        r["model"] = name
        r["fold_sd"] = folds.std()
        rows.append(r)
        print(f"  {name} done [{time.time()-t0:.0f}s]")

    table = pd.DataFrame(rows).set_index("model").round(2)
    print()
    print(table.to_string())

    print("\nWhich model wins, according to each metric:")
    lower_is_better = {"MAE_lakh", "RMSE_lakh", "MAPE_pct", "MedAPE_pct", "fold_sd"}
    for col in table.columns:
        if col == "bias_lakh":
            winner = table.bias_lakh.abs().idxmin()
        elif col in lower_is_better:
            winner = table[col].idxmin()
        else:
            winner = table[col].idxmax()
        print(f"  {col:<14} -> {winner}")

    print("\n" + "=" * 78)
    print("ERROR BY PRICE BAND — best model by median percentage error")
    print("=" * 78)
    best = table.MedAPE_pct.idxmin()
    print(f"({best})\n")
    print(report_by_band(df.price.values, all_preds[best]).to_string(index=False))

    print("\n" + "=" * 78)
    print("PHASE 13 — WHY CROSS-VALIDATION, NOT ONE SPLIT")
    print("=" * 78)
    print("Same model. Same data. Only the random seed of the split changes.\n")
    scores = single_split_instability(df, models["HistGB"], n_seeds=10)
    for i, s in enumerate(scores):
        print(f"  seed {i}: MAE {s:6.2f} L")
    print(f"\n  lowest {scores.min():.2f} L   highest {scores.max():.2f} L"
          f"   spread {np.ptp(scores):.2f} L")

    _, folds = cross_val_predict_grouped(models["HistGB"], df)
    print(f"\n  5-fold CV mean: {folds.mean():.2f} L")
    print(f"  spread across folds: {folds.std():.2f} L")
    print(f"  standard error of the CV mean: {folds.std()/np.sqrt(N_FOLDS):.2f} L")

    gap = table.MAE_lakh.max() - table.MAE_lakh.min()
    print(f"\n  Our three models differ by {gap:.2f} L.")
    print(f"  One split alone swings by {np.ptp(scores):.2f} L.")
    print("  A single split cannot tell these models apart. CV is not optional.")

    table.to_csv("reports/metric_comparison.csv")
    print("\nSaved: reports/metric_comparison.csv")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
