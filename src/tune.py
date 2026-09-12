"""
tune.py — Phase 15: hyperparameter tuning.

WHAT A HYPERPARAMETER IS
------------------------
Two kinds of number live inside a model.

  Parameters      the model LEARNS them from data. The split points inside a
                  tree, the coefficients in a regression. You never set these.

  Hyperparameters YOU set them, before training starts. How many trees. How
                  deep. How fast it learns. The model cannot learn them,
                  because they control HOW it learns.

Until now we used whatever numbers I typed in Phase 11. Some were reasonable
guesses. This script replaces guessing with searching.

GRID SEARCH vs RANDOMIZED SEARCH
--------------------------------
GridSearchCV tries every combination you list. With 4 settings of 5 values
each that is 5^4 = 625 combinations, times 5 folds = 3,125 fits. Exhaustive,
and usually unaffordable.

RandomizedSearchCV samples N random combinations from ranges you give. You
choose N, so you choose the budget. Counter-intuitively it usually finds an
equally good answer, because most hyperparameters barely matter and a random
search spends its budget spread across the ones that do, instead of
exhaustively exploring the ones that do not.

We use RandomizedSearchCV.

THE RULE THAT MATTERS MOST
--------------------------
The search scores candidates with GroupKFold on the TRAINING data only.
`test.csv` is not opened by this file.

If you tune against the test set -- try 200 settings, keep whichever scores
best on test -- then the test score stops being an estimate of real
performance. You have fitted the test set by hand, 200 attempts at a time.
It will look great and mean nothing.

Usage:
    python src/tune.py data/processed/train.csv
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (FEATURES, build_pipeline, get_groups, to_price,
                           to_target)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LAKH = 100_000

# The search uses 3 folds to stay affordable; the winner is then re-scored
# with the full 5 folds so it is comparable to every earlier number.
SEARCH_FOLDS = 3
FINAL_FOLDS = 5
N_CANDIDATES = 12


def mae_lakh(estimator, X, y_rate, area, price) -> float:
    pred = to_price(estimator.predict(X), area)
    return mean_absolute_error(price, pred) / LAKH


# ---------------------------------------------------------------------------
# SEARCH SPACES
#
# Each entry says WHAT the setting does and WHICH DIRECTION overfits, because
# a range chosen without understanding that is just noise.
# ---------------------------------------------------------------------------

def search_spaces():
    spaces = {}

    spaces["RandomForest"] = (
        RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        False,
        {
            # More trees is always at least as good, just slower. It cannot
            # overfit -- averaging more trees only reduces variance. So this
            # is a budget decision, not an accuracy one.
            "model__n_estimators": randint(80, 200),
            # Depth: deeper trees fit finer detail and overfit sooner.
            # None means grow until leaves are pure.
            "model__max_depth": [None, 15, 25],
            # The strongest anti-overfitting control here. A leaf holding 1
            # row has memorised that row. Larger = smoother, safer.
            "model__min_samples_leaf": randint(1, 12),
            # How many features each split may consider. Lower means trees
            # differ more from each other, which is what makes the average
            # work.
            "model__max_features": ["sqrt", 0.4, 0.6, 1.0],
        },
    )

    spaces["HistGB"] = (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        False,
        {
            # Boosting builds trees in sequence, each fixing the last one's
            # mistakes. More rounds keeps fixing -- eventually it starts
            # fixing noise. This one CAN overfit.
            "model__max_iter": randint(200, 700),
            # How big a correction each round applies. Smaller is safer but
            # needs more rounds. Trades off directly against max_iter.
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_leaf_nodes": randint(15, 80),
            "model__min_samples_leaf": randint(10, 60),
            # L2 penalty on leaf values. Pulls extreme predictions towards
            # the middle.
            "model__l2_regularization": loguniform(1e-3, 10),
        },
    )

    try:
        from xgboost import XGBRegressor
        spaces["XGBoost"] = (
            XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE,
                         tree_method="hist"),
            False,
            {
                "model__n_estimators": randint(200, 800),
                "model__learning_rate": loguniform(0.02, 0.2),
                "model__max_depth": randint(4, 10),
                # Each tree sees only this fraction of the rows / columns.
                # Randomness that makes the trees disagree, which is what an
                # ensemble needs.
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__min_child_weight": randint(1, 12),
                "model__reg_lambda": loguniform(0.1, 20),
            },
        )
    except ImportError:
        print("(xgboost not installed -- skipping)")

    return spaces


def tune_one(name, model, scale, grid, X, y_rate, groups, area, price):
    print(f"\n{'-'*72}\n{name}\n{'-'*72}")

    pipe = build_pipeline(model, scale_numbers=scale)

    # scoring: sklearn maximises, so we use the NEGATIVE mean absolute error
    # on the rate. Note this scores the rate, not rupees -- convenient for the
    # search. We re-score the winner in rupees below, which is what we report.
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=N_CANDIDATES,
        cv=GroupKFold(n_splits=SEARCH_FOLDS),
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=1,          # the models already use every core internally
        refit=True,
        verbose=0,
    )

    t0 = time.time()
    search.fit(X, y_rate, groups=groups)
    elapsed = time.time() - t0

    print(f"  {N_CANDIDATES} candidates x {SEARCH_FOLDS} folds "
          f"= {N_CANDIDATES*SEARCH_FOLDS} fits in {elapsed:.0f}s")
    print("  best settings found:")
    for k, v in sorted(search.best_params_.items()):
        pretty = f"{v:.4f}" if isinstance(v, float) else v
        print(f"    {k.replace('model__',''):<22} {pretty}")

    # Re-score the winner with 5 folds, in rupees, so the number is directly
    # comparable to Phases 11-13.
    pred = np.zeros(len(X))
    for tr, va in GroupKFold(n_splits=FINAL_FOLDS).split(X, y_rate, groups):
        p = build_pipeline(search.best_estimator_.named_steps["model"],
                           scale_numbers=scale)
        p.fit(X.iloc[tr], y_rate[tr])
        pred[va] = to_price(p.predict(X.iloc[va]), area[va])

    mae = mean_absolute_error(price, pred) / LAKH
    ape = np.abs(pred - price) / price * 100
    print(f"  5-fold MAE (rupees): {mae:.2f} L   median APE: {np.median(ape):.2f}%")

    return {
        "model": name,
        "MAE_lakh": mae,
        "MedAPE_pct": float(np.median(ape)),
        "bias_lakh": float((pred - price).mean() / LAKH),
        "search_seconds": elapsed,
        "best_params": {k.replace("model__", ""): (
            float(v) if isinstance(v, (np.floating, float)) else
            int(v) if isinstance(v, (np.integer,)) else v)
            for k, v in search.best_params_.items()},
    }


def main(path: str) -> int:
    src = Path(path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1

    df = pd.read_csv(src)
    X = df[FEATURES]
    y_rate = to_target(df)
    groups = get_groups(df)
    area = df["area"].values.astype(float)
    price = df["price"].values.astype(float)

    print(f"{len(df):,} training rows | {df.building.nunique():,} buildings")
    print("test.csv is NOT opened by this script.")
    print(f"Search: {N_CANDIDATES} random candidates, "
          f"{SEARCH_FOLDS}-fold GroupKFold. Winner re-scored on "
          f"{FINAL_FOLDS} folds.")

    # Phase 11-13 numbers, for comparison.
    BEFORE = {"RandomForest": 42.34, "HistGB": 42.70, "XGBoost": 42.18}

    results = []
    for name, (model, scale, grid) in search_spaces().items():
        results.append(tune_one(name, model, scale, grid,
                                X, y_rate, groups, area, price))

    print(f"\n{'='*72}\nBEFORE AND AFTER TUNING\n{'='*72}")
    rows = []
    for r in results:
        before = BEFORE.get(r["model"], float("nan"))
        rows.append({
            "model": r["model"],
            "MAE before": round(before, 2),
            "MAE after": round(r["MAE_lakh"], 2),
            "gain L": round(before - r["MAE_lakh"], 2),
            "gain %": round((before - r["MAE_lakh"]) / before * 100, 1),
            "MedAPE %": round(r["MedAPE_pct"], 2),
            "bias L": round(r["bias_lakh"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    out = Path("reports/best_params.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    print("\nRemember: these are still cross-validation numbers on the "
          "training data.\nThe test set has not been touched. That happens "
          "once, in Phase 18.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
