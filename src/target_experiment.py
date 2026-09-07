"""
target_experiment.py — Phase 6: decide what the model should predict.

We have four candidate targets. Rather than pick one from theory, we measure
all four with the same model and the same folds, and compare every result on
the ORIGINAL rupee scale -- because rupees are what the user of the app cares
about.

    A. price                 -> predict rupees directly
    B. log(price)            -> predict, then exp() back to rupees
    C. price / area          -> predict rate per sqft, then multiply by area
    D. log(price / area)     -> predict log rate, then exp() and multiply

Two rules this script exists to enforce:

  RULE 1 - Compare targets on ONE common scale.
      An R-squared computed on log(price) is not comparable to an R-squared
      computed on price. Different targets have different amounts of variance
      to explain, so their "own scale" scores mean different things. Always
      convert predictions back to rupees before scoring.

  RULE 2 - Check the BIAS, not only the error size.
      Taking exp() of an average log is not the same as the average. A log
      model can be accurate on average in log space and still systematically
      under-predict in rupees. MAE will not show you this. Mean bias will.

Usage:
    python src/target_experiment.py data/processed/mumbai_clean.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

RANDOM_STATE = 42        # fixed so the experiment is reproducible
LAKH = 100_000
N_FOLDS = 5
TOP_LOCALITIES = 200     # HistGradientBoosting caps categories at 255

NUM_COLS = ["area", "bedroom_num", "bathroom_num", "balcony_num", "age",
            "age_is_zero", "has_balcony", "latitude", "longitude"]
CAT_COLS = ["loc_grp", "property_type", "furnished"]


def build_features(d: pd.DataFrame):
    """Assemble the feature matrix.

    Note this is a THROWAWAY feature set, good enough to compare targets
    fairly. The real feature engineering is Phase 7 and the real encoding
    decision is Phase 8. Using the same features for all four targets is what
    makes the comparison valid -- not whether the features are optimal.
    """
    top = d.locality.value_counts().head(TOP_LOCALITIES).index
    d = d.assign(loc_grp=np.where(d.locality.isin(top), d.locality, "Other"))

    X = d[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    cat_mask = [c in CAT_COLS for c in X.columns]
    return X, cat_mask


def evaluate(name, X, cat_mask, price, area, forward, inverse):
    """Cross-validate one target definition.

    forward(price, area) -> the values we train on
    inverse(pred, area)  -> those predictions converted back to rupees
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pred_rupees = np.zeros(len(price))
    pred_own = np.zeros(len(price))
    true_own = np.zeros(len(price))

    t0 = time.time()
    for train_idx, test_idx in kf.split(X):
        model = HistGradientBoostingRegressor(
            categorical_features=cat_mask,
            random_state=RANDOM_STATE,
            max_iter=200,
        )
        model.fit(X.iloc[train_idx], forward(price[train_idx], area[train_idx]))

        raw = model.predict(X.iloc[test_idx])
        pred_own[test_idx] = raw
        true_own[test_idx] = forward(price[test_idx], area[test_idx])
        pred_rupees[test_idx] = inverse(raw, area[test_idx])

    return {
        "target": name,
        "MAE_lakh": mean_absolute_error(price, pred_rupees) / LAKH,
        "RMSE_lakh": np.sqrt(mean_squared_error(price, pred_rupees)) / LAKH,
        "R2_rupees": r2_score(price, pred_rupees),
        # The misleading number. Printed on purpose so you can see the trap.
        "R2_own_scale": r2_score(true_own, pred_own),
        # Positive = over-predicts on average, negative = under-predicts.
        "bias_lakh": (pred_rupees - price).mean() / LAKH,
        "seconds": time.time() - t0,
        "_pred": pred_rupees,
    }


def band_report(price, results):
    """Where does each target win? A single average hides this completely."""
    bands = pd.qcut(price, 5,
                    labels=["cheapest 20%", "20-40%", "40-60%",
                            "60-80%", "priciest 20%"])
    frame = pd.DataFrame({"band": bands, "price": price})
    for r in results:
        frame[r["target"]] = r["_pred"]

    print("\nMean absolute error in lakh, by price band")
    print("-" * 72)
    rows = []
    for band, g in frame.groupby("band", observed=True):
        row = {"band": band, "n": len(g),
               "median price": round(g.price.median() / LAKH, 1)}
        for r in results:
            row[r["target"]] = round((g[r["target"]] - g.price).abs().mean() / LAKH, 1)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nMedian error as a PERCENTAGE of the true price")
    print("-" * 72)
    rows = []
    for band, g in frame.groupby("band", observed=True):
        row = {"band": band}
        for r in results:
            row[r["target"]] = round(
                ((g[r["target"]] - g.price).abs() / g.price * 100).median(), 1)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def smearing_demo(X, cat_mask, price):
    """Show that the log model's under-prediction can be corrected.

    Duan's smearing estimator: fit on log(y), then multiply the exponentiated
    prediction by the average of exp(training residuals). It is a one-line fix
    for the fact that exp(average of logs) < average of the values.
    """
    print("\nRetransformation bias, and the fix")
    print("-" * 72)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    plain = np.zeros(len(price))
    fixed = np.zeros(len(price))

    for train_idx, test_idx in kf.split(X):
        model = HistGradientBoostingRegressor(
            categorical_features=cat_mask, random_state=RANDOM_STATE, max_iter=200)
        y_log = np.log(price[train_idx])
        model.fit(X.iloc[train_idx], y_log)

        residuals = y_log - model.predict(X.iloc[train_idx])
        smear = np.exp(residuals).mean()

        p = np.exp(model.predict(X.iloc[test_idx]))
        plain[test_idx] = p
        fixed[test_idx] = p * smear

    print(f"smearing factor           : {smear:.4f}")
    print(f"log target, uncorrected   : MAE {mean_absolute_error(price, plain)/LAKH:6.1f} L   "
          f"bias {(plain - price).mean()/LAKH:+6.1f} L")
    print(f"log target, smearing-fixed: MAE {mean_absolute_error(price, fixed)/LAKH:6.1f} L   "
          f"bias {(fixed - price).mean()/LAKH:+6.1f} L")


def main(path: str) -> int:
    csv = Path(path)
    if not csv.exists():
        print(f"ERROR: file not found: {csv}")
        return 1

    d = pd.read_csv(csv)
    X, cat_mask = build_features(d)
    price = d.price.values.astype(float)
    area = d.area.values.astype(float)

    print(f"{len(d):,} rows, {N_FOLDS}-fold CV, seed {RANDOM_STATE}")
    print("Every metric below is on the original rupee scale.\n")

    specs = [
        ("A. price",            lambda p, a: p,             lambda q, a: q),
        ("B. log(price)",       lambda p, a: np.log(p),     lambda q, a: np.exp(q)),
        ("C. price/area",       lambda p, a: p / a,         lambda q, a: q * a),
        ("D. log(price/area)",  lambda p, a: np.log(p / a), lambda q, a: np.exp(q) * a),
    ]

    results = []
    for name, fwd, inv in specs:
        r = evaluate(name, X, cat_mask, price, area, fwd, inv)
        results.append(r)
        print(f"{r['target']:<20} MAE {r['MAE_lakh']:6.1f}L   "
              f"RMSE {r['RMSE_lakh']:7.1f}L   R2(rupees) {r['R2_rupees']:6.3f}   "
              f"R2(own scale) {r['R2_own_scale']:6.3f}   "
              f"bias {r['bias_lakh']:+6.1f}L   [{r['seconds']:.0f}s]")

    band_report(price, results)
    smearing_demo(X, cat_mask, price)

    best = min(results, key=lambda r: r["RMSE_lakh"])
    print(f"\nLowest RMSE on the rupee scale: {best['target']}")
    print("Read the band tables before accepting that as the answer -- the "
          "overall winner is not the winner in every price band.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
