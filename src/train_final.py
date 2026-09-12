"""
train_final.py — Phases 18 & 19: choose the final model, then save it.

THIS SCRIPT OPENS test.csv. It is the only script in the project that does,
and it should be run once. Every time you run it, look at the test score, and
then change something, you spend a little of the test set's honesty.

WHAT PHASE 18 DECIDED, AND WHY
------------------------------
Three candidates survived Phase 15. On accuracy they are nearly tied, so
accuracy alone cannot choose. Here is everything we measured:

                          RandomForest    HistGB      XGBoost
  median APE (primary)        13.18%      13.48%      13.57%
  bias                        -3.94 L     -5.34 L     -5.57 L
  MAE                         42.02 L     42.23 L     42.09 L
  spread across folds          2.35        2.66        2.90
  saved file                  54.8 MB      1.6 MB      1.2 MB
  memory when loaded         162.8 MB      3.6 MB      3.4 MB
  time for one prediction     47.8 ms      6.3 ms      3.9 ms
  needs an extra package         no          no         yes

Random Forest is genuinely the most accurate. We checked this properly with a
PAIRED comparison -- the same five folds for both models -- and it won 5 out
of 5, mean gap 0.30 percentage points. Comparing two averages would have
called that noise; comparing fold by fold shows it is consistent.

But look at what 0.30 percentage points buys. On a Rs 1 crore flat, the typical
error moves from Rs 13.48 lakh to Rs 13.18 lakh. A difference of Rs 30,000 on a
crore. No user will ever notice it.

And look at what it costs: 45x the memory, 34x the file size, 7.6x the
prediction time. Streamlit Cloud's free tier gives about 1 GB of RAM, and the
app also needs pandas and a dataframe. A 163 MB model is a genuine deployment
risk for an improvement nobody can perceive.

FINAL CHOICE: HistGradientBoostingRegressor.

  - accuracy statistically behind Random Forest, practically identical
  - ships inside scikit-learn, so one less package to install and pin
  - 1.6 MB on disk and 3.6 MB in memory
  - 6.3 ms per prediction, fast enough that the app feels instant

If this were a nightly batch job with no memory limit, Random Forest would
win. It is a web app on a free tier, so it does not. The right model depends
on where it has to run.

Usage:
    python src/train_final.py data/processed/train.csv data/processed/test.csv
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import FEATURES, build_pipeline, to_price, to_target

LAKH = 100_000
CRORE = 10_000_000
RANDOM_STATE = 42
MODEL_PATH = Path("models/house_price_model.joblib")
META_PATH = Path("models/model_card.json")
LOCALITY_PATH = Path("models/locality_reference.json")

# The Phase 15 tuned settings for the chosen model.
FINAL_MODEL_PARAMS = dict(
    max_iter=443,
    learning_rate=0.0482,
    max_leaf_nodes=78,
    min_samples_leaf=12,
    l2_regularization=0.2975,
    random_state=RANDOM_STATE,
)


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true * 100
    return {
        "MAE_lakh": round(mean_absolute_error(y_true, y_pred) / LAKH, 2),
        "RMSE_lakh": round(np.sqrt(mean_squared_error(y_true, y_pred)) / LAKH, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MedAPE_pct": round(float(np.median(ape)), 2),
        "bias_lakh": round(float((y_pred - y_true).mean()) / LAKH, 2),
        "within_10pct": round(float((ape <= 10).mean() * 100), 1),
        "within_20pct": round(float((ape <= 20).mean() * 100), 1),
    }


def segment_report(df, pred) -> dict:
    """Per-segment test performance. This is what the app's warnings are
    based on, so it belongs in the saved model card rather than only in a
    report nobody reads at runtime."""
    d = df.assign(pred=pred, ape=np.abs(pred - df.price) / df.price * 100,
                  err=pred - df.price)
    out = {}
    for label, mask in [
        ("all", pd.Series(True, index=d.index)),
        ("under_2cr", d.price < 2 * CRORE),
        ("2cr_to_10cr", (d.price >= 2 * CRORE) & (d.price < 10 * CRORE)),
        ("over_10cr", d.price >= 10 * CRORE),
        ("area_over_2500", d.area > 2500),
        ("bhk_4_plus", d.bedroom_num >= 4),
    ]:
        g = d[mask]
        if len(g) < 5:
            continue
        out[label] = {
            "n": int(len(g)),
            "MedAPE_pct": round(float(g.ape.median()), 2),
            "bias_lakh": round(float(g.err.mean()) / LAKH, 1),
        }
    return out


def save_locality_reference(train: pd.DataFrame) -> None:
    """Write the lookup table the Streamlit app needs.

    The app asks the user for a locality NAME. The model needs latitude,
    longitude and the three distance features. Something has to bridge that
    gap, and it must use the same numbers the model trained on -- so we build
    it here, at the same moment we save the model, from the same data.

    Building it anywhere else risks the app and the model quietly disagreeing
    about where Andheri is.
    """
    ref = {}
    for name, g in train.groupby("locality_grouped"):
        ref[name] = {
            "n": int(len(g)),
            # Median, not mean: one row with a bad coordinate should not be
            # able to move a whole locality's position.
            "lat": None if g.latitude.isna().all()
                   else round(float(g.latitude.median()), 6),
            "lon": None if g.longitude.isna().all()
                   else round(float(g.longitude.median()), 6),
            # Shown in the app next to the estimate, so the user has
            # something to judge the number against.
            "median_rate": int(round(float((g.price / g.area).median()))),
            "median_price_lakh": round(float(g.price.median()) / LAKH, 1),
        }
    LOCALITY_PATH.write_text(json.dumps(ref, indent=1, sort_keys=True))
    no_coords = [k for k, v in ref.items() if v["lat"] is None]
    print(f"Saved: {LOCALITY_PATH}  ({len(ref)} localities"
          + (f", {len(no_coords)} with no coordinates)" if no_coords else ")"))


def main(train_path: str, test_path: str) -> int:
    train_file, test_file = Path(train_path), Path(test_path)
    for f in (train_file, test_file):
        if not f.exists():
            print(f"ERROR: file not found: {f}")
            return 1

    train = pd.read_csv(train_file)
    print("=" * 74)
    print("PHASE 18 — FIT THE CHOSEN MODEL ON ALL TRAINING DATA")
    print("=" * 74)
    print(f"Model: HistGradientBoostingRegressor")
    print(f"Training rows: {len(train):,}  buildings: {train.building.nunique():,}")

    pipe = build_pipeline(HistGradientBoostingRegressor(**FINAL_MODEL_PARAMS))
    t0 = time.time()
    pipe.fit(train[FEATURES], to_target(train))
    fit_seconds = time.time() - t0
    print(f"Fitted in {fit_seconds:.1f}s")

    train_pred = to_price(pipe.predict(train[FEATURES]), train.area.values)
    train_metrics = metrics(train.price.values, train_pred)
    print(f"\nOn the training data it just saw (NOT a real score):")
    print(f"  MedAPE {train_metrics['MedAPE_pct']}%   "
          f"MAE {train_metrics['MAE_lakh']} L")
    print("  This will look better than the real result. It has to -- the")
    print("  model has seen every one of these rows.")

    # ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("OPENING test.csv — FIRST AND ONLY TIME")
    print("=" * 74)
    test = pd.read_csv(test_file)
    print(f"Test rows: {len(test):,}  buildings: {test.building.nunique():,}")

    overlap = set(train.building) & set(test.building)
    assert not overlap, f"LEAKAGE: {len(overlap)} buildings in both sets"
    print("Building overlap with training: 0 (verified)")

    t0 = time.time()
    test_pred = to_price(pipe.predict(test[FEATURES]), test.area.values)
    predict_seconds = time.time() - t0
    test_metrics = metrics(test.price.values, test_pred)

    print(f"\n{'':<20}{'cross-validation':>20}{'TEST SET':>14}")
    print("-" * 56)
    cv_reference = {"MedAPE_pct": 13.48, "MAE_lakh": 42.23, "bias_lakh": -5.34}
    for key, label in [("MedAPE_pct", "median APE %"),
                       ("MAE_lakh", "MAE (lakh)"),
                       ("bias_lakh", "bias (lakh)")]:
        print(f"{label:<20}{cv_reference[key]:>20}{test_metrics[key]:>14}")

    print(f"\nFull test results:")
    for k, v in test_metrics.items():
        print(f"  {k:<16} {v}")

    print("\nBy segment (this is what the app's warnings are built from):")
    segs = segment_report(test, test_pred)
    print(f"  {'segment':<18}{'n':>7}{'MedAPE %':>11}{'bias L':>10}")
    for name, s in segs.items():
        print(f"  {name:<18}{s['n']:>7,}{s['MedAPE_pct']:>11}{s['bias_lakh']:>10}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("PHASE 19 — SAVE THE PIPELINE")
    print("=" * 74)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # We save the WHOLE pipeline, not the bare model. The saved object
    # contains the imputer's medians and the encoder's category list, so the
    # Streamlit app feeds it a raw row and never reimplements a single
    # preprocessing step. Reimplementing them is how apps quietly drift out
    # of sync with their model.
    joblib.dump(pipe, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1024**2
    print(f"Saved: {MODEL_PATH}  ({size_mb:.1f} MB)")

    card = {
        "model": "HistGradientBoostingRegressor",
        "params": {k: (float(v) if isinstance(v, float) else v)
                   for k, v in FINAL_MODEL_PARAMS.items()},
        "target": "price per sqft; multiply by area for rupees",
        "features": FEATURES,
        "trained_on": {"rows": int(len(train)),
                       "buildings": int(train.building.nunique())},
        "test_metrics": test_metrics,
        "test_by_segment": segs,
        "cv_metrics_reference": cv_reference,
        # Versions matter: a pipeline pickled by one sklearn version may warn
        # or fail to load under another. Recording them turns a confusing
        # error into a one-line diagnosis.
        "versions": {"python": platform.python_version(),
                     "sklearn": sklearn.__version__,
                     "numpy": np.__version__,
                     "pandas": pd.__version__},
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_mb": round(size_mb, 2),
        "limitations": [
            "Trained on asking prices, not confirmed transaction prices.",
            "The dataset does not say whether area is carpet, built-up or "
            "super built-up.",
            "No listing dates, so the model cannot account for time.",
            "Only 26 training examples above Rs 50 crore; luxury estimates "
            "are unreliable and biased low.",
            "Independent Floor and Villa have under 300 examples each.",
        ],
    }
    META_PATH.write_text(json.dumps(card, indent=2))
    print(f"Saved: {META_PATH}")

    save_locality_reference(train)

    # ---------------------------------------------------------------
    print("\nVerifying the saved file actually works")
    print("-" * 74)
    reloaded = joblib.load(MODEL_PATH)
    check = to_price(reloaded.predict(test[FEATURES].head(200)),
                     test.area.values[:200])
    assert np.allclose(check, test_pred[:200]), \
        "reloaded model gives different answers"
    print("  Reloaded model reproduces the same predictions exactly.")

    one_row = test[FEATURES].head(1)
    t0 = time.time()
    for _ in range(20):
        reloaded.predict(one_row)
    print(f"  Single prediction: {(time.time()-t0)/20*1000:.1f} ms")
    print(f"  {len(test):,} predictions: {predict_seconds*1000:.0f} ms")

    print("\n" + "=" * 74)
    print(f"HEADLINE: median error {test_metrics['MedAPE_pct']}% on "
          f"{len(test):,} unseen properties")
    print(f"          in {test.building.nunique():,} buildings the model "
          f"had never seen")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
