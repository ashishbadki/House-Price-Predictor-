"""
encoding_experiment.py — Phase 8: how to turn text columns into numbers.

Models cannot read text. "Andheri" has to become a number before any model can
use it. There are several ways to do that, and the choice matters far more for
some models than others.

This script measures five encodings on two different model types, then
demonstrates the one mistake that quietly ruins projects: target encoding
computed on the wrong rows.

Usage:
    python src/encoding_experiment.py data/processed/mumbai_clean.csv

Takes about 2 minutes. The one-hot + tree combination is deliberately slow --
that slowness is one of the findings.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LAKH = 100_000
N_FOLDS = 5
MIN_CATEGORY_ROWS = 30    # below this, a locality is folded into "infrequent"

NUM_COLS = ["area", "bedroom_num", "bathroom_num", "balcony_num", "age",
            "age_is_zero", "has_balcony", "latitude", "longitude"]
SMALL_CATS = ["property_type", "furnished"]   # only 5 and 3 values, easy


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with how often it appears.

    "Thane" appears in 10% of rows -> 0.10. "Neral" in 0.7% -> 0.007.

    Written as a transformer rather than a precomputed column ON PURPOSE.
    A transformer learns its counts in `fit`, which means inside a pipeline it
    only ever sees training rows. Precomputing the column would count test
    rows too, which is a (mild) leak.
    """

    def fit(self, X, y=None):
        self.maps_ = [X.iloc[:, i].value_counts(normalize=True)
                      for i in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.column_stack([
            X.iloc[:, i].map(self.maps_[i]).fillna(0).values
            for i in range(X.shape[1])
        ])


def numeric_pipeline():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def cv_mae(df, locality_encoder, model, scale_numbers, label):
    """Cross-validate one (encoding, model) combination.

    Everything -- imputer, scaler, encoder, model -- goes inside one Pipeline.
    That is what makes each fold fit only on its own training rows. Doing the
    encoding before the loop is the mistake demonstrated at the bottom of
    this file.
    """
    cols = NUM_COLS + SMALL_CATS + ["locality"]
    X = df[cols]
    y = (df.price / df.area).values      # Phase 6 target
    price = df.price.values.astype(float)
    area = df.area.values.astype(float)

    parts = [
        ("num", numeric_pipeline() if scale_numbers else "passthrough", NUM_COLS),
        ("small_cats", OneHotEncoder(handle_unknown="ignore"), SMALL_CATS),
    ]
    if locality_encoder is not None:
        parts.append(("locality", locality_encoder, ["locality"]))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros(len(df))
    t0 = time.time()
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([("pre", ColumnTransformer(parts)), ("model", model)])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred[test_idx] = pipe.predict(X.iloc[test_idx]) * area[test_idx]

    mae = mean_absolute_error(price, pred) / LAKH
    print(f"  {label:<44} MAE {mae:6.2f} L   [{time.time()-t0:4.0f}s]")
    return mae


def compare_encodings(df: pd.DataFrame) -> None:
    ridge = lambda: Ridge(alpha=1.0)
    tree = lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE,
                                                 max_iter=200)

    print("\n" + "=" * 70)
    print("RIDGE — a linear model. Encoding matters enormously here.")
    print("=" * 70)
    cv_mae(df, None, ridge(), True, "locality dropped entirely")
    cv_mae(df, OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), ridge(), True,
           "Label / Ordinal encoding")
    cv_mae(df, OneHotEncoder(handle_unknown="ignore"), ridge(), True,
           "One-Hot, all 403 columns")
    cv_mae(df, OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=MIN_CATEGORY_ROWS), ridge(), True,
           f"One-Hot, rare grouped (min {MIN_CATEGORY_ROWS} rows)")
    cv_mae(df, FrequencyEncoder(), ridge(), True, "Frequency encoding")
    cv_mae(df, TargetEncoder(random_state=RANDOM_STATE), ridge(), True,
           "Target encoding (done correctly)")

    print("\n" + "=" * 70)
    print("HISTGRADIENTBOOSTING — a tree model. Encoding barely matters.")
    print("=" * 70)
    cv_mae(df, None, tree(), False, "locality dropped entirely")
    cv_mae(df, OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), tree(), False,
           "Label / Ordinal encoding")
    cv_mae(df, OneHotEncoder(handle_unknown="ignore", sparse_output=False),
           tree(), False, "One-Hot, all 403 columns  (SLOW)")
    cv_mae(df, FrequencyEncoder(), tree(), False, "Frequency encoding")
    cv_mae(df, TargetEncoder(random_state=RANDOM_STATE), tree(), False,
           "Target encoding (done correctly)")


def leakage_demo(df: pd.DataFrame) -> None:
    """Three ways to compute a target encoding. Only one is honest.

    We use `title` (the building name, 9,254 values) because the damage from
    this mistake grows with the number of categories. On `locality` (403
    values) the same mistake is almost invisible.

    A real held-out test set is kept aside and never used for fitting, so we
    can see how far each CV score is from the truth.
    """
    print("\n" + "=" * 70)
    print("THE TARGET-ENCODING TRAP")
    print("=" * 70)

    d = df.copy()
    d["title"] = d.title.fillna("UNKNOWN")
    d["y"] = d.price / d.area
    dev, test = train_test_split(d, test_size=0.2, random_state=RANDOM_STATE)
    print(f"dev {len(dev):,} rows | test {len(test):,} rows (never fitted on)\n")

    mk = lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE,
                                               max_iter=200)

    def run(dev_df, test_df, extra, label, pre=None):
        feats = NUM_COLS + extra
        cols = feats + SMALL_CATS + (["title"] if pre is not None else [])
        if pre is None:
            pre = ColumnTransformer([
                ("n", "passthrough", feats),
                ("o", OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1), SMALL_CATS)])

        kf = KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        p = np.zeros(len(dev_df))
        Xd, yd = dev_df[cols], dev_df.y.values
        for tr, te in kf.split(Xd):
            m = Pipeline([("pre", pre), ("m", mk())]).fit(Xd.iloc[tr], yd[tr])
            p[te] = m.predict(Xd.iloc[te]) * dev_df.area.values[te]
        cv = mean_absolute_error(dev_df.price.values, p) / LAKH

        m = Pipeline([("pre", pre), ("m", mk())]).fit(Xd, yd)
        ts = mean_absolute_error(
            test_df.price.values,
            m.predict(test_df[cols]) * test_df.area.values) / LAKH

        print(f"  {label:<46} CV {cv:6.2f}L  TEST {ts:6.2f}L  "
              f"CV lies by {cv-ts:+6.2f}L")

    # LEVEL 0 — no building feature at all
    run(dev, test, [], "no building feature (reference)")

    # LEVEL 1 — average computed over EVERY row, test rows included.
    # The test set helped build the feature, so even the test score is fake.
    all_rows_mean = d.groupby("title").y.mean()
    d1 = d.assign(title_te=d.title.map(all_rows_mean))
    run(d1.loc[dev.index], d1.loc[test.index], ["title_te"],
        "WRONG 1: average over ALL rows (dev + test)")

    # LEVEL 2 — dev rows only, but computed once BEFORE the CV loop.
    # Test score is now honest. The CV score is not.
    enc = TargetEncoder(random_state=RANDOM_STATE).fit(dev[["title"]], dev.y.values)
    d2_dev = dev.assign(title_te=enc.transform(dev[["title"]]).ravel())
    d2_test = test.assign(title_te=enc.transform(test[["title"]]).ravel())
    run(d2_dev, d2_test, ["title_te"],
        "WRONG 2: dev only, but fitted BEFORE the CV loop")

    # LEVEL 3 — encoder inside the Pipeline, refitted on every fold.
    proper = ColumnTransformer([
        ("n", "passthrough", NUM_COLS),
        ("o", OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1), SMALL_CATS),
        ("t", TargetEncoder(random_state=RANDOM_STATE), ["title"]),
    ])
    run(dev, test, [], "RIGHT: TargetEncoder inside the Pipeline", pre=proper)

    print("\n  Read the 'CV lies by' column. A large number there means your")
    print("  cross-validation score is fiction and you would not know it.")


def main(path: str) -> int:
    csv = Path(path)
    if not csv.exists():
        print(f"ERROR: file not found: {csv}")
        return 1
    df = pd.read_csv(csv)
    print(f"{len(df):,} rows | {df.locality.nunique()} localities | "
          f"{df.title.nunique():,} building names")
    compare_encodings(df)
    leakage_demo(df)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
