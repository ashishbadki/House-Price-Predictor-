"""
split_data.py — Phase 9: split the data into training and test sets, correctly.

THE PROBLEM THIS SOLVES
-----------------------
73.9% of our rows belong to a building that has more than one listing. The
biggest single building has 139 listings.

A plain random split scatters those listings across both sides. The model
studies eight flats in "Runwal Bliss", then gets graded on a ninth flat in
"Runwal Bliss". It does not need to understand what makes property expensive;
it can recall what that building costs.

Measured cost of getting this wrong:

    Random 5-fold CV                MAE 36.65 lakh
    Group 5-fold CV (by building)   MAE 41.09 lakh

Every number in Phases 6, 7 and 8 was therefore about 4.4 lakh (12%)
optimistic. Not because of a bug -- because of the split.

WHAT THIS MODULE PROVIDES
-------------------------
    make_building_groups(df)  - the group label each row belongs to
    train_test_split_by_group - one 80/20 holdout, buildings kept together
    get_cv_splitter()         - GroupKFold, for use inside every later phase

Usage:
    python src/split_data.py data/processed/mumbai_features.csv data/processed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5
LAKH = 100_000


def make_building_groups(df: pd.DataFrame) -> pd.Series:
    """One group label per row. Rows in the same building share a label.

    `title` holds the building name, and it is missing on 18.5% of rows.

    A missing title does NOT mean "these are all the same building". It means
    we do not know. So each row with a missing title becomes its own group of
    one. That is the honest choice: it never claims two rows are related when
    we have no evidence they are.

    The lazy alternative -- fillna("UNKNOWN") -- would put 9,561 unrelated
    listings into one giant group, and GroupKFold would then shove all of them
    into a single fold. That both wastes data and creates a weirdly lopsided
    split.
    """
    known = df.title.notna()
    return pd.Series(
        np.where(known, df.title, "UNKNOWN_" + df.index.astype(str)),
        index=df.index,
        name="building",
    )


def train_test_split_by_group(df: pd.DataFrame, groups: pd.Series):
    """Hold out 20% of BUILDINGS -- not 20% of rows -- as the test set.

    GroupShuffleSplit picks whole groups, so no building can appear on both
    sides. The test set will not be exactly 20% of rows, because buildings
    have different sizes. That is expected and fine.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def get_cv_splitter():
    """The cross-validation splitter every later phase must use.

    GroupKFold keeps each building inside a single fold. Pass `groups=` when
    you call `.split()` or `cross_val_score`, or it silently behaves like a
    plain KFold and you are back to lying to yourself.

    Note GroupKFold has no `shuffle` or `random_state`: it assigns groups to
    folds deterministically, balancing fold sizes. That is already
    reproducible.
    """
    return GroupKFold(n_splits=N_CV_FOLDS)


def why_no_separate_validation_set() -> None:
    """We use two pieces of data, not three. Here is the reasoning.

    The textbook version is train / validation / test:
        train      - fit the model
        validation - compare models and tune settings
        test       - one final honest score

    We use train / test, and get the validation part from cross-validation
    inside the training set. Reasons:

      1. Cross-validation uses all the training data for both roles, in turn.
         A fixed validation set spends 20% of the data on a single estimate.
      2. A CV score is an average over 5 folds, so it is far more stable than
         one validation split -- which matters when we are comparing models
         that differ by a lakh or two.
      3. It keeps the rule simple: the test set is touched exactly once, at
         the very end of Phase 18.

    A separate validation set earns its place when CV is too slow, or when
    time-ordering forces a fixed cut. Neither applies here.
    """


def report(train, test, groups_train, groups_test) -> None:
    print(f"\n{'':<14}{'rows':>10}{'%':>8}{'buildings':>12}"
          f"{'median price':>15}")
    print("-" * 60)
    total = len(train) + len(test)
    for name, part, grp in [("TRAIN", train, groups_train),
                            ("TEST", test, groups_test)]:
        print(f"{name:<14}{len(part):>10,}{len(part)/total*100:>7.1f}%"
              f"{grp.nunique():>12,}{part.price.median()/LAKH:>13,.0f} L")

    overlap = set(groups_train) & set(groups_test)
    print(f"\nBuildings appearing on BOTH sides: {len(overlap)}")
    assert not overlap, "LEAKAGE: a building is in both train and test"

    # A group split does not guarantee similar price distributions, so check.
    # If these were far apart, the test set would be measuring a different
    # problem from the one we trained on.
    print(f"\nSanity check -- the two sides should look alike:")
    for q in [0.25, 0.50, 0.75, 0.95]:
        a = train.price.quantile(q) / LAKH
        b = test.price.quantile(q) / LAKH
        print(f"  {int(q*100):>3}th percentile   train {a:>7,.0f} L   "
              f"test {b:>7,.0f} L   diff {abs(a-b)/a*100:>4.1f}%")


def main(in_path: str, out_dir: str) -> int:
    src = Path(in_path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1

    # We need `title` to form groups, and mumbai_features.csv drops it
    # (correctly -- it is a banned feature). So read groups from the cleaned
    # file, which sits beside it.
    features = pd.read_csv(src)
    clean_path = src.parent / "mumbai_clean.csv"
    if not clean_path.exists():
        print(f"ERROR: need {clean_path} to read building names")
        return 1
    clean = pd.read_csv(clean_path)

    assert len(features) == len(clean), (
        f"row mismatch: features has {len(features):,}, clean has {len(clean):,}. "
        "Re-run data_cleaning.py then feature_engineering.py.")

    groups = make_building_groups(clean)
    print(f"{len(features):,} rows")
    print(f"{groups.nunique():,} distinct buildings")
    known = clean.title.notna()
    vc = clean.loc[known, "title"].value_counts()
    share = clean.loc[known, "title"].map(vc).gt(1).sum() / len(clean)
    print(f"{share*100:.1f}% of rows sit in a building with more than one listing")
    print(f"largest building: {vc.max()} listings")

    features = features.assign(building=groups.values)
    train, test = train_test_split_by_group(features, features.building)
    report(train, test, train.building, test.building)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    train.to_csv(dest / "train.csv", index=False)
    test.to_csv(dest / "test.csv", index=False)
    print(f"\nWritten: {dest/'train.csv'}  and  {dest/'test.csv'}")
    print("\nThe test set is now off limits until Phase 18.")
    print("Use get_cv_splitter() with groups=train.building for everything "
          "before that.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
