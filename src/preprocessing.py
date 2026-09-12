"""
preprocessing.py — Phase 14: ONE definition of how raw input becomes model input.

WHY THIS FILE EXISTS
--------------------
Look at what we have built so far. `train_models.py`, `evaluate.py`,
`encoding_experiment.py` and `target_experiment.py` each contain their own copy
of the column lists and the ColumnTransformer.

Four copies of the same thing is four chances to drift apart. Change
`min_frequency` in one and forget the others, and two scripts silently
disagree about what the model sees -- with no error to tell you.

From here on there is one definition, in this file, and everything imports it.
That includes the Streamlit app in Phase 21, which must preprocess a user's
form input in exactly the way the training data was preprocessed. If those two
ever differ, the app produces confident nonsense.

WHAT A PIPELINE IS, AND WHY IT IS NOT OPTIONAL
----------------------------------------------
A Pipeline chains steps so that calling `.fit()` fits every step, and calling
`.predict()` applies every step in the same order.

    raw row -> impute missing -> scale numbers -> encode text -> model -> price

The reason this matters is not tidiness. It is that `SimpleImputer` and
`OneHotEncoder` LEARN things from data -- a median, a list of categories. If
you fit them once on everything and then split, every validation row has helped
build its own features.

Phase 8 measured that mistake at 6.85 lakh of self-deception. Inside a
Pipeline, `cross_val_score` refits every step on each fold's training rows
only, and the mistake becomes impossible to make by accident.

Usage:
    from src.preprocessing import build_pipeline, FEATURES
    pipe = build_pipeline(RandomForestRegressor(), scale_numbers=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# THE FEATURE CONTRACT — the single source of truth
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "area", "bedroom_num", "bathroom_num", "balcony_num", "age",
    "age_is_zero", "has_balcony", "latitude", "longitude",
    "dist_nariman_km", "dist_bkc_km", "dist_cbd_km",
    "rooms_total", "bath_per_bed",
]

CATEGORICAL_FEATURES = ["locality_grouped", "property_type", "furnished"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Phase 8: a locality with fewer than this many rows joins the "infrequent"
# bucket instead of getting its own column.
MIN_CATEGORY_ROWS = 30

RANDOM_STATE = 42


def build_preprocessor(scale_numbers: bool = False) -> ColumnTransformer:
    """Turn a DataFrame of raw features into a numeric matrix.

    NUMERIC BRANCH
        SimpleImputer(strategy="median") fills the 151 rows with no
        coordinates. Median rather than mean because a median is not dragged
        around by outliers -- and this dataset has a Rs 135 crore listing.

        StandardScaler is applied ONLY for linear models. They compare
        coefficients across features, so without scaling `area` (hundreds) and
        `has_balcony` (0 or 1) are treated as if a one-unit change means the
        same thing. Trees only ask "above or below X?", so scaling changes
        nothing for them -- and skipping it keeps the numbers readable when
        debugging.

    CATEGORICAL BRANCH
        OneHotEncoder, the Phase 8 decision.

        handle_unknown="infrequent_if_exist" is the setting that keeps the
        Streamlit app alive. If a user picks a locality the model never saw
        during training, it is routed to the "infrequent" bucket instead of
        raising an exception at prediction time.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numbers:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=MIN_CATEGORY_ROWS,
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        # Anything not listed above is dropped. This is a safety feature, not
        # a default we happen to accept: `price`, `title` and `building` are
        # all still in the DataFrame, and this line is what guarantees they
        # cannot reach the model.
        remainder="drop",
    )


def build_pipeline(model, scale_numbers: bool = False) -> Pipeline:
    """Preprocessor + model as one object.

    The whole point: this object can be handed to cross_val_score,
    RandomizedSearchCV, or joblib.dump, and it carries its preprocessing with
    it. In Phase 19 we save THIS, not the bare model -- so the app never has
    to reimplement any of the steps above.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numbers)),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# TARGET HELPERS — Phase 6 decided we train on price per sqft
# ---------------------------------------------------------------------------

def to_target(df: pd.DataFrame) -> np.ndarray:
    """price -> price per sqft. This is what the model learns."""
    return (df["price"] / df["area"]).values


def to_price(rate_prediction, area) -> np.ndarray:
    """price per sqft -> rupees. This is what the user sees.

    Every metric in this project is computed AFTER this conversion. Scoring on
    the rate would answer a different question from the one the app asks.
    """
    return np.asarray(rate_prediction) * np.asarray(area)


def get_groups(df: pd.DataFrame) -> np.ndarray:
    """Phase 9: the building each row belongs to.

    Pass this as `groups=` to every splitter. Forget it and GroupKFold silently
    behaves like a plain KFold, and every score becomes 12% too good.
    """
    return df["building"].values


def check_input(df: pd.DataFrame) -> None:
    """Fail loudly if the DataFrame is not what the pipeline expects."""
    missing = [c for c in FEATURES if c not in df.columns]
    assert not missing, f"missing required features: {missing}"

    banned = {"price_per_sqft", "title"}
    present = banned & set(FEATURES)
    assert not present, f"BANNED column in FEATURES: {present}"


def describe() -> None:
    """Print the contract. Useful when debugging a mismatch with the app."""
    print(f"{len(FEATURES)} features")
    print(f"  numeric     ({len(NUMERIC_FEATURES)}): {', '.join(NUMERIC_FEATURES)}")
    print(f"  categorical ({len(CATEGORICAL_FEATURES)}): "
          f"{', '.join(CATEGORICAL_FEATURES)}")
    print(f"  rare-category cutoff: {MIN_CATEGORY_ROWS} rows")
    print(f"  target: price / area   (converted back with to_price)")


if __name__ == "__main__":
    describe()
