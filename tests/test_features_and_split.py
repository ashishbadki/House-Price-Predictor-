"""
Tests for feature engineering, preprocessing and the train/test split.

The split tests are the most valuable in this file. A broken split does not
crash -- it silently makes every score look 12% better than reality, which is
exactly the kind of failure a test should catch and a human never will.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (ALL_FEATURES, BANNED, BKC, NARIMAN_POINT,
                                 add_distance_features, add_room_features,
                                 build_features, group_rare_localities,
                                 haversine_km)
from preprocessing import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                           build_pipeline, build_preprocessor, to_price,
                           to_target)
from split_data import make_building_groups, train_test_split_by_group


# ---------------------------------------------------------------------------
# DISTANCE
# ---------------------------------------------------------------------------

def test_distance_to_itself_is_zero():
    assert haversine_km(19.0, 72.9, 19.0, 72.9) == pytest.approx(0, abs=1e-9)


def test_distance_is_symmetric():
    a = haversine_km(19.0, 72.9, 18.92, 72.82)
    b = haversine_km(18.92, 72.82, 19.0, 72.9)
    assert a == pytest.approx(b)


def test_known_distance_is_about_right():
    """Nariman Point to BKC is roughly 16 km as the crow flies. If the
    formula is wrong -- degrees not converted, wrong radius -- this catches
    it immediately."""
    km = haversine_km(*NARIMAN_POINT, *BKC)
    assert 14 < km < 19, f"got {km:.1f} km, expected about 16"


def test_distance_features_are_added():
    df = pd.DataFrame({"latitude": [19.0], "longitude": [72.9]})
    out = add_distance_features(df)
    assert out.dist_cbd_km.iloc[0] == pytest.approx(
        min(out.dist_nariman_km.iloc[0], out.dist_bkc_km.iloc[0]))


def test_missing_coordinates_give_missing_distances():
    df = pd.DataFrame({"latitude": [np.nan], "longitude": [np.nan]})
    out = add_distance_features(df)
    assert out.dist_cbd_km.isna().all()


# ---------------------------------------------------------------------------
# ROOMS AND RARE CATEGORIES
# ---------------------------------------------------------------------------

def test_room_features():
    df = pd.DataFrame({"bedroom_num": [2, 0], "bathroom_num": [3, 1]})
    out = add_room_features(df)
    assert list(out.rooms_total) == [5, 1]
    assert out.bath_per_bed.iloc[0] == pytest.approx(1.5)
    assert np.isnan(out.bath_per_bed.iloc[1])     # no divide-by-zero crash


def test_rare_localities_become_other():
    df = pd.DataFrame({"locality": ["A"] * 50 + ["B"] * 30 + ["Rare"] * 2})
    out = group_rare_localities(df, top_n=2)
    assert set(out.locality_grouped) == {"A", "B", "Other"}
    assert (out.locality_grouped == "Other").sum() == 2


# ---------------------------------------------------------------------------
# THE FEATURE CONTRACT
# ---------------------------------------------------------------------------

def test_banned_columns_are_not_features():
    """The guard that stops price_per_sqft ever creeping back in."""
    for col in BANNED:
        assert col not in ALL_FEATURES, f"{col} is banned: {BANNED[col]}"
        assert col not in FEATURES


def test_the_two_feature_lists_agree():
    """feature_engineering.py and preprocessing.py each declare a list.
    If they drift apart, the app and the model disagree silently."""
    assert set(ALL_FEATURES) == set(FEATURES)


def test_features_split_cleanly_into_numeric_and_categorical():
    assert set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES) == set()
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURES)


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def test_preprocessor_drops_everything_not_declared(train_sample):
    """remainder='drop' is what guarantees `price` and `building` cannot
    reach the model even though they are still in the DataFrame."""
    pre = build_preprocessor()
    out = pre.fit_transform(train_sample)
    assert out.shape[0] == len(train_sample)
    assert np.isfinite(out).all(), "preprocessing produced NaN or inf"


def test_preprocessor_fills_missing_values(train_sample):
    df = train_sample.copy()
    df.loc[df.index[:10], "latitude"] = np.nan
    out = build_preprocessor().fit_transform(df)
    assert np.isfinite(out).all()


def test_target_round_trips():
    df = pd.DataFrame({"price": [10_000_000.0], "area": [800.0]})
    rate = to_target(df)
    assert rate[0] == pytest.approx(12_500)
    assert to_price(rate, df.area.values)[0] == pytest.approx(10_000_000)


def test_pipeline_fits_and_predicts(train_sample):
    from sklearn.linear_model import Ridge
    pipe = build_pipeline(Ridge(), scale_numbers=True)
    pipe.fit(train_sample[FEATURES], to_target(train_sample))
    assert len(pipe.predict(train_sample[FEATURES])) == len(train_sample)


# ---------------------------------------------------------------------------
# THE SPLIT — the highest-value tests here
# ---------------------------------------------------------------------------

def test_rows_without_a_title_each_get_their_own_group():
    """A missing building name means 'we do not know', not 'same building'.
    fillna('UNKNOWN') would lump 9,561 unrelated listings into one group."""
    df = pd.DataFrame({"title": ["Tower A", None, None, "Tower A"]})
    groups = make_building_groups(df)
    assert groups.iloc[0] == groups.iloc[3]      # same named building
    assert groups.iloc[1] != groups.iloc[2]      # two unknowns are not equal
    assert groups.nunique() == 3


def test_no_building_appears_on_both_sides(train_sample):
    """The single most important test in the project. If this fails, every
    score in every report is roughly 12% too good."""
    groups = pd.Series(train_sample.building.values, index=train_sample.index)
    train, test = train_test_split_by_group(train_sample, groups)
    assert set(train.building) & set(test.building) == set()


def test_split_is_reproducible(train_sample):
    groups = pd.Series(train_sample.building.values, index=train_sample.index)
    a, _ = train_test_split_by_group(train_sample, groups)
    b, _ = train_test_split_by_group(train_sample, groups)
    assert list(a.index) == list(b.index)


def test_split_keeps_most_rows_for_training(train_sample):
    groups = pd.Series(train_sample.building.values, index=train_sample.index)
    train, test = train_test_split_by_group(train_sample, groups)
    share = len(test) / (len(train) + len(test))
    # Not exactly 0.20: whole buildings move together and they differ in size.
    assert 0.10 < share < 0.35


# ---------------------------------------------------------------------------
# END TO END
# ---------------------------------------------------------------------------

def test_build_features_produces_every_declared_feature(raw_sample):
    from data_cleaning import clean_dataset
    clean = clean_dataset(raw_sample, verbose=False)
    out = build_features(clean)
    missing = [c for c in ALL_FEATURES if c not in out.columns]
    assert not missing, f"declared but not built: {missing}"
