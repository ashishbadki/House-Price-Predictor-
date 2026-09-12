"""
Tests for src/data_cleaning.py.

WHAT A TEST IS FOR
------------------
Not to prove the code works today -- you already saw it work when you ran it.
A test exists to tell you the day it STOPS working.

Every one of these locks in a decision we made and reasoned about in an
earlier phase. If someone (including you, in three months) changes a
threshold or reorders a step, the relevant test fails immediately and names
the decision that was broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_cleaning import (BLANK_TOKENS, COLUMNS_TO_DROP, INT32_MAX,
                           MAX_AREA_SQFT, MAX_PRICE_PER_SQFT, MIN_AREA_SQFT,
                           MIN_PRICE_PER_SQFT, add_missing_indicators,
                           clean_dataset, drop_exact_duplicates,
                           drop_unusable_columns, fix_coordinates,
                           normalise_locality, remove_impossible_rows,
                           standardise_text)


def make_row(**overrides) -> dict:
    """One valid row. Tests override just the field they are about, which
    keeps each test about one thing."""
    row = dict(title="Test Tower", price=10_000_000, area=800,
               price_per_sqft=12_500.0, locality="Andheri", city="Mumbai",
               property_type="Apartment", bedroom_num=2, bathroom_num=2,
               balcony_num=1, furnished="Unfurnished", age=5,
               total_floors=1, latitude=19.12, longitude=72.85)
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Each cleaning step, on its own
# ---------------------------------------------------------------------------

def test_blank_strings_become_nan():
    """Phase 3: 16.1% of titles were empty strings, and `isna()` missed all
    of them. This is the fix, and this test is what stops it regressing."""
    df = pd.DataFrame([make_row(title=""), make_row(title="   "),
                       make_row(title="Real Name")])
    out = standardise_text(df)
    assert out.title.isna().sum() == 2
    assert out.title.iloc[2] == "Real Name"


def test_internal_whitespace_is_collapsed():
    df = pd.DataFrame([make_row(locality="Sector  19   Kharghar")])
    assert standardise_text(df).locality.iloc[0] == "Sector 19 Kharghar"


def test_na_placeholders_are_treated_as_missing():
    df = pd.DataFrame([make_row(title=t) for t in
                       ["NA", "n/a", "NULL", "-", "unknown", "Fine"]])
    out = standardise_text(df)
    assert out.title.isna().sum() == 5


def test_duplicates_are_removed():
    df = pd.DataFrame([make_row(), make_row(), make_row(price=9_000_000)])
    assert len(drop_exact_duplicates(df)) == 2


def test_duplicate_removal_resets_the_index():
    """Without reset_index, later positional indexing silently misaligns."""
    df = pd.DataFrame([make_row(), make_row(), make_row(area=900)])
    out = drop_exact_duplicates(df)
    assert list(out.index) == list(range(len(out)))


def test_compound_locality_keeps_the_broader_area():
    df = pd.DataFrame([make_row(locality="Siddharth Nagar, Goregaon")])
    assert normalise_locality(df).locality.iloc[0] == "Goregaon"


@pytest.mark.parametrize("lat,lon,should_survive", [
    (19.12, 72.85, True),    # central Mumbai
    (19.70, 72.77, True),    # Palghar -- real outer MMR, must NOT be dropped
    (28.50, 77.16, False),   # Delhi
    (22.70, 88.39, False),   # Kolkata
    (18.50, 73.80, False),   # Pune
])
def test_only_out_of_region_coordinates_are_nulled(lat, lon, should_survive):
    """Phase 4: a tight box wrongly flagged real Palghar listings. The box is
    deliberately wide. This test is what stops someone tightening it."""
    out = fix_coordinates(pd.DataFrame([make_row(latitude=lat, longitude=lon)]))
    assert pd.notna(out.latitude.iloc[0]) == should_survive


def test_bad_coordinates_null_the_field_but_keep_the_row():
    """The Phase 4 judgement call: one bad field must not destroy a good
    record."""
    out = fix_coordinates(pd.DataFrame([make_row(latitude=28.5, longitude=77.1)]))
    assert len(out) == 1
    assert pd.isna(out.latitude.iloc[0])
    assert out.price.iloc[0] == 10_000_000     # everything else survives


def test_int32_sentinel_price_is_removed():
    """2^31 - 1 appeared twice as a 'price'. It is an overflow, not a price."""
    df = pd.DataFrame([make_row(), make_row(price=INT32_MAX)])
    assert len(remove_impossible_rows(df)) == 1


@pytest.mark.parametrize("price,area,should_survive", [
    (10_000_000, 800, True),         # 12,500/sqft — normal
    (20_000, 800, False),            # 25/sqft — impossible
    (10_000_000, 50, False),         # area below the floor
    (10_000_000, 19_000, False),     # area above the ceiling
    (900_000_000, 800, False),       # 1.1 lakh/sqft — above the ceiling
])
def test_impossible_combinations_are_removed(price, area, should_survive):
    df = pd.DataFrame([make_row(price=price, area=area)])
    assert len(remove_impossible_rows(df)) == (1 if should_survive else 0)


def test_studio_with_zero_bedrooms_survives():
    """Phase 4's most important negative test. A naive 'drop rows where
    bedrooms == 0' rule would delete valid Studio Apartments."""
    df = pd.DataFrame([make_row(bedroom_num=0, property_type="Studio Apartment",
                                area=329, price=8_000_000)])
    assert len(remove_impossible_rows(df)) == 1


def test_missing_indicators_are_added():
    df = pd.DataFrame([make_row(age=0, balcony_num=0),
                       make_row(age=5, balcony_num=2)])
    out = add_missing_indicators(df)
    assert list(out.age_is_zero) == [1, 0]
    assert list(out.has_balcony) == [0, 1]


def test_original_values_survive_the_indicators():
    """We ADD a flag, we do not replace the number. The model decides what
    the zero means, not us."""
    out = add_missing_indicators(pd.DataFrame([make_row(age=0)]))
    assert out.age.iloc[0] == 0
    assert "age" in out.columns


def test_leaking_and_useless_columns_are_dropped():
    out = drop_unusable_columns(pd.DataFrame([make_row()]))
    for col in COLUMNS_TO_DROP:
        assert col not in out.columns


# ---------------------------------------------------------------------------
# Properties that must hold for every function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    standardise_text, drop_exact_duplicates, normalise_locality,
    fix_coordinates, remove_impossible_rows, add_missing_indicators,
    drop_unusable_columns,
])
def test_no_function_mutates_its_input(fn):
    """Every cleaning function starts with df.copy(). If one ever forgets,
    running a notebook cell twice gives a different answer the second time --
    and that costs an afternoon to find. This test finds it in a second."""
    df = pd.DataFrame([make_row(title="", age=0, latitude=28.5)])
    before = df.copy(deep=True)
    fn(df)
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# The whole pipeline, on real data
# ---------------------------------------------------------------------------

def test_full_clean_on_real_data(raw_sample):
    out = clean_dataset(raw_sample, verbose=False)

    assert "price_per_sqft" not in out.columns, "LEAKAGE survived cleaning"
    assert not out.duplicated().any(), "duplicates survived"
    assert (out.price > 0).all()
    assert out.area.between(MIN_AREA_SQFT, MAX_AREA_SQFT).all()
    rate = out.price / out.area
    assert rate.between(MIN_PRICE_PER_SQFT, MAX_PRICE_PER_SQFT).all()
    assert {"age_is_zero", "has_balcony"} <= set(out.columns)
    assert len(out) < len(raw_sample), "cleaning removed nothing at all"


def test_cleaning_is_deterministic(raw_sample):
    """Same input, same output. Reproducibility is Rule 7."""
    a = clean_dataset(raw_sample, verbose=False)
    b = clean_dataset(raw_sample, verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_twice_changes_nothing_the_second_time(raw_sample):
    """Idempotence: cleaning already-clean data should be a no-op. If it is
    not, some step is doing something it should not."""
    once = clean_dataset(raw_sample, verbose=False)
    # The second pass cannot re-drop columns that are already gone, so we
    # only check the row-level steps.
    twice = drop_exact_duplicates(remove_impossible_rows(once))
    assert len(twice) == len(once)
