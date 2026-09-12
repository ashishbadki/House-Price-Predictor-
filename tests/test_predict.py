"""
Tests for src/predict.py and the saved model.

These are the tests that matter most, because this is the code a real user
touches. A bug in `train_models.py` costs you an afternoon. A bug here shows
a stranger a wrong price.
"""

from __future__ import annotations

import numpy as np
import pytest

from predict import (InvalidInput, PropertyInput, build_row, format_inr,
                     list_localities, predict_price, validate)
from preprocessing import FEATURES

CRORE = 10_000_000
LAKH = 100_000


@pytest.fixture(scope="module")
def locality():
    """A locality the model knows well, whatever the data happens to hold."""
    return list_localities()[0]


# ---------------------------------------------------------------------------
# VALIDATION — every one of these is a bug a user would otherwise see
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,value,fragment", [
    ("area_sqft", -100, "positive"),
    ("area_sqft", 0, "positive"),
    ("area_sqft", 5, "between"),
    ("area_sqft", 500_000, "between"),
    ("area_sqft", float("nan"), "positive"),
    ("area_sqft", float("inf"), "positive"),
    ("area_sqft", "900", "number"),
    ("bedrooms", -1, "between"),
    ("bedrooms", 99, "between"),
    ("bedrooms", 2.5, "whole number"),
    ("bathrooms", 0, "at least one"),
    ("bathrooms", 99, "between"),
    ("balconies", -1, "between"),
    ("age_years", -5, "between"),
    ("age_years", 500, "between"),
    ("property_type", "Houseboat", "Property type"),
    ("furnishing", "Very Furnished", "Furnishing"),
])
def test_bad_values_are_rejected(locality, field, value, fragment):
    kwargs = dict(locality=locality, area_sqft=900, bedrooms=2, bathrooms=2)
    kwargs[field] = value
    with pytest.raises(InvalidInput) as exc:
        predict_price(**kwargs)
    assert fragment.lower() in str(exc.value).lower()


def test_unknown_locality_is_rejected():
    with pytest.raises(InvalidInput, match="not a locality"):
        predict_price(locality="Paris", area_sqft=900, bedrooms=2, bathrooms=2)


def test_empty_locality_is_rejected():
    with pytest.raises(InvalidInput):
        predict_price(locality="", area_sqft=900, bedrooms=2, bathrooms=2)


def test_booleans_are_not_accepted_as_numbers(locality):
    """In Python, bool is a subclass of int, so isinstance(True, int) is
    True. Without an explicit check, area=True would silently become 1."""
    with pytest.raises(InvalidInput):
        predict_price(locality=locality, area_sqft=True, bedrooms=2,
                      bathrooms=2)


def test_implausible_combinations_are_flagged(locality):
    """Not impossible, but almost certainly a typo, and the model has no
    examples like it. Asking beats answering confidently."""
    with pytest.raises(InvalidInput, match="typo"):
        predict_price(locality=locality, area_sqft=900, bedrooms=1,
                      bathrooms=9)
    with pytest.raises(InvalidInput, match="too small"):
        predict_price(locality=locality, area_sqft=150, bedrooms=3,
                      bathrooms=2)


def test_studio_with_zero_bedrooms_is_allowed(locality):
    """Zero bedrooms is valid for a studio. The validator must not treat 0 as
    'missing' -- the same trap Phase 4 avoided in cleaning."""
    r = predict_price(locality=locality, area_sqft=350, bedrooms=0,
                      bathrooms=1, property_type="Studio Apartment")
    assert r["estimate"] > 0


# ---------------------------------------------------------------------------
# BUILDING THE ROW
# ---------------------------------------------------------------------------

def test_row_has_exactly_the_columns_the_model_expects(locality):
    row = build_row(PropertyInput(locality, 900, 2, 2))
    assert list(row.columns) == FEATURES
    assert len(row) == 1


def test_derived_features_are_computed_correctly(locality):
    row = build_row(PropertyInput(locality, 900, 2, 3, balconies=1,
                                  age_years=0))
    assert row.rooms_total.iloc[0] == 5           # 2 + 3
    assert row.bath_per_bed.iloc[0] == 1.5        # 3 / 2
    assert row.age_is_zero.iloc[0] == 1
    assert row.has_balcony.iloc[0] == 1


def test_zero_bedrooms_does_not_divide_by_zero(locality):
    row = build_row(PropertyInput(locality, 350, 0, 1))
    assert np.isnan(row.bath_per_bed.iloc[0])     # NaN, not a crash


def test_distance_to_nearer_cbd_is_the_minimum(locality):
    row = build_row(PropertyInput(locality, 900, 2, 2))
    if not np.isnan(row.dist_cbd_km.iloc[0]):
        assert row.dist_cbd_km.iloc[0] == pytest.approx(
            min(row.dist_nariman_km.iloc[0], row.dist_bkc_km.iloc[0]))


# ---------------------------------------------------------------------------
# PREDICTIONS — sanity, not exact values
# ---------------------------------------------------------------------------

def test_prediction_is_a_plausible_price(locality):
    r = predict_price(locality=locality, area_sqft=900, bedrooms=2,
                      bathrooms=2)
    assert 5 * LAKH < r["estimate"] < 100 * CRORE
    assert 1_000 < r["price_per_sqft"] < 300_000


def test_the_range_brackets_the_estimate(locality):
    r = predict_price(locality=locality, area_sqft=900, bedrooms=2,
                      bathrooms=2)
    assert r["range_low"] < r["estimate"] < r["range_high"]
    assert r["typical_error_pct"] > 0


def test_bigger_flat_costs_more_in_the_same_locality(locality):
    """A monotonicity check. The model is free to be non-linear, but a
    1,500 sqft flat costing less than a 600 sqft one in the same area would
    mean something is badly wrong."""
    small = predict_price(locality=locality, area_sqft=600, bedrooms=1,
                          bathrooms=1)["estimate"]
    large = predict_price(locality=locality, area_sqft=1500, bedrooms=3,
                          bathrooms=3)["estimate"]
    assert large > small


def test_expensive_locality_costs_more_than_a_cheap_one():
    """Same flat, two very different areas. If this fails, the location
    features are not reaching the model."""
    known = set(list_localities())
    pricey = next((x for x in ["Bandra", "Lower Parel", "Worli", "Juhu"]
                   if x in known), None)
    cheap = next((x for x in ["Badlapur", "Neral", "Karjat", "Ambernath"]
                  if x in known), None)
    if not pricey or not cheap:
        pytest.skip("expected localities not in this model")
    a = predict_price(locality=pricey, area_sqft=900, bedrooms=2,
                      bathrooms=2)["estimate"]
    b = predict_price(locality=cheap, area_sqft=900, bedrooms=2,
                      bathrooms=2)["estimate"]
    assert a > b * 2, f"{pricey} should be far above {cheap}"


def test_predictions_are_deterministic(locality):
    """Same input, same answer, every time. A user who reloads the page must
    not see a different price."""
    args = dict(locality=locality, area_sqft=900, bedrooms=2, bathrooms=2)
    assert predict_price(**args)["estimate"] == predict_price(**args)["estimate"]


# ---------------------------------------------------------------------------
# WARNINGS — the honesty layer
# ---------------------------------------------------------------------------

def test_large_property_gets_a_warning(locality):
    r = predict_price(locality=locality, area_sqft=4000, bedrooms=4,
                      bathrooms=4)
    assert r["warnings"], "a 4,000 sqft 4 BHK must carry a warning"
    assert r["typical_error_pct"] >= 18


def test_ordinary_property_is_not_over_warned(locality):
    r = predict_price(locality=locality, area_sqft=750, bedrooms=2,
                      bathrooms=2)
    assert r["typical_error_pct"] < 16


def test_rare_property_types_are_flagged(locality):
    r = predict_price(locality=locality, area_sqft=1200, bedrooms=2,
                      bathrooms=2, property_type="Independent Floor")
    assert any("Independent Floor" in w for w in r["warnings"])


def test_every_result_carries_the_disclaimer(locality):
    r = predict_price(locality=locality, area_sqft=900, bedrooms=2,
                      bathrooms=2)
    assert "asking" in r["disclaimer"].lower()


# ---------------------------------------------------------------------------
# THE SAVED MODEL
# ---------------------------------------------------------------------------

def test_model_loads_and_predicts(model, train_sample):
    from preprocessing import to_price
    pred = to_price(model.predict(train_sample[FEATURES]),
                    train_sample.area.values)
    assert len(pred) == len(train_sample)
    assert (pred > 0).all()


def test_model_handles_missing_coordinates(model, locality):
    """151 training rows had no coordinates, so the imputer inside the
    pipeline must handle them at prediction time too."""
    row = build_row(PropertyInput(locality, 900, 2, 2))
    row.loc[:, ["latitude", "longitude", "dist_nariman_km",
                "dist_bkc_km", "dist_cbd_km"]] = np.nan
    assert model.predict(row)[0] > 0


def test_model_survives_an_unseen_category(model, locality):
    """handle_unknown='infrequent_if_exist' is what keeps the app alive when
    a locality the model never saw arrives. Without it: an exception."""
    row = build_row(PropertyInput(locality, 900, 2, 2))
    row.loc[:, "locality_grouped"] = "Some Locality That Does Not Exist"
    assert model.predict(row)[0] > 0


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (5_000_000, "Rs 50.0 lakh"),
    (12_000_000, "Rs 1.20 crore"),
    (250_000_000, "Rs 25.00 crore"),
])
def test_currency_formatting(value, expected):
    assert format_inr(value) == expected
