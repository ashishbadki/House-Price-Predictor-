"""
predict.py — Phase 20: turn a user's answers into a price estimate.

This module is the boundary between the model and the outside world. It has
three jobs, and the second one is the reason it exists as a separate file:

  1. Validate what the user typed.
  2. Build the exact row shape the pipeline was trained on.
  3. Attach honest warnings and a range, based on Phase 18's measurements.

WHY VALIDATION IS NOT OPTIONAL
------------------------------
A trained model has no opinions. Ask it for a 5 sqft, 40 BHK flat and it will
return a number, calmly and with no complaint. Ask it for a negative area and
it will return a negative price.

Every one of those is a bug the user sees. The model cannot catch them; only
this layer can.

WHY WE BUILD THE ROW HERE AND NOWHERE ELSE
------------------------------------------
The saved pipeline expects 17 columns with exact names. The user gives us six
or seven answers. Something must fill the gap -- look up the locality's
coordinates, compute the distance features, derive `rooms_total`.

That work happens ONCE, here, using the same functions the training data used.
If the Streamlit app did it inline, the app and the training script would
slowly drift apart and the model would start receiving numbers that mean
something other than what it learned.

Usage:
    from src.predict import predict_price, list_localities

    result = predict_price(locality="Andheri", area_sqft=900, bedrooms=2,
                           bathrooms=2)
    print(result["estimate_text"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
# Put src/ on the import path before importing its siblings.
#
# Without this, `python src/predict.py` works (Python adds a script's own
# folder automatically) but `from src.predict import predict_price` fails --
# because then Python treats src as a package and `feature_engineering` is no
# longer a top-level module. The README shows the second form, so both have
# to work. Same bootstrap as tune.py and train_final.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import BKC, NARIMAN_POINT, haversine_km
from preprocessing import FEATURES

# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
LOCALITY_PATH = MODEL_DIR / "locality_reference.json"
CARD_PATH = MODEL_DIR / "model_card.json"

LAKH = 100_000
CRORE = 10_000_000

# Limits. Chosen to match the range the model actually saw in training --
# outside these the model is extrapolating, which it cannot do.
MIN_AREA, MAX_AREA = 100, 20_000
MAX_BEDROOMS = 15
MAX_BATHROOMS = 15
MAX_BALCONIES = 10
MAX_AGE = 100

VALID_PROPERTY_TYPES = ["Apartment", "Independent House", "Villa",
                        "Independent Floor", "Studio Apartment"]
VALID_FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]

# From Phase 18's test-set results, by segment. These are measured numbers,
# not guesses, and they are what the range and the warnings are built from.
SEGMENT_ERROR = {
    "default": 13.2,
    "under_2cr": 12.6,
    "2cr_to_10cr": 14.5,
    "over_10cr": 19.6,
    "area_over_2500": 21.4,
    "bhk_4_plus": 18.3,
}


class InvalidInput(ValueError):
    """Raised when the user's input cannot describe a real property."""


@dataclass
class PropertyInput:
    """Everything the app asks a user for.

    Defaults match the most common values in the training data, so a user who
    leaves a field alone still gets a sensible answer.
    """
    locality: str
    area_sqft: float
    bedrooms: int
    bathrooms: int
    balconies: int = 0
    age_years: int = 0            # 0 also means "new or not known"
    property_type: str = "Apartment"
    furnishing: str = "Unfurnished"


# ---------------------------------------------------------------------------
# 1. VALIDATION
# ---------------------------------------------------------------------------

_localities: dict | None = None


def _load_localities() -> dict:
    global _localities
    if _localities is None:
        if not LOCALITY_PATH.exists():
            raise FileNotFoundError(
                f"{LOCALITY_PATH} not found. Run src/train_final.py first.")
        _localities = json.loads(LOCALITY_PATH.read_text())
    return _localities


def list_localities() -> list[str]:
    """Every locality the model knows, most-listed first.

    The app uses this for a dropdown rather than a text box. A dropdown makes
    a whole class of invalid input impossible, which beats validating it.
    """
    ref = _load_localities()
    return sorted(ref, key=lambda k: -ref[k]["n"])


def validate(p: PropertyInput) -> None:
    """Reject anything that cannot describe a real property.

    Each check raises with a message written for a user, not a developer.
    "Area must be a positive number" is useful. "ValueError at line 88" is not.
    """
    ref = _load_localities()

    if not p.locality or p.locality not in ref:
        raise InvalidInput(
            f"'{p.locality}' is not a locality this model knows. "
            f"Pick one of the {len(ref)} available.")

    # Note `bool` is a subclass of int in Python, so True would pass an
    # isinstance(x, int) check. Hence the explicit exclusion.
    if isinstance(p.area_sqft, bool) or not isinstance(p.area_sqft, (int, float)):
        raise InvalidInput("Area must be a number.")
    if not np.isfinite(p.area_sqft) or p.area_sqft <= 0:
        raise InvalidInput("Area must be a positive number.")
    if not MIN_AREA <= p.area_sqft <= MAX_AREA:
        raise InvalidInput(
            f"Area must be between {MIN_AREA:,} and {MAX_AREA:,} sqft. "
            f"Outside that range the model has no training data to learn from.")

    if isinstance(p.bedrooms, bool) or not isinstance(p.bedrooms, int):
        raise InvalidInput("Bedrooms must be a whole number.")
    if p.bedrooms < 0 or p.bedrooms > MAX_BEDROOMS:
        raise InvalidInput(f"Bedrooms must be between 0 and {MAX_BEDROOMS}. "
                           f"Use 0 only for a studio apartment.")

    if isinstance(p.bathrooms, bool) or not isinstance(p.bathrooms, int):
        raise InvalidInput("Bathrooms must be a whole number.")
    if p.bathrooms < 1 or p.bathrooms > MAX_BATHROOMS:
        raise InvalidInput(f"Bathrooms must be between 1 and {MAX_BATHROOMS}. "
                           f"Every property has at least one.")

    if not 0 <= p.balconies <= MAX_BALCONIES:
        raise InvalidInput(f"Balconies must be between 0 and {MAX_BALCONIES}.")

    if not 0 <= p.age_years <= MAX_AGE:
        raise InvalidInput(f"Age must be between 0 and {MAX_AGE} years.")

    if p.property_type not in VALID_PROPERTY_TYPES:
        raise InvalidInput(
            f"Property type must be one of: {', '.join(VALID_PROPERTY_TYPES)}")

    if p.furnishing not in VALID_FURNISHING:
        raise InvalidInput(
            f"Furnishing must be one of: {', '.join(VALID_FURNISHING)}")

    # Not impossible, but almost certainly a typo, and the model has no
    # examples like it. Better to ask than to answer confidently.
    if p.bedrooms > 0 and p.bathrooms > p.bedrooms + 3:
        raise InvalidInput(
            f"{p.bathrooms} bathrooms for {p.bedrooms} bedrooms looks like a "
            f"typo. Please check.")
    if p.bedrooms >= 2 and p.area_sqft < 200:
        raise InvalidInput(
            f"{p.area_sqft:.0f} sqft is too small for {p.bedrooms} bedrooms. "
            f"Please check the area.")


# ---------------------------------------------------------------------------
# 2. BUILD THE ROW
# ---------------------------------------------------------------------------

def build_row(p: PropertyInput) -> pd.DataFrame:
    """Turn the user's answers into the exact 17-column frame the model wants.

    Everything here mirrors `feature_engineering.py`. It has to: the model
    learned from columns built by those functions, so anything different is a
    silent mismatch.
    """
    ref = _load_localities()[p.locality]
    lat, lon = ref["lat"], ref["lon"]

    if lat is None or lon is None:
        # One locality (Jambrung) has no usable coordinates. NaN is correct
        # here -- the pipeline's imputer handles it, exactly as it did for the
        # 151 training rows in the same position.
        lat = lon = np.nan
        d_nariman = d_bkc = d_cbd = np.nan
    else:
        d_nariman = haversine_km(lat, lon, *NARIMAN_POINT)
        d_bkc = haversine_km(lat, lon, *BKC)
        d_cbd = min(d_nariman, d_bkc)

    row = {
        "area": float(p.area_sqft),
        "bedroom_num": int(p.bedrooms),
        "bathroom_num": int(p.bathrooms),
        "balcony_num": int(p.balconies),
        "age": int(p.age_years),
        "age_is_zero": int(p.age_years == 0),
        "has_balcony": int(p.balconies > 0),
        "latitude": lat,
        "longitude": lon,
        "dist_nariman_km": d_nariman,
        "dist_bkc_km": d_bkc,
        "dist_cbd_km": d_cbd,
        "rooms_total": int(p.bedrooms) + int(p.bathrooms),
        "bath_per_bed": (p.bathrooms / p.bedrooms) if p.bedrooms > 0 else np.nan,
        "locality_grouped": p.locality,
        "property_type": p.property_type,
        "furnishing_placeholder": None,   # replaced below, keeps order obvious
        "furnished": p.furnishing,
    }
    row.pop("furnishing_placeholder")

    # Reindex to FEATURES so the column ORDER matches training exactly.
    # ColumnTransformer selects by name, so order is not strictly required --
    # but relying on that is fragile, and this line makes a missing column
    # fail loudly instead of quietly becoming NaN.
    frame = pd.DataFrame([row])
    missing = [c for c in FEATURES if c not in frame.columns]
    assert not missing, f"build_row did not produce: {missing}"
    return frame[FEATURES]


# ---------------------------------------------------------------------------
# 3. PREDICT, WITH A RANGE AND WARNINGS
# ---------------------------------------------------------------------------

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run src/train_final.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def _segment_and_warnings(p: PropertyInput, price: float):
    """Pick the error rate that applies, and say why.

    Phase 16 and Phase 18 both found the model is far less reliable on large
    and expensive property. Knowing that and not telling the user would be
    the dishonest choice.
    """
    warnings_ = []
    error_pct = SEGMENT_ERROR["default"]

    if price >= 10 * CRORE:
        error_pct = max(error_pct, SEGMENT_ERROR["over_10cr"])
        warnings_.append(
            "Above Rs 10 crore the model had only 175 test examples and its "
            "typical error rises to about 20%. Treat this as a rough "
            "indication only.")
    elif price >= 2 * CRORE:
        error_pct = max(error_pct, SEGMENT_ERROR["2cr_to_10cr"])
    else:
        error_pct = min(error_pct, SEGMENT_ERROR["under_2cr"])

    if p.area_sqft > 2500:
        error_pct = max(error_pct, SEGMENT_ERROR["area_over_2500"])
        warnings_.append(
            "Above 2,500 sqft the typical error is about 21%. Very large "
            "properties vary by things this model cannot see -- sea view, "
            "floor height, building prestige.")

    if p.bedrooms >= 4:
        error_pct = max(error_pct, SEGMENT_ERROR["bhk_4_plus"])
        warnings_.append(
            "4 BHK and larger had fewer training examples; typical error "
            "about 18%.")

    if p.property_type in ("Independent Floor", "Villa"):
        warnings_.append(
            f"'{p.property_type}' had under 300 training examples out of "
            f"40,826. This estimate is much less reliable than for an "
            f"apartment.")

    ref = _load_localities()[p.locality]
    if ref["n"] < 50:
        warnings_.append(
            f"{p.locality} has only {ref['n']} listings in the training data, "
            f"so the model knows this area poorly.")

    return error_pct, warnings_


def format_inr(x: float) -> str:
    """Rupees in the units Indians actually use."""
    if x >= CRORE:
        return f"Rs {x/CRORE:.2f} crore"
    return f"Rs {x/LAKH:.1f} lakh"


def predict_price(locality: str, area_sqft: float, bedrooms: int,
                  bathrooms: int, balconies: int = 0, age_years: int = 0,
                  property_type: str = "Apartment",
                  furnishing: str = "Unfurnished") -> dict:
    """Estimate a price. Raises InvalidInput on bad input.

    Returns the point estimate, a range, the warnings that apply, and the
    locality's own median rate for context -- because a number with nothing
    to compare it to is hard to judge.
    """
    p = PropertyInput(locality, area_sqft, bedrooms, bathrooms, balconies,
                      age_years, property_type, furnishing)
    validate(p)

    model = load_model()
    row = build_row(p)

    rate = float(model.predict(row)[0])       # rupees per sqft
    price = rate * p.area_sqft                # Phase 6: multiply by area

    error_pct, warns = _segment_and_warnings(p, price)
    low, high = price * (1 - error_pct / 100), price * (1 + error_pct / 100)

    ref = _load_localities()[p.locality]

    return {
        "estimate": price,
        "estimate_text": format_inr(price),
        "range_low": low,
        "range_high": high,
        "range_text": f"{format_inr(low)} to {format_inr(high)}",
        "typical_error_pct": error_pct,
        "price_per_sqft": rate,
        "locality_median_rate": ref["median_rate"],
        "locality_listings": ref["n"],
        "warnings": warns,
        "disclaimer": (
            "This is a statistical estimate built from listing (asking) "
            "prices, not confirmed sale prices. Actual transaction prices "
            "are usually lower and can differ substantially."),
    }


if __name__ == "__main__":
    print(f"{len(list_localities())} localities available\n")
    for case in [
        dict(locality="Andheri", area_sqft=900, bedrooms=2, bathrooms=2),
        dict(locality="Bandra", area_sqft=1800, bedrooms=3, bathrooms=3,
             furnishing="Furnished"),
        dict(locality="Badlapur", area_sqft=550, bedrooms=1, bathrooms=1),
        dict(locality="Malabar Hill", area_sqft=4500, bedrooms=4, bathrooms=5),
    ]:
        r = predict_price(**case)
        print(f"{case['locality']:<14} {case['area_sqft']:>5.0f} sqft "
              f"{case['bedrooms']}BHK -> {r['estimate_text']:<18} "
              f"(+/-{r['typical_error_pct']}%)  "
              f"rate {r['price_per_sqft']:,.0f} vs locality median "
              f"{r['locality_median_rate']:,}")
        for w in r["warnings"]:
            print(f"      ! {w[:90]}")

    print("\nRejected inputs:")
    for bad in [
        dict(locality="Andheri", area_sqft=-100, bedrooms=2, bathrooms=2),
        dict(locality="Andheri", area_sqft=900, bedrooms=2, bathrooms=0),
        dict(locality="Paris", area_sqft=900, bedrooms=2, bathrooms=2),
        dict(locality="Andheri", area_sqft=150, bedrooms=3, bathrooms=2),
        dict(locality="Andheri", area_sqft=900, bedrooms=1, bathrooms=9),
    ]:
        try:
            predict_price(**bad)
            print(f"  NOT CAUGHT: {bad}")
        except InvalidInput as e:
            print(f"  caught: {e}")
