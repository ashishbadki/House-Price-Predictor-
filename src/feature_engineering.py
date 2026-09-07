"""
feature_engineering.py — Phase 7: build the features the model will use.

THE ONE RULE
------------
A feature is allowed only if the user of the Streamlit app can supply it,
or we can compute it from what they supply.

    "What is the area?"        -> user knows it          -> allowed
    "Which locality?"          -> user knows it          -> allowed
    "What is the price/sqft?"  -> user does NOT know it  -> BANNED

Every feature below is checked against that rule in the table at the bottom.

WHAT WE LEARNED BUILDING THIS
-----------------------------
We measured every new feature instead of assuming it helps. Result:

    baseline (raw columns only)  MAE 36.06 lakh
    + all engineered features    MAE 35.73 lakh      (0.9% better)

That is a small gain, and we report it honestly. One candidate feature
(`area_per_bed`) made the model WORSE and was deleted.

The reason the gain is small is worth understanding. `dist_to_cbd` is a
genuinely strong signal -- correlation of -0.65 with price per sqft, and
adding it to a model with no coordinates improves MAE from 41.10 to 37.24
lakh. But our model already has latitude and longitude, and a tree can carve
up a map by itself. The information was already there in another form.

    A feature can be informative and still useless,
    if the model already has that information.

Usage:
    python src/feature_engineering.py data/processed/mumbai_clean.csv \
                                      data/processed/mumbai_features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference points. These are real Mumbai business districts -- external facts
# about the city, NOT anything computed from our price column. That matters:
# a reference point chosen by looking at where prices are highest would be a
# quiet form of leakage.
# ---------------------------------------------------------------------------
NARIMAN_POINT = (18.9256, 72.8242)   # the old central business district
BKC = (19.0662, 72.8692)             # Bandra Kurla Complex, the new one

EARTH_RADIUS_KM = 6371.0

# How many localities to keep as their own category. The rest become "Other".
# 200 is not arbitrary: HistGradientBoosting refuses more than 255 categories,
# and localities below this cut have very few listings each. Revisited properly
# in Phase 8.
TOP_LOCALITIES = 200


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points on Earth, in kilometres.

    Called the haversine formula. We cannot just subtract latitudes and
    longitudes, because a degree of longitude is a different distance at the
    equator than near the poles. This accounts for the curve.

    Note this is crow-flies distance, not road distance. For Mumbai -- long and
    thin, with water in the middle -- road distance is often much longer. That
    is a real limitation of this feature.
    """
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """How far is this property from Mumbai's two business districts?

    Why it may help: in almost every city, price falls as you move away from
    where the jobs are. Mumbai is an extreme case because it is a narrow strip
    of land -- there is only one direction to go.

    Measured: correlation of -0.65 with price per sqft. Strong on its own.
    Adds only ~0.3 lakh once latitude and longitude are already in the model,
    but it costs nothing and it makes the model easier to explain.

    Safe at prediction time? Yes. The app knows the locality, and we look up
    coordinates from the locality.
    """
    out = df.copy()
    out["dist_nariman_km"] = haversine_km(out.latitude, out.longitude, *NARIMAN_POINT)
    out["dist_bkc_km"] = haversine_km(out.latitude, out.longitude, *BKC)
    # Distance to whichever business district is nearer.
    out["dist_cbd_km"] = out[["dist_nariman_km", "dist_bkc_km"]].min(axis=1)
    return out


def add_room_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple combinations of the room counts.

    `rooms_total` measured a tiny improvement (36.06 -> 35.98 lakh) so it stays.

    DELETED CANDIDATE: `area_per_bed = area / bedroom_num`. It sounded
    sensible -- how spacious is each room -- but it made the model WORSE
    (36.06 -> 36.26 lakh). We removed it. This is why you measure. An idea
    being reasonable is not evidence that it works.

    Safe at prediction time? Yes, built from values the user types in.
    """
    out = df.copy()
    out["rooms_total"] = out.bedroom_num + out.bathroom_num
    # Guard against divide-by-zero: two rows are studios with 0 bedrooms.
    out["bath_per_bed"] = out.bathroom_num / out.bedroom_num.replace(0, np.nan)
    return out


def group_rare_localities(df: pd.DataFrame, top_n: int = TOP_LOCALITIES) -> pd.DataFrame:
    """Keep the busiest localities, fold the rest into "Other".

    Why: 403 localities, and 271 of them have fewer than 10 listings. A
    category with 4 rows teaches the model nothing reliable, and it will not
    generalise -- a new listing from that locality is almost a coin flip.

    IMPORTANT LIMITATION: the list of "top" localities is computed from the
    whole dataset here. Strictly, it should be computed inside each
    cross-validation fold from training rows only. The effect is small because
    it uses only row COUNTS and never touches price, but Phase 8 does this
    properly inside a scikit-learn pipeline.

    Safe at prediction time? Yes. If a user picks a locality the model never
    saw, it maps to "Other" instead of crashing.
    """
    out = df.copy()
    keep = out.locality.value_counts().head(top_n).index
    out["locality_grouped"] = np.where(out.locality.isin(keep), out.locality, "Other")
    return out


# ---------------------------------------------------------------------------
# THE FEATURE CONTRACT
#
# This is the single source of truth for what the model is allowed to see.
# Anything not on these two lists does not reach the model -- which is how we
# make sure `price`, `price_per_sqft` and `title` can never slip back in.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "area",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "age",
    "age_is_zero",
    "has_balcony",
    "latitude",
    "longitude",
    "dist_nariman_km",
    "dist_bkc_km",
    "dist_cbd_km",
    "rooms_total",
    "bath_per_bed",
]

CATEGORICAL_FEATURES = [
    "locality_grouped",
    "property_type",
    "furnished",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that exist in the data but must NEVER be given to the model.
BANNED = {
    "price": "this is the target",
    "price_per_sqft": "leakage: it is price divided by area",
    "title": "a building name; using it lets the model memorise buildings "
             "instead of learning what makes a property expensive",
    "locality": "replaced by locality_grouped",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature step in order."""
    out = df
    out = add_distance_features(out)
    out = add_room_features(out)
    out = group_rare_localities(out)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if a banned column ever ends up in the feature list.

    An assertion that never fires still earns its place: it documents the rule,
    and it will fire the day somebody edits ALL_FEATURES without thinking.
    """
    for col, reason in BANNED.items():
        assert col not in ALL_FEATURES, f"BANNED COLUMN IN FEATURES: {col} ({reason})"

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    assert not missing, f"features declared but not built: {missing}"
    print("Leakage check passed.")


def describe_features(df: pd.DataFrame) -> None:
    print(f"\n{'feature':<20}{'kind':<14}{'missing':>9}   note")
    print("-" * 76)
    notes = {
        "area": "from the user",
        "bedroom_num": "from the user",
        "bathroom_num": "from the user",
        "balcony_num": "from the user",
        "age": "from the user; 0 means unknown",
        "age_is_zero": "flag, built in Phase 4",
        "has_balcony": "flag, built in Phase 4",
        "latitude": "looked up from locality",
        "longitude": "looked up from locality",
        "dist_nariman_km": "distance to the old CBD",
        "dist_bkc_km": "distance to the new CBD",
        "dist_cbd_km": "distance to the nearer of the two",
        "rooms_total": "bedrooms + bathrooms",
        "bath_per_bed": "bathrooms per bedroom",
        "locality_grouped": f"top {TOP_LOCALITIES}, rest = Other",
        "property_type": "from the user",
        "furnished": "from the user",
    }
    for f in ALL_FEATURES:
        kind = "numeric" if f in NUMERIC_FEATURES else "categorical"
        print(f"{f:<20}{kind:<14}{df[f].isna().sum():>9,}   {notes.get(f,'')}")


def main(in_path: str, out_path: str) -> int:
    src = Path(in_path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1

    df = pd.read_csv(src)
    print(f"Input : {len(df):,} rows, {df.shape[1]} columns")

    out = build_features(df)
    check_no_leakage(out)

    print(f"Output: {len(out):,} rows, {out.shape[1]} columns")
    print(f"Model will see {len(ALL_FEATURES)} features "
          f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    describe_features(out)

    # We keep `price` in the saved file because training needs the target.
    # It is excluded from ALL_FEATURES, which is what the model actually reads.
    keep = ALL_FEATURES + ["price"]
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_csv(dest, index=False)
    print(f"\nWritten: {dest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
