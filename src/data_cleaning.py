"""
data_cleaning.py — Phase 4: reusable cleaning functions.

Design rules followed here:
  1. Every transformation is a small, separately-testable function.
  2. No function mutates its input. Each returns a NEW DataFrame (df.copy()).
     This means you can re-run any step without corrupting earlier state --
     which matters enormously inside a Jupyter notebook, where cells run
     out of order.
  3. Thresholds live in named constants at the top, not buried in the code.
     Anyone reviewing this project can see and challenge every judgement call.
  4. Nothing is hard-coded to specific row indices. Fixing individual rows
     by hand does not survive a dataset update.

Usage:
    python src/data_cleaning.py data/raw/mumbai-house-price-data-raw.csv \
                                data/processed/mumbai_clean.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TUNABLE CONSTANTS — every one of these is a judgement call, documented.
# ---------------------------------------------------------------------------

# Signed 32-bit integer maximum. Appears in this dataset as a price, which is
# impossible -- it is a scraper overflow or a "no value" sentinel.
INT32_MAX = 2_147_483_647

# Plausible price-per-sqft band for the Mumbai Metropolitan Region, in INR.
# Deliberately WIDE. The aim is to remove impossible rows (Rs 25/sqft), not to
# remove genuinely expensive property. Narrowing these would delete real
# luxury listings, which the model needs to see.
MIN_PRICE_PER_SQFT = 2_000
MAX_PRICE_PER_SQFT = 150_000

# Plausible floor-area band in sqft. Below 150 is smaller than a legal
# habitable room; above 15,000 is not a flat.
MIN_AREA_SQFT = 150
MAX_AREA_SQFT = 15_000

# Generous bounding box around the Mumbai Metropolitan Region.
# Chosen wide ON PURPOSE: a tight box around Mumbai city wrongly flags real
# outer-MMR localities such as Palghar, Saphale and Vevoor. This box keeps
# those and catches only coordinates from entirely different cities
# (Delhi ~28.5/77.1, Kolkata ~22.7/88.4, Pune ~18.5/73.8).
LAT_MIN, LAT_MAX = 18.0, 20.5
LON_MIN, LON_MAX = 72.0, 73.5

# Columns removed, with the reason. Reasons are kept in code so the decision
# is visible at the point of action, not only in a separate report.
COLUMNS_TO_DROP = {
    "price_per_sqft": "TARGET LEAKAGE: equals price/area for 100% of rows",
    "total_floors": "NO INFORMATION: 99.9% of rows contain the value 1",
    "city": "INCONSISTENT: mixes a city name with zone names; 98.5% one value",
}

# Text values that mean 'missing' but are not stored as NaN.
BLANK_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "unknown"}


# ---------------------------------------------------------------------------
# INDIVIDUAL CLEANING STEPS
# ---------------------------------------------------------------------------

def standardise_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in text columns and convert blank placeholders to NaN.

    Why: `df.isna()` reported zero missing values, but 16.1% of `title` was an
    empty string. Converting these to real NaN makes the missingness visible
    to every later step (and to scikit-learn's imputers in Phase 14).
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        s = out[col].astype("string").str.strip()
        # Collapse runs of internal whitespace: "Sector  19" -> "Sector 19"
        s = s.str.replace(r"\s+", " ", regex=True)
        out[col] = s.mask(s.str.lower().isin(BLANK_TOKENS))
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows identical across every column.

    Why this must happen FIRST and BEFORE the train/test split: if the same
    listing lands in both train and test, the model is graded on rows it
    memorised. Test scores become fiction. See Phase 9.
    """
    return df.drop_duplicates().reset_index(drop=True)


def normalise_locality(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise locality names.

    This dataset's localities are already unusually tidy: no case variants, no
    stray whitespace. The only irregularity is 8 rows shaped like
    "Siddharth Nagar, Goregaon" -- a sub-area plus its parent area. We keep the
    part after the last comma (the broader, better-populated area) so those
    rows join an existing category instead of forming a category of one.
    """
    out = df.copy()
    if "locality" not in out.columns:
        return out
    out["locality"] = (
        out["locality"]
        .astype("string")
        .str.split(",")
        .str[-1]          # take the broader area
        .str.strip()
    )
    return out


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Null out latitude/longitude that fall outside the MMR bounding box.

    JUDGEMENT CALL, and a real trade-off:
      Option A - drop the whole row. Costs us ~154 otherwise-valid properties.
      Option B - keep the row, null only the coordinates. Costs us nothing
                 except that geography becomes missing for those rows, which
                 the Phase 14 imputer can handle.

    We choose B. The rule is: one bad field should not destroy a good record.
    Price, area, BHK and locality on these rows are all fine, and locality
    already carries most of the geographic signal anyway.
    """
    out = df.copy()
    if not {"latitude", "longitude"}.issubset(out.columns):
        return out

    valid = (
        out["latitude"].between(LAT_MIN, LAT_MAX)
        & out["longitude"].between(LON_MIN, LON_MAX)
    )
    # .loc with a boolean mask assigns to the original frame, avoiding the
    # chained-assignment trap (df[mask]["col"] = x silently does nothing).
    out.loc[~valid, ["latitude", "longitude"]] = np.nan
    return out


def remove_impossible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose values cannot describe a real property.

    Note what we do NOT drop: rows with bedroom_num == 0. Both such rows are
    Studio Apartments, where zero bedrooms is correct. A naive
    "drop rows where bedrooms == 0" rule would delete valid data. Always
    check WHY a value looks wrong before deleting it.
    """
    out = df.copy()

    price_per_sqft = out["price"] / out["area"]

    keep = (
        (out["price"] != INT32_MAX)
        & out["area"].between(MIN_AREA_SQFT, MAX_AREA_SQFT)
        & price_per_sqft.between(MIN_PRICE_PER_SQFT, MAX_PRICE_PER_SQFT)
    )
    return out.loc[keep].reset_index(drop=True)


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Flag values that are probably 'not stated' rather than genuinely zero.

    60.8% of `age` is 0 and 90.8% of `balcony_num` is 0. Two readings exist for
    each -- a genuinely new building / a flat with no balcony, OR a blank field
    that the scraper filled with zero. The dataset ships no documentation, so
    WE CANNOT RESOLVE THIS. Asserting either meaning would be inventing data.

    So we keep the original number AND add a boolean flag. The model can then
    learn whatever the zero actually signals, instead of us deciding for it.
    This is the standard "missing indicator" pattern.
    """
    out = df.copy()
    if "age" in out.columns:
        out["age_is_zero"] = (out["age"] == 0).astype(int)
    if "balcony_num" in out.columns:
        out["has_balcony"] = (out["balcony_num"] > 0).astype(int)
    return out


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leaking and information-free columns."""
    out = df.copy()
    present = [c for c in COLUMNS_TO_DROP if c in out.columns]
    return out.drop(columns=present)


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run every cleaning step in order and report what each one did.

    ORDER MATTERS:
      - standardise_text first, so blank strings become NaN before anything
        tries to group or compare on them.
      - drop_exact_duplicates early, so later steps do 28% less work and the
        row counts we report are honest.
      - remove_impossible_rows BEFORE dropping price_per_sqft, because the
        plausibility check needs price and area, and we still have both.
      - drop_unusable_columns LAST, so every earlier step could use the
        columns it needed.
    """
    steps = [
        ("standardise text", standardise_text),
        ("drop exact duplicates", drop_exact_duplicates),
        ("normalise locality", normalise_locality),
        ("fix coordinates", fix_coordinates),
        ("remove impossible rows", remove_impossible_rows),
        ("add missing indicators", add_missing_indicators),
        ("drop unusable columns", drop_unusable_columns),
        # A SECOND dedup, on purpose, and it must come last.
        # Dropping price_per_sqft / total_floors / city can make two rows
        # that previously differed become identical. Found by an assert while
        # writing the script template in Phase 7: 9 such rows exist.
        # Small, but the rule is that a cleaning step which can create
        # duplicates must be followed by a step that removes them.
        ("dedup again after drops", drop_exact_duplicates),
    ]

    out = df
    if verbose:
        print(f"{'step':<26}{'rows':>10}{'cols':>7}  change")
        print("-" * 60)
        print(f"{'RAW INPUT':<26}{len(out):>10,}{out.shape[1]:>7}")

    for name, fn in steps:
        before_rows, before_cols = out.shape
        out = fn(out)
        after_rows, after_cols = out.shape
        if verbose:
            dr = after_rows - before_rows
            dc = after_cols - before_cols
            change = ""
            if dr:
                change += f"{dr:+,} rows "
            if dc:
                change += f"{dc:+d} cols"
            print(f"{name:<26}{after_rows:>10,}{after_cols:>7}  {change}")

    if verbose:
        print("-" * 60)
        kept = len(out) / len(df) * 100
        print(f"Kept {len(out):,} of {len(df):,} rows ({kept:.1f}%)")
        print("\nMissing values after cleaning:")
        na = out.isna().sum()
        na = na[na > 0]
        print(na.to_string() if len(na) else "  none")

    return out


def main(raw_path: str, out_path: str) -> int:
    raw = Path(raw_path)
    if not raw.exists():
        print(f"ERROR: file not found: {raw}")
        return 1

    df = pd.read_csv(raw, low_memory=False)
    clean = clean_dataset(df)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    print(f"\nWritten: {out}  ({out.stat().st_size / 1024**2:.1f} MB)")

    # Safety net: assert the leaking column really is gone. An assert that
    # never fires still documents an invariant, and it WILL fire the day
    # someone edits COLUMNS_TO_DROP carelessly.
    assert "price_per_sqft" not in clean.columns, "LEAKAGE: price_per_sqft survived"
    print("Leakage check passed: price_per_sqft is absent.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
