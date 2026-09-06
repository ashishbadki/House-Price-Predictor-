"""
inspect_data.py — Phase 3: dataset inspection.

Purpose: describe a raw CSV honestly, before any cleaning decisions are made.
This script NEVER modifies the data. It only reports.

Why a script instead of typing commands in a notebook?
  - It is reproducible: anyone can re-run it and get the same report.
  - It works on any CSV, so we can re-run it in Phase 4 to confirm cleaning worked.
  - It is version-controlled, so the report is reviewable in Git.

Usage:
    python src/inspect_data.py data/raw/mumbai-house-price-data-raw.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Show wide tables without pandas truncating them into "..."
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def rule(title: str) -> None:
    """Print a section header. Purely cosmetic, but a long report is
    unreadable without visual structure."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def overview(df: pd.DataFrame) -> None:
    rule("1. SHAPE")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    # memory_usage(deep=True) follows pointers into Python string objects.
    # Without deep=True, object columns report only the pointer size and you
    # get a wildly optimistic number.
    mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory:  {mb:.1f} MB")


def column_profile(df: pd.DataFrame) -> None:
    rule("2. COLUMN PROFILE")
    profile = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "null_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
            "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None
                        for c in df.columns],
        }
    )
    print(profile)


def hidden_missing(df: pd.DataFrame) -> None:
    """`nulls = 0` does NOT mean 'no missing data'.

    Datasets frequently encode 'unknown' as an empty string, the text "NA",
    or the number 0. pandas counts none of those as null. This function looks
    for them explicitly."""
    rule("3. HIDDEN MISSING VALUES (not counted as null)")

    sentinel_text = {"", " ", "na", "n/a", "nan", "none", "null", "-", "unknown"}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype(str).str.strip().str.lower()
        hits = s.isin(sentinel_text).sum()
        if hits:
            print(f"  {col:<16} {hits:>7,} blank/placeholder ({hits/len(df)*100:.1f}%)")

    for col in df.select_dtypes(include=[np.number]).columns:
        zeros = (df[col] == 0).sum()
        # Only flag zero-heavy columns. A zero can be legitimate (0 balconies),
        # so this is a prompt to investigate, not a verdict.
        if zeros / len(df) > 0.20:
            print(f"  {col:<16} {zeros:>7,} zeros ({zeros/len(df)*100:.1f}%) "
                  f"-- legitimate value, or code for 'not stated'?")


def duplicates(df: pd.DataFrame) -> None:
    rule("4. DUPLICATES")
    exact = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact:,} ({exact/len(df)*100:.1f}%)")
    print(f"Rows remaining after dropping them: {len(df) - exact:,}")

    if exact:
        # How concentrated is the duplication? One row copied 50 times is a
        # different problem from 50 rows copied twice.
        counts = df.groupby(list(df.columns), dropna=False).size()
        print(f"Distinct rows that appear more than once: {(counts > 1).sum():,}")
        print(f"Most copies of one identical row: {counts.max():,}")


def numeric_summary(df: pd.DataFrame) -> None:
    rule("5. NUMERIC COLUMNS")
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        print("None.")
        return
    # Quantiles reveal outliers far better than mean/std, which outliers
    # themselves distort. Compare the 99.9th percentile to the max: a big gap
    # means a handful of extreme values.
    print(num.quantile([0, 0.01, 0.25, 0.50, 0.75, 0.99, 0.999, 1.0]).T.round(2))


def categorical_summary(df: pd.DataFrame, top: int = 10) -> None:
    rule("6. CATEGORICAL COLUMNS")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        n = df[col].nunique()
        print(f"\n-- {col} ({n:,} unique) --")
        if n <= top:
            print(df[col].value_counts(dropna=False).to_string())
        else:
            print(df[col].value_counts().head(top).to_string())
            vc = df[col].value_counts()
            rare = (vc < 10).sum()
            print(f"   ... plus {n - top:,} more. "
                  f"{rare:,} categories have fewer than 10 rows.")


def leakage_scan(df: pd.DataFrame) -> None:
    """Look for columns that are arithmetic combinations of other columns.

    If column C == A / B exactly, then C secretly contains A. Training on C to
    predict A hands the model the answer. See Phase 6."""
    rule("7. LEAKAGE SCAN (derived-column detection)")
    num = df.select_dtypes(include=[np.number]).columns
    found = False
    for a in num:
        for b in num:
            if a == b:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = df[a] / df[b].replace(0, np.nan)
            for c in num:
                if c in (a, b):
                    continue
                rel = ((ratio - df[c]).abs() / df[c].replace(0, np.nan))
                match = (rel < 0.001).mean()
                if match > 0.95:
                    print(f"  WARNING: {c} == {a} / {b} for {match*100:.1f}% of rows")
                    found = True
    if not found:
        print("  No exact derived columns detected.")


def main(path: str) -> int:
    csv = Path(path)
    if not csv.exists():
        print(f"ERROR: file not found: {csv}")
        return 1

    # low_memory=False stops pandas from guessing dtypes chunk-by-chunk, which
    # can otherwise assign different types to the same column and emit warnings.
    df = pd.read_csv(csv, low_memory=False)

    print(f"INSPECTION REPORT — {csv.name}")
    overview(df)
    column_profile(df)
    hidden_missing(df)
    duplicates(df)
    numeric_summary(df)
    categorical_summary(df)
    leakage_scan(df)

    rule("DONE — nothing was modified")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
