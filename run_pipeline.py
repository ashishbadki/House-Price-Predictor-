"""
run_pipeline.py — run every step in the right order.

WHY THIS EXISTS
---------------
In Phase 7 a real bug cost us 3,413 rows. The cause was not bad code: it was
running `data_cleaning.py`, then editing a threshold, then never re-running
it. The CSV on disk was older than the script that produced it, and nothing
said so.

That failure mode has a name -- a stale pipeline -- and it is the most common
bug in data work, because there is no error message. The script ran. A file
appeared. It was just built from the wrong rules.

This script removes the possibility. One command runs the chain in order:

    raw CSV
      -> data_cleaning.py       -> mumbai_clean.csv
      -> feature_engineering.py -> mumbai_features.csv
      -> split_data.py          -> train.csv, test.csv
      -> train_final.py         -> model + model card + locality lookup

It also checks the timestamps before it starts and tells you which outputs
are already older than their inputs.

Usage:
    python run_pipeline.py              # run everything
    python run_pipeline.py --check      # only report what is stale
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/mumbai-house-price-data-raw.csv"
CLEAN = ROOT / "data/processed/mumbai_clean.csv"
FEATURES = ROOT / "data/processed/mumbai_features.csv"
TRAIN = ROOT / "data/processed/train.csv"
TEST = ROOT / "data/processed/test.csv"
MODEL = ROOT / "models/house_price_model.joblib"

# (name, script, inputs, outputs)
STEPS = [
    ("clean", ["src/data_cleaning.py", str(RAW), str(CLEAN)],
     [RAW, ROOT / "src/data_cleaning.py"], [CLEAN]),
    ("features", ["src/feature_engineering.py", str(CLEAN), str(FEATURES)],
     [CLEAN, ROOT / "src/feature_engineering.py"], [FEATURES]),
    ("split", ["src/split_data.py", str(FEATURES), str(ROOT / "data/processed")],
     [FEATURES, ROOT / "src/split_data.py"], [TRAIN, TEST]),
    ("final model", ["src/train_final.py", str(TRAIN), str(TEST)],
     [TRAIN, TEST, ROOT / "src/train_final.py", ROOT / "src/preprocessing.py"],
     [MODEL]),
]


def is_stale(inputs, outputs) -> bool:
    """An output is stale if it is missing, or older than any of its inputs.

    This is the same rule `make` has used since 1976. It is simple and it is
    exactly what we needed in Phase 7.
    """
    if any(not o.exists() for o in outputs):
        return True
    newest_input = max(i.stat().st_mtime for i in inputs if i.exists())
    oldest_output = min(o.stat().st_mtime for o in outputs)
    return newest_input > oldest_output


def report() -> bool:
    print(f"{'step':<16}{'status':<12}outputs")
    print("-" * 70)
    any_stale = False
    for name, _, inputs, outputs in STEPS:
        stale = is_stale(inputs, outputs)
        any_stale |= stale
        status = "STALE" if stale else "up to date"
        names = ", ".join(o.name for o in outputs)
        print(f"{name:<16}{status:<12}{names}")
    return any_stale


def main(check_only: bool) -> int:
    if not RAW.exists():
        print(f"ERROR: raw data not found at {RAW}")
        print("Download it from Kaggle and place it there. See the README.")
        return 1

    any_stale = report()
    if check_only:
        print("\n" + ("Some steps are stale. Run `python run_pipeline.py`."
                      if any_stale else "Everything is up to date."))
        return 0

    print()
    for name, args, _, _ in STEPS:
        print("=" * 70)
        print(f"RUNNING: {name}")
        print("=" * 70)
        # Once one step re-runs, every later step must re-run too, because
        # its input just changed. So we do not skip steps that merely look
        # up to date -- that check is for reporting, not for skipping.
        result = subprocess.run([sys.executable, *args], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nFAILED at step '{name}'. Stopping.")
            return result.returncode
        print()

    print("=" * 70)
    print("Pipeline complete. Run `pytest tests/` to confirm nothing broke,")
    print("then `streamlit run app/app.py`.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
