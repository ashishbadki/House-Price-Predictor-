"""
script_template.py — copy this file whenever you start a new script.

WHAT THIS TEMPLATE IS FOR
-------------------------
Every script in this project has the same eight parts, in the same order.
Once you know the shape, writing a new one is filling in blanks instead of
starting from an empty file.

    1. Docstring     - what it does, how to run it
    2. Imports       - standard library, then third-party, then your own
    3. Constants     - every number and choice, named, at the top
    4. Small funcs   - one job each, no side effects
    5. Orchestrator  - calls the small funcs in order
    6. main()        - reads input, calls orchestrator, writes output
    7. Safety check  - assert the things that must be true
    8. Entry guard   - if __name__ == "__main__"

HOW TO USE IT
-------------
    copy  src/templates/script_template.py  ->  src/my_new_script.py
    change the docstring, the constants, and the functions
    delete what you do not need

Usage:
    python src/script_template.py <input.csv> <output.csv>
"""

# ---------------------------------------------------------------------------
# 2. IMPORTS — three groups, in this order, blank line between them.
#    This is a Python convention (PEP 8). Follow it and your code looks
#    like everyone else's, which is the point.
# ---------------------------------------------------------------------------
from __future__ import annotations   # lets you write type hints on old Pythons

import sys                            # standard library
from pathlib import Path

import numpy as np                    # third-party
import pandas as pd

# from src.data_cleaning import clean_dataset   # your own code goes here


# ---------------------------------------------------------------------------
# 3. CONSTANTS
#
#    RULE: if a number appears in your logic, it gets a name up here.
#
#    Bad:   df = df[df.price > 500000]
#    Good:  MIN_PRICE = 500_000
#           df = df[df.price > MIN_PRICE]
#
#    Why: six months from now, "500000" means nothing. MIN_PRICE means
#    something. And when you need to change it, there is exactly one place
#    to look instead of six places to hunt for.
#
#    Write big numbers with underscores: 500_000 is easier to read than
#    500000. Python ignores the underscores.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # fixed seed, so results repeat exactly
MIN_PRICE = 500_000        # example threshold
REQUIRED_COLUMNS = ["price", "area"]


# ---------------------------------------------------------------------------
# 4. SMALL FUNCTIONS
#
#    Three rules for every function you write:
#
#    RULE A - one job. If you need the word "and" to describe what it does,
#             split it into two functions.
#
#    RULE B - never change the input. Start with df.copy(), return the copy.
#             If a function edits the DataFrame you passed in, then running
#             a notebook cell twice gives a different answer the second time,
#             and you will lose an afternoon to that.
#
#    RULE C - say what it does and WHY in the docstring. The code says what
#             happens. The docstring says why you chose it.
# ---------------------------------------------------------------------------

def do_one_thing(df: pd.DataFrame) -> pd.DataFrame:
    """One-line summary of what this does.

    Longer explanation of WHY this step exists, and what would go wrong
    without it. Mention any judgement call you made.
    """
    out = df.copy()          # RULE B, always the first line
    # ... your work here ...
    return out


def do_another_thing(df: pd.DataFrame) -> pd.DataFrame:
    """Second step."""
    out = df.copy()
    return out


# ---------------------------------------------------------------------------
# 5. ORCHESTRATOR
#
#    Calls the small functions in order and reports what each one did.
#
#    The steps list is the valuable part: someone can read those seven words
#    and understand the whole script without reading any code. And reordering
#    a step means moving one line.
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run all steps in order, printing the row/column count after each."""
    steps = [
        ("first step", do_one_thing),
        ("second step", do_another_thing),
    ]

    out = df
    if verbose:
        print(f"{'step':<26}{'rows':>10}{'cols':>7}")
        print("-" * 45)
        print(f"{'INPUT':<26}{len(out):>10,}{out.shape[1]:>7}")

    for name, fn in steps:
        out = fn(out)
        if verbose:
            print(f"{name:<26}{len(out):>10,}{out.shape[1]:>7}")

    return out


# ---------------------------------------------------------------------------
# 7. SAFETY CHECKS
#
#    An `assert` stops the program with a message when something that must be
#    true is not.
#
#    Use one whenever a mistake would be SILENT. A wrong number that still
#    looks like a number is far more dangerous than a crash, because you will
#    not notice it for two weeks.
#
#    Do NOT use asserts for user input errors (a missing file) -- print a
#    clear message and return 1 instead. Asserts are for "this should be
#    impossible", not "the user typed the wrong thing".
# ---------------------------------------------------------------------------

def check_input(df: pd.DataFrame) -> None:
    """Fail fast if the input is not what we expect."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"input is missing required columns: {missing}"
    assert len(df) > 0, "input file has no rows"


def check_output(df: pd.DataFrame) -> None:
    """Fail fast if the output is not what we promised."""
    assert "price_per_sqft" not in df.columns, "LEAKAGE: price_per_sqft present"
    assert not df.duplicated().any(), "duplicates survived"


# ---------------------------------------------------------------------------
# 6. main()
#
#    Everything that touches the outside world lives here: reading files,
#    writing files, printing. The functions above stay pure, which is what
#    makes them testable in Phase 22.
#
#    main() returns an integer: 0 means success, 1 means failure. This is the
#    standard every command-line tool follows, and it is how a Makefile or a
#    CI system knows whether your step worked.
# ---------------------------------------------------------------------------

def main(in_path: str, out_path: str) -> int:
    src = Path(in_path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1                                   # not an assert -- see above

    df = pd.read_csv(src, low_memory=False)
    check_input(df)

    result = run_pipeline(df)
    check_output(result)

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)  # create folders if needed
    result.to_csv(dest, index=False)                # index=False, always

    print(f"\nWritten: {dest}  ({dest.stat().st_size / 1024**2:.1f} MB)")
    return 0


# ---------------------------------------------------------------------------
# 8. ENTRY GUARD
#
#    Code under this line runs ONLY when you type `python script.py`.
#    It does NOT run when another file does `import script`.
#
#    Without this guard, importing your script would execute the whole thing,
#    which is why every real Python script has it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)        # wrong number of arguments -> show the docstring
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
