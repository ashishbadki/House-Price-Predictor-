"""
conftest.py — shared setup that pytest loads automatically.

Two jobs:

1. Put `src/` on the import path, so tests can `from predict import ...`
   without every test file repeating the same sys.path dance.

2. Define FIXTURES. A fixture is a named piece of setup that pytest builds
   once and hands to any test that asks for it by name. Loading the model
   takes about 100 ms; with `scope="session"` it happens once for the whole
   run instead of once per test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """A small slice of the real raw file.

    Tests read real data rather than invented data wherever the point is
    "does this handle our actual mess". Only 2,000 rows, so the suite stays
    fast.
    """
    path = ROOT / "data/raw/mumbai-house-price-data-raw.csv"
    if not path.exists():
        pytest.skip("raw data not present")
    return pd.read_csv(path, nrows=2000)


@pytest.fixture(scope="session")
def train_sample() -> pd.DataFrame:
    path = ROOT / "data/processed/train.csv"
    if not path.exists():
        pytest.skip("train.csv not present — run src/split_data.py")
    return pd.read_csv(path, nrows=500)


@pytest.fixture(scope="session")
def model():
    """The saved pipeline, loaded once for the entire test session."""
    from predict import load_model
    if not (ROOT / "models/house_price_model.joblib").exists():
        pytest.skip("model not present — run src/train_final.py")
    return load_model()
