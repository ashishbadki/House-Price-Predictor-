# Phases 22 & 23 — Tests and Project Structure

**92 tests, 1.4 seconds, all passing.**

---

## Phase 22 — What tests are actually for

Not to prove the code works today. You already saw it work when you ran it.

**A test exists to tell you the day it stops working.**

Every test here locks in a decision we reasoned about in an earlier phase. If
someone — including you, in three months — changes a threshold or reorders a
step, the relevant test fails immediately and names the decision that broke.

### Coverage

| File | Tests | What it protects |
|---|---:|---|
| `test_data_cleaning.py` | 40 | Every Phase 4 decision, plus "no function mutates its input" |
| `test_predict.py` | 35 | Input validation, prediction sanity, warnings, the saved model |
| `test_features_and_split.py` | 17 | Distance maths, the feature contract, and the group split |

---

## Proving the tests are worth having

A suite that passes first time proves nothing. It might be testing that
`1 == 1`. So we broke the code on purpose, four times, and checked that the
right test caught each one.

| Bug introduced | Test that caught it |
|---|---|
| Tightened the coordinate box to `18.85–19.55` — the Phase 4 mistake that would delete real Palghar listings | `test_only_out_of_region_coordinates_are_nulled[19.7-72.77-True]` |
| Replaced the per-row unknown-building ids with `fillna("UNKNOWN")` | `test_rows_without_a_title_each_get_their_own_group` |
| Removed `df.copy()` from one cleaning function | `test_no_function_mutates_its_input[normalise_locality]` |
| Allowed `bathrooms = 0` in the validator | `test_bad_values_are_rejected[bathrooms-0-at least one]` |

**Each bug produced exactly one failure, in the right place, with a name that
says what broke.** That is what a good test suite feels like. If a single bug
had produced twenty failures, the tests would be tangled together; if it had
produced none, they would be decorative.

**Do this to your own test suite.** Break something on purpose and check the
suite notices. It is the only way to know your tests are real.

## The tests that matter most

### `test_no_building_appears_on_both_sides`

The single highest-value test in the project. If the group split ever breaks,
**nothing crashes** — every score in every report just becomes about 12%
too good. There is no error message for that. This test is the error message.

### `test_no_function_mutates_its_input`

One parametrised test covering all seven cleaning functions. If a function
ever forgets its `df.copy()`, running a notebook cell twice gives a different
answer the second time. That costs an afternoon to find by hand and one second
here.

### `test_studio_with_zero_bedrooms_is_allowed`

A **negative** test — it checks that something is NOT rejected. Phase 4's
most instructive moment was noticing that a naive "drop rows where bedrooms
== 0" rule would delete valid Studio Apartments. Tests should guard the
things you decided not to do, as well as the things you did.

### `test_model_survives_an_unseen_category`

`handle_unknown="infrequent_if_exist"` is one word in one line of
`preprocessing.py`, and it is the only thing standing between a user picking
an unfamiliar locality and the app throwing an exception at them.

### `test_expensive_locality_costs_more_than_a_cheap_one`

Bandra must come out well above Badlapur. This does not check an exact value —
it checks that the location features are actually reaching the model. A
plumbing bug that silently drops a column would pass every unit test and fail
this one.

## How these tests are written

**Real data where the point is real mess.** `raw_sample` reads 2,000 rows of
the actual raw file. Invented data would never contain the blank titles and
sentinel prices we are defending against.

**Invented data where the point is one rule.** `make_row(latitude=28.5)`
constructs exactly the case under test and nothing else.

**Sanity, not exact values.** No test asserts "this flat costs ₹2.56 crore".
Retrain the model and that changes, and the test becomes noise. They assert
`5 lakh < estimate < 100 crore`, and that a bigger flat costs more.

**Fixtures for expensive setup.** `conftest.py` loads the model once with
`scope="session"` rather than once per test. It is why 92 tests take 1.4
seconds.

**Parametrise instead of repeating.** One `test_bad_values_are_rejected`
covers 17 invalid inputs. Each shows as its own named result, so a failure
still says exactly which case broke.

---

## Phase 23 — Final structure

```
house-price-predictor/
├── data/
│   ├── raw/                    original CSV, never edited (gitignored)
│   └── processed/              regenerable outputs (gitignored)
├── notebooks/
│   └── 02_eda.ipynb            exploration, step by step
├── src/
│   ├── inspect_data.py         Phase 3   describe a CSV honestly
│   ├── data_cleaning.py        Phase 4   7 reusable cleaning functions
│   ├── eda.py                  Phase 5   six figures
│   ├── target_experiment.py    Phase 6   which target to predict
│   ├── feature_engineering.py  Phase 7   the 17 features
│   ├── encoding_experiment.py  Phase 8   how to encode locality
│   ├── split_data.py           Phase 9   group-aware split
│   ├── train_models.py         Ph 10-11  baselines and 8 models
│   ├── evaluate.py             Ph 12-13  metrics and CV
│   ├── preprocessing.py        Phase 14  THE shared pipeline
│   ├── tune.py                 Phase 15  hyperparameter search
│   ├── error_analysis.py       Ph 16-17  where it fails, what it uses
│   ├── train_final.py          Ph 18-19  final model + save
│   ├── predict.py              Phase 20  the public API
│   └── templates/
│       └── script_template.py            starting point for new scripts
├── app/
│   └── app.py                  Phase 21  Streamlit application
├── models/
│   ├── house_price_model.joblib          the pipeline (1.6 MB)
│   ├── model_card.json                   metrics, versions, limitations
│   └── locality_reference.json           199 localities + coordinates
├── tests/                      Phase 22  92 tests
├── reports/                              11 findings documents
│   └── figures/                          six EDA charts
├── run_pipeline.py                       run every step in order
├── pytest.ini
├── requirements.txt                      runtime only
├── requirements-dev.txt                  everything else
├── .gitignore
└── README.md
```

### Three changes from the structure we started with

**Added `reports/`.** Findings documents belong neither in `notebooks/`
(exploration) nor in `src/` (code). Eleven of them now, and they are what a
reviewer can read without running anything.

**Added `run_pipeline.py`.** In Phase 7 a real bug cost us 3,413 rows — not
from bad code, but from editing a threshold and never re-running the script
that used it. The CSV on disk was older than the code that produced it, and
nothing said so. A stale pipeline has no error message.

```bash
python run_pipeline.py --check     # what is out of date?
python run_pipeline.py             # rebuild everything in order
```

The staleness rule is the one `make` has used since 1976: an output is stale
if it is missing or older than any of its inputs.

**Split the requirements in two.**

| File | Contents | Why |
|---|---|---|
| `requirements.txt` | pandas, numpy, scikit-learn, joblib, streamlit | What the deployed app needs |
| `requirements-dev.txt` | + matplotlib, seaborn, jupyterlab, xgboost, pytest | What development needs |

Streamlit Cloud installs `requirements.txt` on every cold start, so every
package in it is startup time. The app does not need jupyterlab. It does not
need xgboost either — Phase 18 chose HistGradientBoosting precisely so that
it would not.

### `notebooks/` versus `src/` — the distinction that matters

**Notebooks are for thinking.** Out of order, messy, exploratory. Ours has one
notebook, and it is the EDA walkthrough.

**`src/` is for code that ships.** Importable, testable, deterministic.

Notebooks cannot be imported, cannot be unit-tested, run out of order, and
hide state. Every idea that survived exploration moved into `src/` as a
function with a docstring and a test. That is the whole pattern.

---

## Running everything

```bash
python run_pipeline.py --check     # is anything stale?
python run_pipeline.py             # rebuild the chain
pytest                             # 92 tests
streamlit run app/app.py           # the app
```

Four commands, in that order. If all four are clean, the project is
consistent from raw CSV to running application.
