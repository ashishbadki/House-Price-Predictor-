# Phase 9 — Train/Test Split Decision

**Question:** how do we divide the data so the test score means something?
**Script:** `src/split_data.py`
**Answer:** split by **building**, not by row.

---

## The situation

| | |
|---|---|
| Rows | 51,706 |
| Distinct buildings | 18,815 |
| Rows in a building with more than one listing | **73.9%** |
| Largest single building | 139 listings |
| Rows with no building name | 18.5% |

Three quarters of the data sits in a building that appears more than once.

---

## Finding 1 — The random split was flattering us by 12%

Same model, same features, same seed. Only the split changed.

| Split | MAE |
|---|---:|
| Random 5-fold | 36.65 L |
| **Group 5-fold (by building)** | **41.09 L** |

**Every number in Phases 6, 7 and 8 was about 4.44 lakh optimistic.**

Nothing was buggy. The random split let flats from the same building sit on
both sides, so the model could recall a building's price rather than reason
about it. On a building it has genuinely never seen, it is 12% worse.

The group split is the honest number, and 41.09 L is now our real baseline.

## Finding 2 — The `title` question is settled: it hurts

Phase 8 left this open. Target-encoded building name improved the score on a
random split by 11%, which contradicted our Phase 7 decision to ban it.

Tested under both splits:

| Split | Without `title` | With target-encoded `title` | Effect |
|---|---:|---:|---|
| Random | 36.65 L | **32.54 L** | looks like a 11% gain |
| Group | 41.09 L | **43.89 L** | actually a 7% **loss** |

Under a random split, the encoding for "Runwal Bliss" is built from Runwal
Bliss flats that are sitting in the training set. The model looks the answer up.

Under a group split, a test building never appeared in training, so its
encoding falls back to a global average — which is *less* informative than the
locality and coordinates the model already has. The feature becomes noise, and
the model does worse.

**The entire 11% was memorisation.** `title` stays banned. Phase 7's instinct
was right, and now it is measured rather than assumed.

This is the clearest lesson in the project so far: **a feature that helps under
the wrong split can hurt under the right one.** The split is not a formality
you do before the interesting work. It decides which results are real.

---

## The design

### Groups

```python
known = df.title.notna()
groups = np.where(known, df.title, "UNKNOWN_" + df.index.astype(str))
```

A missing title does not mean "same building" — it means we do not know. Each
such row becomes its own group of one.

The lazy `fillna("UNKNOWN")` would put 9,561 unrelated listings into one giant
group, and `GroupKFold` would then dump all of them into a single fold.

### Two sets, not three

The textbook split is train / validation / test. We use train / test, and get
the validation role from cross-validation inside the training set.

Reasons:

1. CV uses all the training data for both fitting and validating, in turn. A
   fixed validation set spends 20% of the data on a single noisy estimate.
2. A CV score averages 5 folds, so it is much more stable — which matters when
   we are comparing models that differ by one or two lakh.
3. It keeps the rule simple: the test set is touched exactly once, in Phase 18.

A separate validation set earns its place when CV is too slow, or when time
ordering forces a fixed cut. Neither applies here.

### Why not a time-based split?

Time-based splitting is the right answer for price data in general — you train
on the past and test on the future, which is what a deployed model faces.

**We cannot do it.** The dataset has no listing date. This is a genuine
limitation, and it means our test set cannot tell us how the model behaves as
the market moves. It goes in the README.

### Result

| | Rows | Share | Buildings | Median price |
|---|---:|---:|---:|---:|
| TRAIN | 40,826 | 79.0% | 15,052 | 120 L |
| TEST | 10,880 | 21.0% | 3,763 | 123 L |

Buildings on both sides: **0**.

The split is 79/21 rather than exactly 80/20 because whole buildings are moved,
and buildings differ in size. That is expected.

### One honest wrinkle

Checking that the two sides look alike:

| Percentile | Train | Test | Difference |
|---|---:|---:|---:|
| 25th | 65 L | 67 L | 2.6% |
| 50th | 120 L | 123 L | 2.5% |
| 75th | 220 L | 220 L | 0.1% |
| 95th | 650 L | 550 L | **15.4%** |

The middle of the distribution matches closely. The luxury tail does not — the
test set's 95th percentile is 15% lower.

That happens because expensive listings cluster in a small number of large
premium buildings, and whole buildings move together. A few of them landing on
one side shifts the tail.

**We are not going to hunt for a seed that looks tidier.** Trying seeds until
the split flatters you is a quiet form of cheating, and it makes the test set
meaningless. We keep seed 42, note the wrinkle, and remember it in Phase 16
when we look at errors on expensive properties.

---

## The rule for every phase from here

```python
from src.split_data import get_cv_splitter

cv = get_cv_splitter()
for train_idx, val_idx in cv.split(X, y, groups=train.building):
    ...
```

**`groups=` is not optional.** Leave it out and `GroupKFold` cannot group
anything — you are back to a random split and back to a 12% lie, with no error
message to tell you.

The test set is untouched until Phase 18.
