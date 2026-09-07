# Phase 8 — Categorical Encoding Decision

**Question:** how do we turn `locality` (403 text values) into numbers?
**Script:** `src/encoding_experiment.py`
**Data:** 51,706 rows, 403 localities, 9,254 building names

---

## Results — Ridge (linear model)

| Encoding | MAE |
|---|---:|
| locality dropped entirely | 68.94 L |
| Label / Ordinal | 68.75 L |
| **One-Hot, all 403 columns** | **50.14 L** |
| One-Hot, rare grouped (min 30 rows) | 51.69 L |
| Frequency | 68.95 L |
| Target encoding (correct) | 51.30 L |

## Results — HistGradientBoosting (tree model)

| Encoding | MAE | Time |
|---|---:|---:|
| locality dropped entirely | 36.95 L | 5 s |
| **Label / Ordinal** | **36.39 L** | 5 s |
| One-Hot, all 403 columns | 36.74 L | **62 s** |
| Frequency | 36.47 L | 5 s |
| Target encoding (correct) | 36.48 L | 5 s |

---

## Finding 1 — Label encoding is useless for a linear model

68.75 L with label encoding vs 68.94 L with no locality at all. A gain of
0.19 L, which is nothing.

Label encoding assigns arbitrary numbers: Andheri → 0, Bandra → 1, Chembur → 2.
A linear model reads that as an *ordered scale* and fits a single coefficient
to it, so it is forced to claim something like "each step up the alphabet adds
₹X per sqft". That statement is meaningless, so the coefficient lands near zero
and the column contributes nothing.

**Rule:** label encoding is only correct when the categories have a real order —
Small < Medium < Large, or Unfurnished < Semi-Furnished < Furnished. Locality
names have no order.

## Finding 2 — Frequency encoding is also useless here

68.95 L. Slightly *worse* than dropping the column.

Frequency encoding replaces a locality with how common it is. Thane (10% of
listings) → 0.10. But *how many listings a locality has* tells you nothing about
*how expensive it is*. Thane is common and mid-priced; Prabhadevi is rare and
very expensive; Neral is rare and very cheap. Two rare localities at opposite
ends of the price range get nearly the same number.

It works when frequency genuinely correlates with the target. Here it does not.

## Finding 3 — One-Hot works, and the reason is structural

50.14 L, a 27% improvement over dropping the column.

One-hot makes one column per locality, holding 0 or 1. The model then learns a
separate number for each locality with no ordering assumed. That is exactly
what unordered categories need.

The cost is width: 403 new columns. For Ridge that is fine. For the tree model
it took **62 seconds instead of 5** — twelve times slower — and produced a
slightly *worse* score (36.74 vs 36.39).

## Finding 4 — For a tree model, the encoding hardly matters

Every option lands between 36.39 and 36.95 L. The spread is 0.56 L, about 1.5%.

Trees do not read a number as a quantity. They ask yes/no questions
("is locality_code < 47?") and can split the same column many times, so they can
isolate individual categories even from an arbitrary numbering. Given enough
splits they reconstruct what one-hot would have given them.

**This is the phase's main lesson:** encoding is a big decision for linear
models and a small one for tree models. Which encoding is "best" is not a
property of the data alone — it depends on the model that will consume it.

Also note the tree loses only 0.56 L when locality is removed entirely, because
`latitude` and `longitude` are still present. Same redundancy we measured in
Phase 7.

---

## Finding 5 — The target-encoding trap

Target encoding replaces a category with the average target for that category.
Bandra → the average price per sqft in Bandra. It is powerful, and it is the
easiest way in this whole project to fool yourself.

Tested on `title` (building name, 9,254 values), with a 20% test set held out
and never fitted on:

| How the encoding was computed | CV score | True test score | CV error |
|---|---:|---:|---:|
| No building feature at all | 37.98 L | 35.65 L | +2.32 L |
| **Wrong 1** — averaged over ALL rows, test included | 28.48 L | 26.80 L | +1.69 L |
| **Wrong 2** — dev rows only, but fitted before the CV loop | 28.43 L | 35.29 L | **−6.85 L** |
| **Right** — `TargetEncoder` inside the `Pipeline` | 33.64 L | 31.87 L | +1.77 L |

### Wrong 1 — the encoding was built using the answers

Every row's own price contributed to its building's average. A building with
one listing gets an "average" that is exactly that listing's price. The model
reads the answer off the feature.

The dangerous part: **the test score also looks brilliant (26.80 L)**, because
the test rows' prices went into building the feature too. The test set was
supposed to be the honest check, and it has been contaminated. Nothing in the
output warns you.

In production this cannot work. A new listing arrives with no price, so it
cannot contribute to its own building average. The 26.80 L will never be seen
again.

### Wrong 2 — the subtle one, and the common one

Here the encoding used dev rows only. The test score (35.29 L) is therefore
honest. But the encoding was computed once, before the CV loop, so inside CV
every validation fold was scored with a feature built partly from its own rows.

**CV said 28.43 L. Reality was 35.29 L.** Cross-validation was wrong by 6.85
lakh, in the flattering direction.

This is the mistake that survives code review, because the split looks correct
and the test set is genuinely untouched. Only the CV is broken — and CV is what
you use to choose your model.

### Right — put the encoder inside the Pipeline

```python
ColumnTransformer([
    ("num", "passthrough", NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), SMALL_CATS),
    ("title", TargetEncoder(random_state=42), ["title"]),
])
```

`Pipeline.fit()` refits every step on each fold's training rows only. CV 33.64
vs test 31.87 — the gap is now the same small gap the reference model has.

**Rule:** any transformer that learns from the data belongs inside the Pipeline.
Encoders, imputers, scalers, selectors. If it has a `.fit()`, it goes inside.

### Why cardinality decides the damage

The same mistake on `locality` (403 values, ~128 rows each) was nearly
invisible. On `title` (9,254 values, ~5 rows each) it was worth 6.85 lakh.

With 5 rows per category, one row is 20% of its own average. With 128 rows it
is 0.8%. **Leakage from target encoding scales with the number of categories.**

---

## Decision

**Use One-Hot encoding with rare categories grouped (`min_frequency=30`) for
`locality`, inside a `ColumnTransformer`.**

```python
OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30)
```

Reasons:

1. **It works for every model.** Best for Ridge (51.69 L, within 1.5 L of full
   one-hot), fine for trees. We do not know yet which model wins Phase 11, and
   this choice does not need revisiting when we find out.
2. **Leakage is impossible.** One-hot never looks at the target. Target encoding
   is a little better for Ridge (51.30 vs 51.69) but carries a failure mode that
   is invisible when you get it wrong.
3. **`handle_unknown="infrequent_if_exist"` handles the app.** A user picking a
   locality the model never saw is mapped to the "infrequent" bucket instead of
   crashing.
4. **Fewer columns than full one-hot**, so the tree stays fast.

`property_type` (5 values) and `furnished` (3 values) get plain one-hot — too
small for any of this to matter.

### Left open for Phase 9

Target-encoded `title` improved the test score from 35.65 L to 31.87 L, which is
a real 10.6% gain — and it contradicts our Phase 7 decision to ban `title`.

But that test set came from a **random** split, so the same buildings appear in
both training and test. The model may simply be recalling buildings it has
already priced, which is worth nothing on a building it has never seen.

Phase 9 rebuilds the split so that a building appears on only one side. We
re-run this comparison there. Expectation: most of the 10.6% disappears. Until
that is tested, `title` stays banned.
