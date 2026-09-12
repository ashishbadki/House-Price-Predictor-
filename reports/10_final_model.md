# Phases 18 & 19 — Final Model and Saved Pipeline

**Script:** `src/train_final.py` — the only script that opens `test.csv`.
**Chosen model:** `HistGradientBoostingRegressor`
**Headline:** median error **13.22%** on 10,880 unseen properties in 3,763
buildings the model had never seen.

---

## Phase 18 — The decision

Three candidates survived Phase 15. Everything we measured:

| | RandomForest | **HistGB** | XGBoost |
|---|---:|---:|---:|
| median APE (primary metric) | **13.18%** | 13.48% | 13.57% |
| bias | **−3.94 L** | −5.34 L | −5.57 L |
| MAE | **42.02 L** | 42.23 L | 42.09 L |
| spread across folds | **2.35** | 2.66 | 2.90 |
| saved file | 54.8 MB | 1.6 MB | **1.2 MB** |
| memory when loaded | 162.8 MB | 3.6 MB | **3.4 MB** |
| one prediction | 47.8 ms | 6.3 ms | **3.9 ms** |
| needs an extra package | no | no | **yes** |

### First: is Random Forest's lead real?

Comparing two averages (13.18 vs 13.48) against fold spreads of 0.18 and 0.36
would call that noise. So we ran a **paired** comparison — the same five folds
for both models, differences taken fold by fold:

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| HistGB − RandomForest | +0.03 | +0.23 | +0.18 | +0.52 | +0.57 |

Random Forest wins **5 out of 5**. Mean gap 0.30 percentage points, spread
0.21.

**The lead is real.** A paired comparison removes the fold-to-fold variation
that both models share, so it can detect a difference far smaller than
comparing two independent averages could. Worth knowing as a technique.

### Second: is it worth what it costs?

0.30 percentage points, in practice: on a ₹1 crore flat the typical error goes
from ₹13.48 lakh to ₹13.18 lakh.

**A ₹30,000 difference on a ₹1 crore property.** No user will ever notice it.

The cost of that ₹30,000: **45× the memory**, 34× the file size, 7.6× the
prediction time. Streamlit Cloud's free tier gives about 1 GB of RAM, and the
app also needs pandas and a dataframe loaded. A 163 MB model is a real
deployment risk in exchange for an improvement nobody can perceive.

### Decision: HistGradientBoostingRegressor

- Accuracy statistically behind Random Forest, practically identical
- Ships inside scikit-learn — one less package to install, pin and break
- 1.6 MB on disk, 3.6 MB in memory
- 6.3 ms per prediction

**If this were a nightly batch job with no memory limit, Random Forest would
win.** It is a web app on a free tier, so it does not. The right model depends
on where it has to run — which is a sentence worth being able to say in an
interview.

---

## Phase 18 result — the test set

Opened once. Building overlap with training: **0**, verified by assertion.

| Metric | Cross-validation | **Test set** |
|---|---:|---:|
| median APE | 13.48% | **13.22%** |
| MAE | 42.23 L | **35.72 L** |
| bias | −5.34 L | **−0.90 L** |
| RMSE | — | 90.15 L |
| R² | — | 0.8598 |
| within 10% | — | 39.7% |
| within 20% | — | 68.0% |

### Why MAE improved so much and median APE barely moved

MAE dropped from 42.23 to 35.72 lakh — a 15% improvement that we did nothing
to earn. Median APE moved 13.48 → 13.22, essentially unchanged.

This is Phase 9 coming back. When we built the split we noticed and documented
one wrinkle: the test set's 95th percentile was 15% lower than the training
set's, because whole luxury buildings move together and a few of them landed
on the training side. We refused to hunt for a tidier seed.

MAE is dragged by extreme properties. A test set with a lighter luxury tail
gets a better MAE for free. **Median APE is not affected by the tail, and it
matches cross-validation almost exactly.**

That agreement is the real result here: **our cross-validation was honest.**
It predicted 13.48% and the truth was 13.22%. Everything built in Phases 9–15
— the group split, the pipeline discipline, keeping the test set sealed — was
so that this sentence could be written.

It also retroactively justifies the Phase 12 choice of median APE as the
primary metric. Had we chosen MAE, we would now be reporting a 15% improvement
that is entirely an accident of the split.

### Overfitting check

| | median APE |
|---|---:|
| on training data it had already seen | 10.16% |
| on the unseen test set | 13.22% |

A gap of 3 percentage points. Some gap is normal and expected — a model always
does better on rows it has memorised. A gap of, say, 10.16% versus 25% would
mean serious overfitting. Three points is healthy.

### By segment — this is what the app's warnings are built from

| Segment | n | median APE | bias |
|---|---:|---:|---:|
| everything | 10,880 | 13.22% | −0.9 L |
| under ₹2 crore | 7,713 | **12.64%** | +4.3 L |
| ₹2–10 crore | 2,992 | 14.52% | −12.1 L |
| **over ₹10 crore** | 175 | **19.56%** | −39.3 L |
| **area over 2,500 sqft** | 249 | **21.37%** | +79.1 L |
| **4+ BHK** | 566 | **18.25%** | +28.1 L |

Phase 16's findings hold on genuinely unseen data. The model is reliable in
the mainstream market and unreliable above ₹10 crore or 2,500 sqft.

One difference worth noting honestly: on training data the large-property bias
was strongly **negative** (−57.9 L); on the test set it is **positive**
(+79.1 L on area over 2,500 sqft). With only 249 such rows, that number is
unstable in either direction. The safe reading is not "the model over-prices
big flats" but **"the model is unreliable on big flats"** — which is what the
warning should say.

---

## Phase 19 — Saving the pipeline

```python
joblib.dump(pipe, "models/house_price_model.joblib", compress=3)
```

**We save the whole Pipeline, not the bare model.** The saved object carries
the imputer's medians and the encoder's category list with it. The Streamlit
app hands it a raw row and never reimplements a single preprocessing step.

Reimplementing preprocessing in the app is how applications quietly drift out
of sync with their model. The app computes a median slightly differently, or
encodes a category in a different order, and the model receives numbers that
mean something other than what it was trained on. It produces confident,
wrong answers, and nothing errors.

Result: **1.6 MB**, loads and predicts in 6.7 ms.

### The verification step

Saving is not done until you have proved the saved file works:

```python
reloaded = joblib.load(MODEL_PATH)
check = to_price(reloaded.predict(test[FEATURES].head(200)), area[:200])
assert np.allclose(check, test_pred[:200])
```

Passed — the reloaded model reproduces the original predictions exactly.

### The model card

`models/model_card.json` records, alongside the metrics: the exact
hyperparameters, the feature list, the training set size, the test results by
segment, and the **library versions**.

Versions matter. A pipeline pickled by one scikit-learn version may warn or
fail to load under another. Recording them turns a baffling error six months
from now into a one-line diagnosis.

It also records the limitations, so they travel with the file rather than
living only in a report nobody opens at runtime:

- Asking prices, not confirmed transaction prices
- The dataset never says whether area is carpet, built-up or super built-up
- No listing dates, so the model cannot account for time
- 26 training examples above ₹50 crore
- Under 300 examples each for Independent Floor and Villa

---

## The honest headline

> A HistGradientBoosting model that estimates Mumbai residential asking prices
> with a **median error of 13.2%**, measured on 10,880 properties in 3,763
> buildings that were held out completely. **68% of estimates land within 20%
> of the listed price**; **40% land within 10%**.
>
> Compared with a locality-median lookup table — what a broker does mentally —
> the model reduces MAE by 22%.
>
> It should not be relied on above ₹10 crore, where the median error rises to
> 19.6% from only 175 test examples.
