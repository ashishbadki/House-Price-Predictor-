# Phase 6 — Target Decision

**Question:** what exactly should the model predict?
**Method:** train the same model on four different targets, with the same
5-fold split and seed 42, then convert every prediction back to rupees and
score them all on that one common scale.
**Script:** `src/target_experiment.py`

---

## Results

All metrics are on the original rupee scale. 51,715 rows, 5-fold CV.

| Target | MAE | RMSE | R² (rupees) | R² (own scale) | Mean bias |
|---|---:|---:|---:|---:|---:|
| A. `price` | 42.6 L | 135.4 L | 0.827 | 0.827 | −0.0 L |
| B. `log(price)` | 37.4 L | 129.0 L | 0.843 | **0.939** | −6.9 L |
| C. `price / area` | **36.1 L** | **105.4 L** | **0.895** | 0.844 | +0.3 L |
| D. `log(price / area)` | 36.2 L | 112.6 L | 0.881 | 0.890 | −6.6 L |

### Where each target wins — mean absolute error by price band

| Band | Median price | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| cheapest 20% | 39 L | 11.7 | 7.2 | 8.6 | **7.1** |
| 20–40% | 75 L | 16.0 | 11.4 | 12.3 | **11.3** |
| 40–60% | 120 L | 24.4 | **19.6** | 21.0 | **19.6** |
| 60–80% | 193 L | 36.2 | **30.9** | 31.8 | **30.9** |
| priciest 20% | 400 L | 125.0 | 117.8 | **106.6** | 112.0 |

### Median error as a percentage of true price

| Band | A | B | C | D |
|---|---:|---:|---:|---:|
| cheapest 20% | 18.5% | 12.5% | 14.4% | **12.2%** |
| 20–40% | 13.8% | 10.7% | 11.4% | **10.6%** |
| 40–60% | 14.9% | 11.9% | 12.2% | **11.7%** |
| 60–80% | 13.9% | **12.3%** | 12.1% | 12.2% |
| priciest 20% | 16.1% | 14.8% | **13.6%** | 14.5% |

---

## Finding 1 — Raw price is clearly the worst option

Target A loses on every band and every metric. This confirms the Phase 5
prediction: with skew of 10.53, squared-error training on raw rupees spends the
model's capacity on the long tail.

Note that A is *unbiased* (−0.0 L). Being unbiased is not the same as being
accurate — it only means the errors cancel out on average.

## Finding 2 — "R² on its own scale" is a trap in both directions

Look at the `R² (own scale)` column against `R² (rupees)`:

- Target **B** scores **0.939** on the log scale but only **0.843** in rupees.
  Reporting 0.939 would overstate the model badly.
- Target **C** scores **0.844** on its own scale but **0.895** in rupees —
  the error runs the *other* way.

So this is not a constant bias you can mentally adjust for. The two numbers are
simply not comparable, because each target has a different amount of variance
available to explain.

**Rule:** convert predictions back to the unit the user cares about, then score.
Never compare an R² across different target definitions.

## Finding 3 — The subtle trap: log models under-predict systematically

Targets B and D both show a mean bias of about **−7 lakh**. They are not
randomly wrong; they are wrong in one direction, on average, across 51,715
properties.

This is **retransformation bias**, and it is a mathematical certainty, not a
bug in the code.

The model is trained to predict the *average of log(price)*. When you take
`exp()` of that, you do not get the average price — you get roughly the
**median** price. For a right-skewed distribution the median is below the mean,
so every prediction lands low. The formal name for the underlying inequality is
Jensen's inequality.

**The fix — Duan's smearing estimator.** Take the residuals on the training
fold, exponentiate them, average them, and multiply your prediction by that
factor:

```python
residuals = y_log_train - model.predict(X_train)
smear = np.exp(residuals).mean()          # 1.0217 here
prediction = np.exp(model.predict(X_test)) * smear
```

Measured effect:

| | MAE | Mean bias |
|---|---:|---:|
| log target, uncorrected | 37.4 L | −6.9 L |
| log target, smearing-fixed | 37.4 L | **−2.7 L** |

Bias drops by 60% and MAE is unchanged. This matters for the app: without it,
every user gets an estimate that is systematically too low.

**Why this is easy to miss:** MAE, RMSE and R² all look fine. None of them
reveal directional bias, because they use absolute or squared errors. You only
see it if you specifically compute `mean(prediction − actual)`. Add that to
every evaluation you ever run.

## Finding 4 — No single target wins everywhere

Target **C** wins overall on RMSE and R², driven entirely by the expensive band
(106.6 L vs 117.8 L for B). Targets **B** and **D** win on the four cheaper
bands, both in rupees and in percentage terms.

This is a genuine trade-off, not a rounding difference:

- **C** is best when large rupee errors are what you want to avoid.
- **D** is best when percentage accuracy matters equally at every price point.

---

## Decision

**Train on `price / area`. Multiply the prediction by area to get the price.**

Reasons, in order:

1. **Best RMSE (105.4 L) and best R² (0.895) on the rupee scale.** RMSE punishes
   large errors, and large errors are what destroy trust in a price estimator.
2. **Effectively unbiased (+0.3 L) with no correction needed.** B and D both need
   a smearing correction just to reach −2.7 L.
3. **Best on expensive properties**, where a percentage error turns into the
   largest rupee error.
4. **It is interpretable.** The model learns a *rate* — rupees per sqft — which
   is exactly how the Mumbai market talks about price. Area is then applied
   mechanically. This is easy to explain to a non-technical person.

### This decision is provisional

Three reasons to revisit it:

- It rests on one model type. Phase 11 may change the ranking.
- The 5-fold split is random, so flats from the same building sit in both
  training and test folds. Every number above is therefore optimistic in
  absolute terms. The *comparison between targets* is still valid, because all
  four share the identical folds.
- **Target C divides by `area`, and we do not know whether `area` means carpet,
  built-up or super built-up.** If area is noisy, that noise goes straight into
  the target. This is the strongest argument against C and it cannot be tested
  with the data we have.

Re-check after Phase 9 (proper split) and Phase 12 (metric choice).

---

## Data leakage review

### Is `price_per_sqft` allowed now?

This confuses people, so state it precisely:

| Use | Verdict | Why |
|---|---|---|
| As a **feature** to predict price | **Leakage** | The user of the app does not know it. They came to find the price. |
| As the **target** we train on | **Fine** | A target is allowed to be built from price — it *is* the thing being predicted. |

The test is always the same: **at prediction time, is this value known?**
`area` is known (the user types it in). `price_per_sqft` is not. So dividing the
target by area is legitimate; feeding price-per-sqft in as an input is not.

### Leakage still ahead of us

| Phase | Risk | Guard |
|---|---|---|
| 8 | Target encoding of `locality` computed on all rows before splitting means test prices help build their own feature | Compute the encoding inside each CV fold, on training rows only |
| 9 | Flats in the same building split across train and test — near-duplicates the model can look up | Group-aware split on building |
| 14 | Imputer and scaler fitted on the full dataset | Put everything inside a `Pipeline` so it fits per fold |
| 15 | Tuning repeatedly against the test set | Tune on CV only; touch the test set once, at the end |

### One thing to disclose honestly

In Phase 4 we removed 52 rows using price-per-sqft bounds derived from looking
at the whole dataset — including rows that will later become test rows. Strictly
this is a small leak.

It affects 0.10% of rows and the thresholds were set to catch impossible values
rather than to tune performance, so the practical effect is negligible. But the
correct answer to "is your pipeline completely leak-free?" is *"almost — here is
the one place it is not, and here is why I judged it acceptable."* That answer
is worth more in an interview than a confident "yes".
