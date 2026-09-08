# Phases 12 & 13 — Evaluation and Cross-Validation

**Data:** `train.csv`, 40,826 rows. `test.csv` still sealed.
**Validation:** 5-fold GroupKFold by building.
**Script:** `src/evaluate.py`

---

## Phase 12 — The metrics

| Metric | What it means | Weakness |
|---|---|---|
| **MAE** | Average size of the mistake, in rupees | Treats ₹10 L wrong on a ₹50 L flat the same as on a ₹50 Cr flat |
| **RMSE** | Errors squared before averaging, so big misses dominate | Hard to explain; one bad prediction moves it a lot |
| **R²** | Share of the variation explained; 1.0 perfect, 0.0 no better than the mean | Hides rupees entirely; not comparable across datasets |
| **MAPE** | Error as a % of the true price | Explodes on very cheap properties |
| **Median APE** | The middle percentage error | Ignores the tail completely |
| **Bias** | `mean(prediction − actual)`, **with the sign** | Says nothing about error size |
| **Hit rate** | Share of predictions within 10% / 20% | Throws away how wrong the misses are |

### Why "accuracy" does not apply

Accuracy is *correct answers ÷ total answers*. It needs a clean right and wrong.

A flat sold for ₹1,20,00,000 and we predicted ₹1,19,80,000. Correct? By exact
match, no — and by exact match every regression model scores 0% forever.

You can invent a rule ("correct if within 10%"), which is exactly what
`within_10pct` is. But notice what it discards: a prediction 11% off and one
300% off both count as simply "wrong". Report the hit rate because people
understand it; never choose a model on it.

---

## Finding 1 — The metric decides the winner

Same three models, same predictions, same folds.

| Model | MAE | RMSE | R² | MAPE | MedAPE | Bias | ≤10% | ≤20% | fold SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 42.34 | 145.44 | 0.822 | **19.14%** | **13.14%** | **−4.06** | **40.1%** | **67.2%** | **2.35** |
| HistGB | 42.70 | **143.63** | **0.827** | 19.98% | 13.99% | −4.31 | 37.4% | 65.3% | 2.66 |
| XGBoost | **42.18** | 146.35 | 0.820 | 19.35% | 13.60% | −5.76 | 38.4% | 66.7% | 2.90 |

| Metric | Winner |
|---|---|
| MAE | XGBoost |
| RMSE, R² | HistGB |
| MAPE, MedAPE, bias, ≤10%, ≤20%, fold SD | **RandomForest** |

Six of nine metrics pick Random Forest. MAE picks XGBoost. RMSE and R² pick
HistGB.

**There is no way to read one number and be done.** You have to decide which
kind of wrongness matters, then measure that.

### Why they disagree

RMSE squares the errors, so the handful of very expensive properties dominate
it. HistGB is better on those, so it wins RMSE and R².

MAPE and median APE work in percentages, so a ₹5 L miss on a ₹40 L flat counts
far more than the same miss on a ₹4 Cr flat. Random Forest is better across the
bulk of the market, so it wins those.

Notice RMSE (≈145 L) is **3.4× MAE** (≈42 L). That ratio is itself a finding: if
all errors were similar, the two would be close. A large gap means a small
number of properties are predicted very badly. Phase 16 goes looking for them.

## Finding 2 — Every model under-predicts, and only bias shows it

| Model | Bias |
|---|---:|
| RandomForest | −4.06 L |
| HistGB | −4.31 L |
| XGBoost | −5.76 L |

All negative. Every model quotes low, on average, by ₹4–6 lakh.

**MAE, RMSE, R², MAPE and the hit rates cannot see this.** They all discard the
sign, so "always 5 lakh low" looks identical to "randomly 5 lakh out".

Broken down by price band (Random Forest):

| Band | Median price | MAE | Median APE | **Bias** |
|---|---:|---:|---:|---:|
| cheapest 20% | 38.6 L | 8.35 L | 13.4% | **+4.94 L** |
| 20–40% | 75.0 L | 12.32 L | 11.4% | **+4.38 L** |
| 40–60% | 120.0 L | 21.43 L | 12.3% | **+6.01 L** |
| 60–80% | 192.3 L | 33.70 L | 12.7% | **+2.11 L** |
| priciest 20% | 400.0 L | 136.01 L | 16.4% | **−37.77 L** |

Cheap properties are over-priced by ₹2–6 lakh. Expensive ones are
**under-priced by ₹37.77 lakh**.

This pattern is called **regression to the mean**. The model hedges towards the
middle: it rarely predicts an extreme value, because being extreme is risky
when the training data thins out. The overall −4 L is these two opposite errors
partly cancelling — which is exactly why an overall number can be misleading.

Consequences: the app must warn users on high-value properties, and Phase 16
starts here.

---

## Phase 13 — Cross-validation

### How K-Fold works

Cut the training data into 5 parts. Train on 4, score on the 1 left out. Repeat
5 times so every row is scored exactly once, by a model that never saw it.
Average the 5 scores.

Every row is used for training (4 times) and for scoring (once). Nothing is
wasted, and the answer is an average rather than a single measurement.

### Finding 3 — One split cannot tell our models apart

Same model, same data. **Only the split's random seed changes.**

| seed | MAE | | seed | MAE |
|---|---:|---|---|---:|
| 0 | 42.03 L | | 5 | 41.56 L |
| 1 | 41.95 L | | 6 | 47.27 L |
| 2 | 46.44 L | | 7 | 45.86 L |
| 3 | 47.93 L | | 8 | 42.25 L |
| 4 | 41.48 L | | 9 | 46.80 L |

**Lowest 41.48 L. Highest 47.93 L. Spread 6.45 lakh — from the same model.**

Our three candidate models differ by **0.52 lakh**.

So a single 80/20 split swings **twelve times more** than the gap we are trying
to measure. If you tested two models on one split each and one won by 3 lakh,
you would have learned nothing about the models and quite a lot about the seed.

By comparison, 5-fold CV:

```
per-fold: 46.02  41.20  38.67  45.12  42.47
CV mean: 42.70 L
spread across folds: 2.66 L
standard error of the mean: 1.19 L
```

The standard error is `sd / √5` — how much the *average* would move if we ran
it again. 1.19 L, against 6.45 L for a single split.

### The honest conclusion

Even 5-fold CV has a standard error of 1.19 L, and our models differ by 0.52 L.

**Cross-validation still cannot separate these three models with confidence.**

That is not a failure of the method. It is the correct answer: on this data,
with these features, Random Forest, HistGB and XGBoost are equally good, and
any ranking of them on accuracy alone is noise.

This is why Phase 18 decides on more than accuracy — stability, speed,
simplicity and dependency count all become tie-breakers, precisely because
accuracy has run out of resolution.

### Why the pipeline must be inside the loop

```python
for train_idx, val_idx in GroupKFold(5).split(X, y, groups):
    pipe = Pipeline([("pre", make_preprocessor()), ("model", model_fn())])
    pipe.fit(X.iloc[train_idx], y[train_idx])      # refits everything
```

The imputer's median and the encoder's category list are **learned from data**.
Building them once outside the loop lets every fold's validation rows influence
its own features — the Phase 8 mistake, measured there at 6.85 lakh.

---

## Decision — the metrics we report

**Primary: Median APE.** The app's users care about percentage accuracy, and
the median is not distorted by the luxury tail. Current best: **13.14%**.

**Secondary: MAE in lakh.** Concrete, in rupees, easy to explain.

**Always reported alongside:**
- **Bias**, because it is the only metric that shows direction
- **Error by price band**, because the overall average hides a ₹38 lakh
  under-prediction on expensive property
- **Hit rate (≤10%, ≤20%)** for the README, because it is the one number a
  non-technical reader immediately understands

**Not used as the deciding metric:** R². It hides rupees, it is not comparable
across datasets, and here it separates the models by 0.007.
