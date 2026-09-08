# Phases 10 & 11 — Baselines and Models

**Data:** `train.csv` only — 40,826 rows, 15,052 buildings. `test.csv` not opened.
**Validation:** 5-fold GroupKFold, grouped by building.
**Target:** price per sqft; all metrics converted back to rupees.
**Script:** `src/train_models.py`

---

## Phase 10 — Baselines

| Model | MAE | RMSE | R² | fold SD |
|---|---:|---:|---:|---:|
| B1 global median rate | 110.51 L | 288.38 L | 0.302 | 6.78 |
| B2 locality median rate | 55.86 L | 178.94 L | 0.731 | 4.17 |
| **B3 linear regression** | **53.94 L** | 174.37 L | 0.745 | 3.75 |

### What each one is

**B1** predicts the same price per sqft for every property in Mumbai. It is the
dumbest thing that is not zero, and it exists to set a floor.

**B2 is the baseline that matters.** It looks up each locality's median price
per sqft and multiplies by area — exactly what an experienced broker does in
their head: *"Andheri is running about 18,000 a foot, the flat is 900 feet, so
roughly 1.6 crore."* It needs no machine learning, no Python, and could be a
spreadsheet.

**B3** is ordinary linear regression on all 17 features.

### Finding — linear regression barely beats the lookup table

53.94 L vs 55.86 L. A **3.4%** improvement for using every feature we built,
over a rule that only knows the locality.

That is worth sitting with. If we had stopped at linear regression and reported
"my ML model achieves R² of 0.74", it would have sounded respectable while
being almost entirely reproducible by a broker with a calculator.

**This is what a baseline is for.** A metric on its own tells you nothing.
A metric next to a baseline tells you whether the work was worth doing.

---

## Phase 11 — Machine learning models

| Model | MAE | RMSE | R² | fold SD | Time |
|---|---:|---:|---:|---:|---:|
| M1 Ridge | 53.96 L | 174.31 L | 0.745 | 3.75 | 1 s |
| M2 Lasso | 55.36 L | 174.70 L | 0.744 | 3.68 | 37 s |
| M3 Decision Tree | 48.35 L | 162.42 L | 0.779 | 2.34 | 2 s |
| M4 Random Forest | 42.34 L | 145.44 L | 0.822 | **2.35** | 84 s |
| M5 Extra Trees | 42.56 L | 150.42 L | 0.810 | 3.02 | 66 s |
| M6 Gradient Boosting | 46.60 L | 150.98 L | 0.809 | 2.83 | 97 s |
| M7 HistGradientBoosting | 42.70 L | **143.63 L** | **0.827** | 2.66 | 17 s |
| **M8 XGBoost** | **42.18 L** | 146.35 L | 0.820 | 2.90 | 9 s |

**Best model 42.18 L vs best baseline 53.94 L — a 21.8% improvement.**

---

## Finding 1 — Linear models cannot represent this problem

Ridge (53.96 L) is identical to plain linear regression (53.94 L), and Lasso is
worse (55.36 L) while taking 37 seconds.

Ridge and Lasso are linear regression with a penalty that stops coefficients
growing too large. That penalty helps when a model is memorising noise. Here
the linear model is not memorising — it is **underfitting**. It cannot draw the
shape the data has, and no amount of penalty fixes that.

Phase 5 predicted this: the BHK curve bends upward (5 BHK is 15× the price of
1 BHK, not 5×), and the area–price relationship is a funnel rather than a line.
A straight line cannot follow either.

## Finding 2 — Trees are the right family, and one tree is not enough

| | MAE |
|---|---:|
| One decision tree | 48.35 L |
| 150 trees (Random Forest) | 42.34 L |

A single tree splits the data into boxes and predicts the average inside each
box. That handles curves and interactions naturally — which is why even one
tree beats every linear model here.

But one tree is unstable. Change a few rows and its splits change, so it fits
the noise in whatever data it saw.

An **ensemble** grows many trees on different random slices of the data and
averages them. Individual mistakes are random and cancel out; the real pattern
is in every tree and survives. That is worth **6 lakh** here.

## Finding 3 — The top four are effectively tied

Random Forest 42.34, Extra Trees 42.56, HistGradientBoosting 42.70, XGBoost
42.18. The spread is 0.52 lakh — about 1.2%.

The fold-to-fold standard deviation is 2.3–2.9 lakh, which is **five times**
that spread. In other words, the gap between these four models is smaller than
the noise between folds. Declaring XGBoost the winner on MAE alone would be
reading meaning into randomness.

They also disagree about who wins depending on the metric:

- **Lowest MAE:** XGBoost (42.18)
- **Lowest RMSE and highest R²:** HistGradientBoosting (143.63, 0.827)
- **Most stable across folds:** Random Forest (SD 2.35)

MAE and RMSE disagree because RMSE punishes large errors much harder. XGBoost is
better on typical properties; HistGradientBoosting is better on the expensive
ones where errors are large. Phase 12 defines which of those we care about.

## Finding 4 — Speed differs by 10×, and it is not the model you would guess

| Model | Time | MAE |
|---|---:|---:|
| XGBoost | 9 s | 42.18 L |
| HistGradientBoosting | 17 s | 42.70 L |
| Extra Trees | 66 s | 42.56 L |
| Random Forest | 84 s | 42.34 L |
| Gradient Boosting | 97 s | 46.60 L |

Gradient Boosting is the slowest **and** among the least accurate. It is the
original boosting implementation and grows trees one at a time on exact splits.
HistGradientBoosting and XGBoost bucket the numbers first, which loses almost
nothing and is roughly ten times faster.

**Practical consequence:** in Phase 15 we tune hyperparameters, which means
fitting a model hundreds of times. At 9 seconds per fit that is fine. At 97
seconds it is an overnight job.

## Finding 5 — On XGBoost as a dependency

XGBoost wins on MAE by 0.52 lakh over HistGradientBoosting, which is inside the
fold-to-fold noise. HistGradientBoosting wins on RMSE and R², and ships inside
scikit-learn — no extra package, one less thing to install on the deployment
server in Phase 26.

Keeping XGBoost has to be justified by more than a 1.2% MAE difference that the
fold spread cannot distinguish. Revisit in Phase 15: if tuning opens a real gap,
it earns its place. If not, HistGradientBoosting is the simpler choice.

---

## Decisions

**Carry forward to Phase 12–15:** Random Forest, HistGradientBoosting, XGBoost.
Drop Ridge, Lasso, plain Gradient Boosting, and the single Decision Tree — they
lose on accuracy, speed, or both.

**Do not pick a winner yet.** Phase 12 defines the metric properly, Phase 13
sets up cross-validation, Phase 15 tunes, and Phase 18 decides — on accuracy,
stability, speed, and simplicity together.

**The honest headline for the README:** 21.8% better than a locality-median
lookup table, measured on a building-aware split.
