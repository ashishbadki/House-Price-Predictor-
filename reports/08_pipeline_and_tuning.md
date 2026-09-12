# Phases 14 & 15 — Pipeline and Tuning

**Scripts:** `src/preprocessing.py`, `src/tune.py`
**Data:** `train.csv` only. `test.csv` still sealed.

---

## Phase 14 — One pipeline, one definition

### The problem being fixed

By the end of Phase 13, four scripts each contained their own copy of the
column lists and the `ColumnTransformer`:

```
target_experiment.py    encoding_experiment.py
train_models.py         evaluate.py
```

Four copies of the same thing is four chances to drift apart. Change
`min_frequency` in one and forget the others, and two scripts silently
disagree about what the model sees — with no error to say so.

`src/preprocessing.py` is now the only definition. Everything imports it,
including the Streamlit app in Phase 21, which must preprocess a user's form
input in exactly the way the training data was preprocessed. If those two ever
differ, the app produces confident nonsense.

### What the pipeline does

```
raw row -> impute missing -> (scale numbers) -> one-hot encode -> model -> rate
                                                                            |
                                                              x area -> rupees
```

| Step | Choice | Why |
|---|---|---|
| Impute | `SimpleImputer(strategy="median")` | Fills the 151 rows with no coordinates. Median, not mean, because this dataset has a ₹135 crore listing that would drag a mean around. |
| Scale | `StandardScaler`, **linear models only** | Linear models compare coefficients across features; without scaling, `area` (hundreds) and `has_balcony` (0/1) are treated as if one unit means the same thing. Trees only ask "above or below X?", so scaling changes nothing for them. |
| Encode | `OneHotEncoder(min_frequency=30)` | Phase 8 decision. |
| Unknown categories | `handle_unknown="infrequent_if_exist"` | A user picking a locality the model never saw is routed to the "infrequent" bucket instead of crashing the app. |
| Everything else | `remainder="drop"` | This is the line that guarantees `price`, `title` and `building` cannot reach the model, even though they are still in the DataFrame. |

### Why a Pipeline is not just tidiness

`SimpleImputer` and `OneHotEncoder` **learn** things from data — a median, a
list of categories. Fit them once on everything and then split, and every
validation row has helped build its own features.

Phase 8 measured that mistake at **6.85 lakh** of self-deception.

Inside a `Pipeline`, `cross_val_score` and `RandomizedSearchCV` refit every
step on each fold's training rows only. The mistake becomes impossible to make
by accident, which is a better guarantee than remembering not to make it.

---

## Phase 15 — Hyperparameter tuning

### Parameters vs hyperparameters

| | Who sets it | Examples |
|---|---|---|
| **Parameters** | the model learns them from data | tree split points, regression coefficients |
| **Hyperparameters** | you set them, before training | number of trees, depth, learning rate |

The model cannot learn its hyperparameters, because they control *how* it
learns. Until now we used numbers I typed in Phase 11 — reasonable guesses.

### Grid search vs randomized search

`GridSearchCV` tries every combination. Four settings with five values each is
5⁴ = 625 combinations × 5 folds = **3,125 fits**. Exhaustive and usually
unaffordable.

`RandomizedSearchCV` samples N random combinations from ranges. You pick N, so
you pick the budget. It usually finds an equally good answer, because most
hyperparameters barely matter — a random search spreads its budget across the
ones that do, instead of exhaustively exploring the ones that do not.

Budget used here: **12 candidates × 3 folds = 36 fits per model.** The winner
is then re-scored on the full 5 folds so it is comparable to Phases 11–13.

### The rule that matters most

The search scores candidates with GroupKFold **on the training data only**.
`test.csv` is not opened by `tune.py`.

Tuning against the test set — trying 200 settings and keeping whichever scores
best on test — stops the test score from being an estimate of anything. You
have fitted the test set by hand, 200 attempts at a time. It will look
excellent and mean nothing.

---

## Results

| Model | MAE before | MAE after | Gain | Gain % | MedAPE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest | 42.34 L | **42.02 L** | 0.32 L | 0.8% | **13.20%** | **−3.94 L** |
| HistGB | 42.70 L | 42.23 L | 0.47 L | 1.1% | 13.51% | −5.34 L |
| XGBoost | 42.18 L | 42.09 L | 0.09 L | 0.2% | 13.57% | −5.57 L |

### Best settings found

**RandomForest** — `max_depth=25`, `max_features=0.4`, `min_samples_leaf=3`,
`n_estimators=187`

**HistGB** — `learning_rate=0.048`, `max_iter=443`, `max_leaf_nodes=78`,
`min_samples_leaf=12`, `l2_regularization=0.30`

**XGBoost** — `learning_rate=0.036`, `n_estimators=589`, `max_depth=7`,
`subsample=0.83`, `colsample_bytree=0.96`, `min_child_weight=2`,
`reg_lambda=0.30`

---

## Finding 1 — Tuning bought almost nothing

Between **0.2% and 1.1%**. In rupees: 0.09 to 0.47 lakh.

Phase 13 measured the standard error of a 5-fold CV score at **1.19 lakh**.
Every one of these gains is well inside that noise. We cannot honestly claim
any of them is real.

This is worth saying plainly, because tuning is the step beginners expect to
be transformative. It is usually the smallest lever in the project. Compare
what the earlier phases were worth:

| Decision | Effect |
|---|---:|
| Choosing the right target (Phase 6) | ~15% |
| Moving from linear to tree models (Phase 11) | ~22% |
| Fixing the split (Phase 9) | 12% — of honesty, not accuracy |
| **Hyperparameter tuning (Phase 15)** | **0.2 – 1.1%** |

Data quality, target definition and model family are where the gains are.
Tuning is polish applied at the end.

## Finding 2 — The tuned settings are readable

`RandomForest` chose `max_features=0.4` — each split considers 40% of the
features rather than all of them. That makes the individual trees disagree
more, and disagreement is what makes an average work. It is the setting that
moved most.

`min_samples_leaf=3` rather than 1 means no leaf is allowed to hold a single
property. A leaf of one row has memorised that row.

Both tuned boosters chose a **low learning rate with many rounds** (0.048 ×
443, and 0.036 × 589) over a high rate with few. Many small corrections beat
a few large ones — the same reason careful small steps beat lunging.

## Finding 3 — Random Forest is still ahead on what we said we care about

Phase 12 chose **median APE** as the primary metric and **bias** as a
mandatory companion. On both, after tuning:

| Model | MedAPE | Bias |
|---|---:|---:|
| **RandomForest** | **13.20%** | **−3.94 L** |
| HistGB | 13.51% | −5.34 L |
| XGBoost | 13.57% | −5.57 L |

XGBoost is best on MAE (42.09 vs 42.02 — actually Random Forest now leads
there too). The margins are still inside the noise, but Random Forest is
consistently on the right side of it, and it does not need an extra package.

**Phase 18 makes the final call.** This is evidence, not a verdict.

---

## Still true, and still to be done

The bias is barely improved: −3.94 L against −4.06 L before tuning. Tuning
does not fix a systematic under-prediction, because none of these
hyperparameters is about direction — they are about flexibility.

The ₹37.77 lakh under-prediction on the priciest 20% found in Phase 12 is
still there. **Phase 16 goes after it.**
