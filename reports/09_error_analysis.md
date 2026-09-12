# Phases 16 & 17 — Error Analysis and Interpretability

**Model:** tuned Random Forest (Phase 15). **Data:** `train.csv`, 5-fold GroupKFold.
**Baseline:** MAE 42.02 L, MedAPE 13.20%, bias −3.94 L.
**Script:** `src/error_analysis.py`

---

## The most important thing in this phase

I found what looked like a **6.2% improvement** and it was fake. Here is how.

Phase 4 filtered prices with one global floor of ₹2,000/sqft. That cannot
work, because Neral genuinely trades at ₹3,604/sqft while Thane does not trade
at ₹2,045/sqft. So I built a **locality-relative** check: drop any row whose
rate is below 0.40× or above 2.5× its own locality's median. That flagged 636
rows, including obvious nonsense like a 2,200 sqft 4 BHK in Thane listed at
₹45 lakh.

Re-running cross-validation on the filtered data:

| | MAE |
|---|---:|
| baseline | 42.02 L |
| after removing 636 suspect rows | **39.42 L** |

A 6.2% gain — six times what all of Phase 15's tuning achieved.

**It is not a gain.** I removed those rows from the training data *and* from
the scoring data. The model did not get better; the exam got easier.

The honest test keeps the evaluation set fixed and changes only what the model
learns from:

| Removed from training only, scored on all 40,826 rows | MAE | MedAPE | bias |
|---|---:|---:|---:|
| baseline | 42.02 L | 13.20% | −3.94 L |
| 636 suspect rows dropped from training | **42.01 L** | 13.21% | −5.46 L |
| 344 suspect rows dropped from training | 41.88 L | 13.16% | −4.44 L |
| sqrt(area) sample weights | **41.84 L** | 13.20% | −3.68 L |

**The filter is worth 0.01 lakh.** Nothing.

The clue was visible before the honest test: MAE improved 6.2% but median APE
improved only 1.7%. MAE is dragged by extreme rows; the median is not. When
one moves far more than the other, you have changed *which rows you are
scoring*, not how well you score them.

**The rule:** when you test a data-cleaning idea, the evaluation set must not
change. Remove rows from training, score on everything.

---

## Finding 1 — The error is extremely concentrated

| Worst rows | Share of data | Share of total error |
|---:|---:|---:|
| 10 | 0.02% | 3.0% |
| 100 | 0.2% | **12.7%** |
| 500 | 1.2% | 25.8% |
| 1,000 | 2.4% | 34.3% |

Two and a half percent of the rows carry a third of all the error.

This decides the strategy. Evenly spread error means the model needs to be
better everywhere. Concentrated error means a small, identifiable group is the
problem — and the answer may be a warning message rather than a better model.

## Finding 2 — The model collapses on ultra-luxury property

| Price | Rows | Share | MedAPE | Bias |
|---|---:|---:|---:|---:|
| above ₹10 crore | 848 | 2.08% | 19.3% | −255 L |
| above ₹20 crore | 205 | 0.50% | 27.0% | −772 L |
| above ₹50 crore | **26** | 0.06% | **47.7%** | **−3,315 L** |

On property above ₹50 crore, the model is typically 48% wrong and
under-predicts by an average of **₹33 crore**.

The ten worst predictions in the entire dataset are all the same thing:
4–6 BHK, 5,000–10,000 sqft, in Marine Lines, Colaba, Malabar Hill, Churchgate
or Juhu. Actual ₹56–135 crore, predicted ₹21–75 crore.

**Why.** There are 26 such properties in 40,826 rows. What separates a ₹60
crore Colaba flat from a ₹130 crore one is sea view, floor height, building
prestige, ceiling height, the specific tower. **We have none of those columns.**
Given only area, BHK and locality, these properties are genuinely
indistinguishable from each other.

This is a data limitation, not a modelling failure. No amount of tuning fixes
a column that does not exist.

## Finding 3 — The same story by area and BHK

| Area | Rows | MedAPE | Bias |
|---|---:|---:|---:|
| 500–750 sqft | 13,269 | **11.10%** | −1.18 L |
| 1k–1.5k | 9,149 | 13.60% | −4.64 L |
| >2,500 sqft | 1,056 | **21.59%** | **−57.90 L** |

| BHK | Rows | MedAPE | Bias |
|---|---:|---:|---:|
| 1 | 13,521 | 12.08% | +0.72 L |
| 2 | 16,971 | 12.97% | −1.19 L |
| 4 | 1,834 | 17.04% | **−55.79 L** |
| 5 | 340 | 19.55% | −43.61 L |

The model is good where the data is thick (1–2 BHK, 500–750 sqft, outer
suburbs) and poor where it is thin. Mira Road, with 2,512 listings of very
similar flats, has a 9.13% median error. Bandra, with 641 wildly varied
listings, has 21.96%.

## Finding 4 — Rare property types are broken

| Type | Rows | MedAPE | Bias |
|---|---:|---:|---:|
| Apartment | 40,194 | 13.11% | −2.74 L |
| Villa | 245 | 17.79% | −72.98 L |
| **Independent Floor** | **100** | **35.76%** | **−363.39 L** |
| Independent House | 285 | 20.91% | +12.37 L |

Independent Floor is off by 36% and under-priced by ₹3.6 crore on average,
from 100 examples. The app should either refuse these or attach a loud warning.

## Finding 5 — Some rows have wrong labels, not wrong predictions

192 rows are off by more than 150%. Examples:

| Locality | Area | BHK | Listed price | Rate | Model said |
|---|---:|---:|---:|---:|---:|
| Versova | 349 | 1 | ₹8 L | ₹2,260/sqft | ₹91 L |
| Thane | 2,200 | 4 | ₹45 L | ₹2,045/sqft | ₹4.9 Cr |
| Andheri | 651 | 1 | ₹20 L | ₹3,072/sqft | ₹1.8 Cr |

₹2,045/sqft in Thane is not a price. **The model is right and the data is
wrong.** These are scraping or entry errors that survived Phase 4's global
floor.

They are worth documenting as a data-quality finding — but as the honest test
above showed, removing them does not improve the model.

---

# Phase 17 — What the model actually uses

Permutation importance: shuffle one column and see how much worse the model
gets. Preferred over a tree's built-in `feature_importances_`, which is biased
towards high-cardinality numeric columns whether or not they help.

| Feature | Share of importance |
|---|---:|
| longitude | 21.6% |
| dist_bkc_km | 19.9% |
| dist_cbd_km | 16.4% |
| dist_nariman_km | 12.7% |
| area | 10.2% |
| latitude | 4.9% |
| rooms_total | 3.6% |
| bedroom_num | 3.3% |
| balcony_num | 1.6% |
| furnished | 1.5% |
| locality_grouped | 1.2% |
| bathroom_num | 0.7% |
| age | 0.7% |
| property_type | 0.01% |

**Geography accounts for 75.5%.** Which matches Phase 5, where locality alone
explained 63.7% of the variance in log price.

## Warning 1 — Importance is not causation

`longitude` is the single most important feature. That does **not** mean
moving a flat west makes it more expensive.

Longitude helps the model separate expensive areas from cheap ones, because
Mumbai's geography happens to line up that way. The model has no concept of
*why*. It has never heard of the sea, or of where the offices are.

The correct sentence is: *"longitude is the most useful feature for
predicting price in this model."*
The wrong sentence is: *"longitude affects price."*

The same applies to everything we found earlier. `furnished` looks important;
Phase 5 showed the gap is mostly size and location. `age` looks meaningful;
old buildings are simply concentrated in South Mumbai.

## Warning 2 — Importance is not necessity

`dist_cbd_km` scores 16.4% here. In Phase 7 we measured its actual
contribution at **0.9%** — remove it and latitude/longitude absorb its job
completely.

Both numbers are correct. Correlated features split the credit between
themselves, so importance tells you **what this model leans on**, not **what
you could delete**.

If you want to know whether a feature is needed, you have to remove it and
re-measure. Importance cannot answer that question, and using it to prune
features is a common and costly mistake.

`locality_grouped` at 1.2% is the clearest example. Phase 8 showed locality
alone explains 63.7% of price variance — but by the time the model has
latitude, longitude and three distance features, the locality name adds almost
nothing new.

---

## Decisions

**No modelling change.** `sqrt(area)` sample weights improve MAE by 0.18 L
(0.4%) and improve the big-property bias from −57.9 L to −47.5 L. That is
inside the noise on MAE but the bias improvement points the right way. It is
optional; Phase 18 decides.

**The real output of this phase is honesty, not accuracy.** We now know exactly
where the model cannot be trusted, and that goes into the app and the README:

| Situation | What the app must do |
|---|---|
| Price estimate above ₹10 crore | Warn: typical error 19%, and we under-estimate |
| Area above 2,500 sqft | Warn: typical error 22% |
| 4+ BHK | Warn: typical error 17% |
| Independent Floor / Villa | Warn: very few training examples |
| Everything else | 11–13% typical error |

**For the README:** the model works well on the 1–2 BHK, 500–1,500 sqft
market that is 75% of Mumbai listings, and should not be relied on for luxury
property. Saying that plainly is worth more than a slightly better MAE.

**Carried to Phase 27:** prediction intervals, so the app shows a range instead
of a single confident number.
