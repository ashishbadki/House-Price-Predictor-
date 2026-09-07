# Phase 5 — EDA Findings

**Input:** `data/processed/mumbai_clean.csv` — 51,715 rows × 14 columns
**Figures:** `reports/figures/01..06`

| Metric | Value |
|---|---|
| Median price | ₹120 lakh (₹1.2 Cr) |
| Mean price | ₹202 lakh (₹2.02 Cr) |
| Skew, raw price | **10.53** |
| Skew, log price | **0.35** |
| corr(area, price) | 0.705 |
| Localities | 403 |
| Variance of log(price) explained by locality alone | **63.7%** |

---

## Finding 1 — Price is extremely right-skewed; log fixes it

Skew of 10.53 is not "a bit skewed", it is one long tail. Mean (₹202 L) sits
68% above median (₹120 L), pulled up by a small number of very expensive
listings — the largest is ₹135 crore.

Taking the natural log drops skew to 0.35, which is close to symmetric.

**Why this matters for modelling.** Squared-error losses (used by linear
regression and most tree ensembles by default) penalise large absolute errors.
On raw price, one ₹135 crore listing contributes more to the loss than
thousands of ₹50 lakh flats combined, so the model spends its capacity fitting
the tail and does worse on the properties that make up most of the market.

Training on `log(price)` and exponentiating the prediction back is the standard
fix. Decided in Phase 6; tested rather than assumed.

*Figure 01*

## Finding 2 — Area explains less than you would guess

Pearson r = 0.705 sounds strong, but r² = 0.497 — area alone accounts for
under half the variance in price. Log-log gives r = 0.69.

The linear-scale hexbin shows why: the point cloud is a **funnel**, not a line.
At 500 sqft prices are tightly bunched. At 1,500 sqft they range from ₹50 lakh
to over ₹8 crore. Same size, wildly different price — because the other half of
the answer is *where* the flat is.

On the log-log panel that funnel straightens into a band, which is a second
argument for the log transform.

**Limitation:** the dataset never states whether `area` is carpet, built-up or
super built-up. In Indian real estate these differ by 20–40% for the same flat.
Some of the vertical spread is measurement inconsistency, not real price
variation, and no model can separate the two.

*Figure 02*

## Finding 3 — Price rises faster than room count

| BHK | Listings | Median price |
|---:|---:|---:|
| 1 | 16,918 | ₹59 L |
| 2 | 21,673 | ₹135 L |
| 3 | 10,236 | ₹266 L |
| 4 | 2,284 | ₹561 L |
| 5 | 418 | ₹900 L |

5 BHK is roughly **15×** the price of 1 BHK, not 5×. The relationship is
convex, not linear.

Two reasons, and they are entangled: larger flats have more sqft *and* larger
flats cluster in expensive localities. This is a case for either a non-linear
model or a log-transformed target — a plain linear regression on raw price
cannot represent this curve.

1 and 2 BHK together are 75% of the data. The model will be well-trained on
those and comparatively weak on 4–5 BHK. Expect that to show up in Phase 16.

*Figure 03*

## Finding 4 — Location is the strongest single feature

Median price per sqft, localities with at least 200 listings:

| Most expensive | ₹/sqft | Cheapest | ₹/sqft |
|---|---:|---|---:|
| Prabhadevi | 47,768 | Neral | 3,604 |
| Lower Parel | 46,981 | Badlapur | 3,955 |
| Matunga | 45,004 | Karjat | 4,177 |
| Byculla | 43,696 | Ambernath | 4,225 |
| Bandra | 41,994 | Nala Sopara | 5,778 |

A **13× spread**, on price per sqft — so this is not a size effect.

Locality alone explains **63.7%** of the variance in log(price) (one-way η²),
against 47.6% for area. Location is the single most informative column in the
dataset, and Phase 8's encoding decision is therefore the most consequential
modelling choice in the project.

*Figure 04*

## Finding 5 — The furnishing trap (correlation ≠ causation)

This is the most instructive result in the EDA.

**By median price**, furnishing looks decisive:

| Furnishing | Median price | Median area |
|---|---:|---:|
| Unfurnished | ₹115 L | 750 sqft |
| Semi-Furnished | ₹119 L | 923 sqft |
| Furnished | ₹210 L | 1,000 sqft |

Furnished flats cost 83% more. Tempting conclusion: *furnishing adds ~₹95 lakh
to a property's value.*

**By median price per sqft** — which removes the size difference — the ordering
breaks:

| Furnishing | Median ₹/sqft |
|---|---:|
| Unfurnished | 15,950 |
| Semi-Furnished | **13,228** |
| Furnished | 22,620 |

Semi-Furnished is **cheaper per sqft than Unfurnished**, despite having a higher
total price. The pattern holds when restricted to 2 BHK only (13,704 vs 16,809),
so it is not an artefact of mixing flat sizes.

The gap in the first table is mostly **size and location**, not furnishing.
Furnished flats are bigger and sit in more expensive areas. Furnishing is partly
a *marker* of an expensive property rather than a *cause* of the price.

**Interpretation rule:** if the model later ranks `furnished` as important, the
correct statement is *"furnishing status helps predict price"*, never
*"furnishing raises price"*. Phase 17 revisits this.

*Figure 05*

## Finding 6 — Older buildings are not cheaper

Among rows with a stated age (`age > 0` — 39% of the data):

| Age band | Listings | Median ₹/sqft |
|---|---:|---:|
| 1–5 | 11,392 | 16,000 |
| 6–10 | 5,261 | 11,111 |
| 11–15 | 2,270 | 13,090 |
| 16–20 | 746 | 17,214 |
| 21+ | 388 | 17,500 |

Non-monotonic, and the oldest band is the *most* expensive per sqft. This
contradicts the intuition that property depreciates with age.

The likely explanation is again geographic confounding: old buildings are
concentrated in South and Central Mumbai (Dadar, Matunga, Byculla — all in the
expensive list above), while new construction happens in the outer suburbs
(Badlapur, Karjat, Neral — all in the cheap list). Age is acting as a partial
proxy for location.

This cannot be confirmed from these numbers alone — it is a hypothesis
consistent with the data, not a demonstrated fact.

*Figure 06*

---

## Decisions carried into later phases

| Phase | Decision |
|---|---|
| 6 — Target | Test `log(price)` against raw `price`. Skew 10.53 → 0.35 is a strong prior in its favour. |
| 7 — Features | Area is a funnel, not a line — the model needs interaction between area and locality. |
| 8 — Encoding | Locality carries 63.7% of the signal and has 403 levels. This is the key encoding decision. |
| 11 — Models | The BHK curve is convex; plain linear regression on raw price will underfit. |
| 16 — Errors | Expect the worst errors on 4–5 BHK and on the luxury tail — thin training data there. |
| 17 — Interpretation | Never state feature importance as causation. Furnishing and age are both confounded with location. |
| 25 — README | Unknown area unit (carpet/built-up/super) is a real, unfixable limitation. |
