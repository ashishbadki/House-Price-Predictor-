# Phase 3 — Data Inspection Findings

**File:** `data/raw/mumbai-house-price-data-raw.csv`
**Source:** Kaggle — "Mumbai House Price Data (70k Entries)"
**Shape:** 71,938 rows × 15 columns
**Reported nulls:** 0 (this is misleading — see Finding 2)

---

## Data dictionary

Built from the actual file, not from any dataset description.

| Column | Meaning | Data type | Example | Purpose in the project |
|---|---|---|---|---|
| `title` | Building or project name | text | `Runwal Bliss` | Identifier only. **Not a model feature** — but critical for detecting same-building leakage. Blank on 16.1% of rows. |
| `price` | Listing price in rupees | int64 | `6600283` | **Target variable.** Asking price, not transaction price. |
| `area` | Floor area in sqft | int64 | `757` | Core feature. Unit type (carpet/built-up/super) is **not stated** anywhere. |
| `price_per_sqft` | Price ÷ area | float64 | `8719.0` | **MUST BE DROPPED — direct leakage.** See Finding 1. |
| `locality` | Suburb / neighbourhood | text | `Kalyan` | Strongest categorical feature. 408 values, long-tailed. |
| `city` | City / zone label | text | `Mumbai` | Near-constant (98.5% "Mumbai"). Low value. |
| `property_type` | Property category | text | `Apartment` | 98.5% "Apartment". Low variance but cheap to keep. |
| `bedroom_num` | Number of bedrooms (BHK) | int64 | `2` | Core feature. Range 0–15. |
| `bathroom_num` | Number of bathrooms | int64 | `2` | Feature. Range 0–15. |
| `balcony_num` | Number of balconies | int64 | `0` | 90.8% zeros — likely "not stated". Treat with suspicion. |
| `furnished` | Furnishing status | text | `Unfurnished` | Clean 3-category feature. No missing values. |
| `age` | Property age in years | int64 | `0` | 60.8% zeros. Ambiguous — see Finding 5. |
| `total_floors` | Floors in the building | int64 | `1` | **Broken — 99.9% are the value 1.** See Finding 4. |
| `latitude` | Latitude | float64 | `19.2444` | Usable after cleaning; 143 rows point outside Mumbai. |
| `longitude` | Longitude | float64 | `73.1233` | Same as above. |

**Columns the dataset does NOT have,** despite being on our Phase 2 wish list: floor number of the unit, parking, possession status, amenities, listing date, carpet-vs-builtup distinction, listing ID.

The absence of a **listing date** is significant. It means we cannot do a time-based train/test split, and we cannot know how stale these prices are.

---

## Findings

### Finding 1 — `price_per_sqft` is pure target leakage (CRITICAL)

`price_per_sqft == price / area` for **100.0% of rows.** Verified to within 0.1%.

The column is not independent information; it is the target divided by another feature. A model given both `price_per_sqft` and `area` can recover `price` exactly by multiplication and will score near-perfect R² — while being completely useless in production, because a user of the app does not know the price per sqft of a flat whose price they are trying to estimate.

**Action:** drop `price_per_sqft` from the feature set entirely. Keep it only for EDA.

### Finding 2 — Missing data is hidden, not absent

`df.isna().sum()` returns 0 for every column. That is not because the data is complete. Missingness is encoded as:

- empty strings in `title` (11,595 rows, 16.1%)
- the number `0` in `balcony_num` (65,337 rows, 90.8%)
- the number `0` in `age` (43,771 rows, 60.8%)

A cleaning step that only looks for `NaN` will find nothing and declare the data clean.

### Finding 3 — 28% of rows are exact duplicates (CRITICAL)

- 20,171 exact duplicate rows
- 51,767 unique rows remain after deduplication
- 12,214 distinct rows appear more than once
- One identical row (`Reputed Builder Wild Wood Park 2`, Andheri, ₹20,00,000) appears **59 times**

This is the single most dangerous issue in the dataset. If duplicates are split across train and test, the model memorises rows it will be graded on and test performance becomes a lie. Deduplication must happen **before** the train/test split, not after.

### Finding 4 — `total_floors` is broken

| Value | Rows |
|---:|---:|
| 1 | 71,875 |
| everything else | 63 |

99.91% of the column is the value 1. It carries essentially no information and the values that aren't 1 (including two rows at 70) are unexplained.

**Action:** drop the column. Note it in the README as a data-quality limitation.

### Finding 5 — `age` is ambiguous

60.8% of rows have `age = 0`. Two readings are possible: genuinely new/under-construction property, or "not stated" encoded as zero. The dataset ships no documentation, so **we cannot resolve this from the data alone.**

**Action:** treat `age = 0` as its own category rather than as the number zero, so the model can learn whatever it means without us asserting a meaning we can't verify.

### Finding 6 — Impossible and sentinel values

- **Two rows priced at exactly ₹2,147,483,647** — this is `2^31 − 1`, the maximum value of a signed 32-bit integer. It is an overflow or sentinel artifact, not a price. Both are `Kalpataru Prive`, Malabar Hill.
- **Minimum price ₹32,000** for a 3BHK in Deonar, giving ₹25/sqft. Not a real Mumbai price.
- 11 rows under ₹5 lakh; 68 rows under ₹10 lakh.
- `area` ranges 123 – 24,109 sqft. 76 rows under 200 sqft.
- `bedroom_num`, `bathroom_num` and `balcony_num` all top out at exactly **15**, which looks like a scraper cap rather than three coincidences.

Note: the two rows with `bedroom_num = 0` are both `Studio Apartment` — those are **legitimate**, not errors. Worth catching, because a naive rule "drop rows where bedrooms = 0" would delete valid data.

### Finding 7 — 143 rows have coordinates outside Mumbai

Examples: `(28.50, 77.16)` is Delhi. `(22.70, 88.39)` is Kolkata. `(23.06, 72.55)` is Ahmedabad. Latitude reaches 72.87 and longitude 91.80.

The `locality` field on these rows still says Mumbai suburbs, so the coordinates are wrong rather than the rows belonging to another city.

### Finding 8 — Locality has a long tail

408 unique localities. 271 of them (66%) have fewer than 10 rows, together covering only 670 rows. One-hot encoding all 408 would create 408 columns, most of them almost entirely zeros.

This drives the Phase 8 encoding decision.

### Finding 9 — Same-building leakage risk

`title` holds 9,277 distinct building names across 71,938 rows. `Paradise Sai World Empire Phase 1` alone appears 384 times.

Even after removing exact duplicates, many rows are **different flats in the same building** — near-identical in location, age and price per sqft. A random train/test split puts some of a building's flats in train and others in test, and the model can effectively look up the answer. Test scores will overstate real-world performance on a building the model has never seen.

Addressed in Phase 9.

---

## Implications for later phases

| Phase | What this changes |
|---|---|
| 4 — Cleaning | Deduplicate first. Drop `price_per_sqft` and `total_floors`. Fix or null out bad coordinates. Remove sentinel prices. Decide zero-handling for `age` and `balcony_num`. |
| 6 — Target | Target is `price`. `price_per_sqft` cannot be a feature. |
| 8 — Encoding | 408 localities with a long tail rules out naive one-hot. |
| 9 — Split | Group-aware split on building, not a plain random split. |
| 21 — App | The app cannot ask users for `price_per_sqft`, so the model must never have used it. |
| 25 — README | Asking prices; unknown area unit; no listing date; `total_floors` unusable. |
