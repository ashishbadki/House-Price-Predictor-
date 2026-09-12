"""
error_analysis.py — Phases 16 & 17: where the model fails, and what it uses.

Phase 15 bought 0.8%. This phase is where the real understanding comes from,
even though — as you will see below — it also bought very little accuracy.
That is a finding, not a failure.

THE TRAP THIS SCRIPT DEMONSTRATES
---------------------------------
Removing awkward rows from your data makes your score improve without making
your model better. You have made the exam easier, not the student smarter.

    remove 636 suspect rows from BOTH training and scoring -> MAE 42.02 -> 39.42
    remove the same 636 rows from TRAINING ONLY            -> MAE 42.02 -> 42.01

The first number is a 6.2% "improvement" and it is entirely fake. The honest
test keeps the evaluation set fixed and changes only what the model learns
from. `honest_test()` below is the correct pattern.

Usage:
    python src/error_analysis.py data/processed/train.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import FEATURES, build_pipeline, get_groups

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LAKH = 100_000
CRORE = 10_000_000
N_FOLDS = 5

pd.set_option("display.width", 200)


def tuned_model():
    """The Phase 15 winner, with its tuned settings."""
    return RandomForestRegressor(n_estimators=187, max_depth=25,
                                 max_features=0.4, min_samples_leaf=3,
                                 n_jobs=-1, random_state=RANDOM_STATE)


def cv_predict(df, drop_mask=None, weights=None):
    """Out-of-fold predictions in rupees.

    `drop_mask` removes rows from the TRAINING half of each fold only. The
    rows still get scored. That is what makes a comparison honest: the exam
    stays the same while the studying changes.
    """
    X = df[FEATURES]
    y = (df.price / df.area).values
    area = df.area.values.astype(float)
    groups = get_groups(df)

    pred = np.zeros(len(df))
    for train_idx, val_idx in GroupKFold(N_FOLDS).split(X, y, groups):
        if drop_mask is not None:
            train_idx = train_idx[~drop_mask[train_idx]]
        pipe = build_pipeline(tuned_model())
        if weights is None:
            pipe.fit(X.iloc[train_idx], y[train_idx])
        else:
            pipe.fit(X.iloc[train_idx], y[train_idx],
                     model__sample_weight=weights[train_idx])
        pred[val_idx] = pipe.predict(X.iloc[val_idx]) * area[val_idx]
    return pred


def summarise(price, pred, area, label):
    ape = np.abs(pred - price) / price * 100
    big = area > 2500
    print(f"  {label:<46} MAE {mean_absolute_error(price, pred)/LAKH:6.2f}L  "
          f"MedAPE {np.median(ape):5.2f}%  bias {(pred-price).mean()/LAKH:+7.2f}L  "
          f">2.5k sqft bias {(pred-price)[big].mean()/LAKH:+8.1f}L")


# ===========================================================================
# PHASE 16 — WHERE THE ERRORS ARE
# ===========================================================================

def error_concentration(price, pred) -> None:
    """Is the error spread evenly, or carried by a few rows?

    This decides your whole strategy. Evenly spread means the model needs to
    be better everywhere. Concentrated means a small, identifiable group is
    the problem — and the answer may be a warning message rather than a
    better model.
    """
    err = np.abs(pred - price)
    order = np.argsort(-err)
    total = err.sum()
    print("\nHow concentrated is the damage?")
    for k in (10, 50, 100, 500, 1000):
        print(f"  worst {k:>4} rows ({k/len(price)*100:4.1f}% of data) carry "
              f"{err[order[:k]].sum()/total*100:5.1f}% of all error")


def error_by_segment(df, pred) -> None:
    d = df.assign(pred=pred, err=pred - df.price,
                  ape=np.abs(pred - df.price) / df.price * 100)

    print("\nBy price band")
    for cut in (10, 20, 50):
        m = d.price > cut * CRORE
        if m.sum():
            print(f"  above Rs {cut:>2} crore: {m.sum():>4} rows "
                  f"({m.mean()*100:5.2f}%)  MedAPE {d.ape[m].median():5.1f}%  "
                  f"bias {d.err[m].mean()/LAKH:+9.1f} L")

    print("\nBy area")
    bands = pd.cut(d.area, [0, 500, 750, 1000, 1500, 2500, 30000],
                   labels=["<500", "500-750", "750-1k", "1k-1.5k",
                           "1.5k-2.5k", ">2.5k"])
    print(d.groupby(bands, observed=True).agg(
        n=("price", "size"),
        med_price_L=("price", lambda s: s.median() / LAKH),
        MedAPE=("ape", "median"),
        bias_L=("err", lambda s: s.mean() / LAKH)).round(2).to_string())

    print("\nBy BHK")
    b = d[d.bedroom_num.between(1, 5)]
    print(b.groupby("bedroom_num").agg(
        n=("price", "size"),
        med_price_L=("price", lambda s: s.median() / LAKH),
        MedAPE=("ape", "median"),
        bias_L=("err", lambda s: s.mean() / LAKH)).round(2).to_string())

    print("\nBy property type")
    print(d.groupby("property_type").agg(
        n=("price", "size"), MedAPE=("ape", "median"),
        bias_L=("err", lambda s: s.mean() / LAKH)).round(2).to_string())

    print("\nWorst and best localities (min 300 rows), by median % error")
    loc = d.groupby("locality_grouped").agg(
        n=("price", "size"),
        med_price_L=("price", lambda s: s.median() / LAKH),
        MedAPE=("ape", "median"),
        bias_L=("err", lambda s: s.mean() / LAKH))
    loc = loc[loc.n >= 300].sort_values("MedAPE", ascending=False).round(2)
    print(loc.head(5).to_string())
    print("  ...")
    print(loc.tail(5).to_string())


def suspicious_labels(df, pred) -> pd.DataFrame:
    """Rows where the model is probably right and the DATA is probably wrong.

    Phase 4 used one global floor of Rs 2,000/sqft. That cannot work, because
    Neral genuinely trades at Rs 3,604/sqft while Thane does not trade at
    Rs 2,045/sqft. A plausibility check has to be relative to the locality.
    """
    d = df.assign(pred=pred, ape=np.abs(pred - df.price) / df.price * 100,
                  rate=df.price / df.area)
    med = d.groupby("locality_grouped").rate.transform("median")
    d["rate_vs_locality"] = d.rate / med

    flagged = d[(d.ape > 150)]
    print(f"\n{len(flagged)} rows are off by more than 150%.")
    print("The largest, with their rate against their locality's median:\n")
    print(flagged.nlargest(6, "ape")[
        ["locality_grouped", "area", "bedroom_num", "price", "pred",
         "rate", "rate_vs_locality", "ape"]
    ].assign(price=lambda x: (x.price / LAKH).round(0),
             pred=lambda x: (x.pred / LAKH).round(0),
             rate=lambda x: x.rate.round(0),
             rate_vs_locality=lambda x: x.rate_vs_locality.round(2),
             ape=lambda x: x.ape.round(0)).to_string(index=False))
    return d


def honest_test(df) -> None:
    """The centrepiece of this phase.

    Two ways to test a data-cleaning idea. One of them lies.
    """
    price = df.price.values.astype(float)
    area = df.area.values.astype(float)
    rate = price / area
    med = df.assign(rate=rate).groupby("locality_grouped").rate.transform("median")
    outlier = (~pd.Series(rate).between(med * 0.40, med * 2.5)).values

    print("\n" + "=" * 78)
    print("THE WRONG WAY — drop suspect rows from training AND scoring")
    print("=" * 78)
    sub = df[~outlier].reset_index(drop=True)
    p = cv_predict(sub)
    summarise(sub.price.values.astype(float), p, sub.area.values.astype(float),
              f"filtered set ({len(sub):,} rows)")
    print("  This looks like a big win. It is not a win at all — we deleted")
    print("  the hardest rows from the exam.")

    print("\n" + "=" * 78)
    print("THE RIGHT WAY — drop from training only, score on all rows")
    print("=" * 78)
    base = cv_predict(df)
    summarise(price, base, area, "baseline")

    p2 = cv_predict(df, drop_mask=outlier)
    summarise(price, p2, area, f"same {outlier.sum()} rows dropped from training")

    w = np.sqrt(area / area.mean())
    p3 = cv_predict(df, weights=w)
    summarise(price, p3, area, "sqrt(area) sample weights")

    return base


# ===========================================================================
# PHASE 17 — WHAT THE MODEL USES
# ===========================================================================

def importance(df) -> None:
    """Permutation importance: shuffle one column, see how much worse it gets.

    Preferred over a tree's built-in `feature_importances_`, which is biased
    towards high-cardinality numeric columns regardless of whether they help.
    Permutation importance measures the effect on actual predictions.

    TWO WARNINGS, BOTH ESSENTIAL:

    1. IMPORTANCE IS NOT CAUSATION. "longitude is 21% important" does NOT mean
       moving a flat west raises its price. It means longitude helps the model
       tell expensive areas from cheap ones. The model has no idea why.

    2. IMPORTANCE IS NOT NECESSITY. Correlated features split the credit
       between them. Phase 7 measured `dist_cbd_km` as worth only 0.9% —
       remove it and latitude/longitude absorb its job. Yet it scores 16% here.
       Both facts are true. Importance answers "what does this model lean on",
       not "what could I delete".
    """
    X = df[FEATURES]
    y = (df.price / df.area).values
    groups = get_groups(df)
    train_idx, val_idx = next(iter(GroupKFold(N_FOLDS).split(X, y, groups)))

    pipe = build_pipeline(tuned_model())
    pipe.fit(X.iloc[train_idx], y[train_idx])

    result = permutation_importance(
        pipe, X.iloc[val_idx], y[val_idx], n_repeats=5,
        random_state=RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=1)

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": result.importances_mean,
        "sd": result.importances_std,
    }).sort_values("importance", ascending=False)
    imp["share_pct"] = imp.importance / imp.importance.clip(lower=0).sum() * 100
    print(imp.round(2).to_string(index=False))

    geo = ["latitude", "longitude", "dist_nariman_km", "dist_bkc_km",
           "dist_cbd_km", "locality_grouped"]
    share = imp[imp.feature.isin(geo)].share_pct.sum()
    print(f"\n  Geography accounts for {share:.1f}% of total importance.")


def main(path: str) -> int:
    src = Path(path)
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 1
    df = pd.read_csv(src)
    price = df.price.values.astype(float)

    print(f"{len(df):,} training rows | tuned Random Forest | "
          f"{N_FOLDS}-fold GroupKFold\n")
    print("=" * 78)
    print("PHASE 16 — ERROR ANALYSIS")
    print("=" * 78)

    base = honest_test(df)
    error_concentration(price, base)
    error_by_segment(df, base)
    suspicious_labels(df, base)

    print("\n" + "=" * 78)
    print("PHASE 17 — WHAT THE MODEL ACTUALLY USES")
    print("=" * 78)
    importance(df)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
