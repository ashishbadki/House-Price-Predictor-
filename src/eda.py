"""
eda.py — Phase 5: Exploratory Data Analysis.

Produces six figures in reports/figures/ plus a printed summary.

Chart design rules applied here (they are not decoration -- each one exists
because the alternative misleads the reader):
  * One hue for magnitude, a second hue only when two things are contrasted.
  * No chart junk: no top/right spines, recessive gridlines, no 3D, no pie.
  * 51,715 points cannot go on a scatter plot -- overlapping dots hide density.
    We use a hexbin (2D histogram) instead, which shows WHERE the mass is.
  * Direct labels on bars, so the reader never counts gridlines.
  * Money is shown in lakh (1 L = 100,000), because raw rupee axes like
    "1.4e8" are unreadable to a human.

Usage:
    python src/eda.py data/processed/mumbai_clean.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --- palette -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

LAKH = 100_000
OUTDIR = Path("reports/figures")


def style_axes(ax) -> None:
    """Strip the frame down to what carries information.

    Top and right spines enclose the data for no reason. Gridlines should be
    visible enough to read a value against, faint enough that the data is what
    your eye lands on first."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)  # grid BEHIND the data, never on top of it


def new_fig(w: float, h: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        style_axes(ax)
    return fig, axes


def save(fig, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path}")


def title(ax, main: str, sub: str = "") -> None:
    """Title above, subtitle below it, neither overlapping the other.

    `pad` is in points and reserves vertical room for BOTH lines; the subtitle
    is then placed just above the axes in axes-fraction coordinates. Get the
    pad wrong and the two strings render on top of each other -- which is
    exactly what happened the first time this was written."""
    pad = 30 if sub else 12
    ax.set_title(main, color=INK, fontsize=12, fontweight="600", loc="left", pad=pad)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=INK_SOFT,
                fontsize=9.5, va="bottom")


# =========================================================================
# FIGURE 1 — price distribution, raw vs log
# =========================================================================
def fig_price_distribution(d: pd.DataFrame) -> None:
    fig, (ax1, ax2) = new_fig(12, 4.4, 2)

    price_l = d.price / LAKH
    # Clip the x-axis at the 99th percentile. Without this the single
    # Rs 135 crore listing stretches the axis and every other bar becomes
    # invisible. We label the clip so the reader knows we did it.
    cut = price_l.quantile(0.99)
    ax1.hist(price_l[price_l <= cut], bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax1.axvline(price_l.median(), color=ORANGE, linewidth=2, linestyle="-")
    ax1.axvline(price_l.mean(), color=ORANGE, linewidth=2, linestyle="--")
    ax1.text(price_l.median(), ax1.get_ylim()[1] * 0.92,
             f"  median {price_l.median():.0f}L", color=ORANGE, fontsize=9)
    ax1.text(price_l.mean(), ax1.get_ylim()[1] * 0.80,
             f"  mean {price_l.mean():.0f}L", color=ORANGE, fontsize=9)
    ax1.set_xlabel("price (lakh)", color=INK_SOFT, fontsize=9.5)
    ax1.set_ylabel("listings", color=INK_SOFT, fontsize=9.5)
    title(ax1, "Raw price is heavily right-skewed",
          f"skew {d.price.skew():.1f}  ·  x-axis clipped at 99th pct for readability")

    ax2.hist(np.log10(d.price), bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax2.set_xlabel("log10(price)", color=INK_SOFT, fontsize=9.5)
    ax2.set_ylabel("listings", color=INK_SOFT, fontsize=9.5)
    title(ax2, "Log price is close to symmetric",
          f"skew {np.log(d.price).skew():.2f}  ·  no clipping needed")

    save(fig, "01_price_distribution.png")


# =========================================================================
# FIGURE 2 — area vs price
# =========================================================================
def fig_area_vs_price(d: pd.DataFrame) -> None:
    fig, (ax1, ax2) = new_fig(12, 4.8, 2)

    # A scatter of 51,715 points is a solid blob -- it shows the outline of the
    # data but hides where the mass actually sits. A hexbin bins the points and
    # colours each cell by count, so density becomes visible.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)

    ax1.hexbin(d.area, d.price / LAKH, gridsize=45, cmap=cmap, mincnt=1,
               bins="log", extent=(0, 3000, 0, 800))
    ax1.set_xlabel("area (sqft)", color=INK_SOFT, fontsize=9.5)
    ax1.set_ylabel("price (lakh)", color=INK_SOFT, fontsize=9.5)
    title(ax1, "Area vs price — linear scale",
          f"Pearson r = {d.price.corr(d.area):.2f}  ·  view limited to 3,000 sqft / 800L")

    hb = ax2.hexbin(np.log10(d.area), np.log10(d.price), gridsize=45,
                    cmap=cmap, mincnt=1, bins="log")
    ax2.set_xlabel("log10(area)", color=INK_SOFT, fontsize=9.5)
    ax2.set_ylabel("log10(price)", color=INK_SOFT, fontsize=9.5)
    r_log = np.log(d.price).corr(np.log(d.area))
    title(ax2, "Area vs price — log-log scale",
          f"Pearson r = {r_log:.2f}  ·  the cloud straightens out")
    cb = fig.colorbar(hb, ax=ax2)
    cb.set_label("listings per cell (log scale)", color=INK_SOFT, fontsize=9)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=8)

    save(fig, "02_area_vs_price.png")


# =========================================================================
# FIGURE 3 — BHK
# =========================================================================
def fig_bhk(d: pd.DataFrame) -> None:
    fig, (ax1, ax2) = new_fig(12, 4.4, 2)
    sub = d[d.bedroom_num.between(1, 5)]

    counts = sub.bedroom_num.value_counts().sort_index()
    bars = ax1.bar(counts.index.astype(str), counts.values, color=BLUE, width=0.62)
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center",
                 va="bottom", color=INK_SOFT, fontsize=9)
    ax1.set_xlabel("BHK", color=INK_SOFT, fontsize=9.5)
    ax1.set_ylabel("listings", color=INK_SOFT, fontsize=9.5)
    ax1.margins(y=0.14)
    title(ax1, "How many listings of each size",
          "1 and 2 BHK are 75% of the market")

    med = sub.groupby("bedroom_num").price.median() / LAKH
    bars = ax2.bar(med.index.astype(str), med.values, color=BLUE, width=0.62)
    for b, v in zip(bars, med.values):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}L", ha="center",
                 va="bottom", color=INK_SOFT, fontsize=9)
    ax2.set_xlabel("BHK", color=INK_SOFT, fontsize=9.5)
    ax2.set_ylabel("median price (lakh)", color=INK_SOFT, fontsize=9.5)
    ax2.margins(y=0.14)
    title(ax2, "Median price rises faster than room count",
          "5 BHK is 15x the price of 1 BHK, not 5x")

    save(fig, "03_bhk.png")


# =========================================================================
# FIGURE 4 — locality
# =========================================================================
def fig_locality(d: pd.DataFrame, min_rows: int = 200) -> None:
    g = (d.assign(ppsf=d.price / d.area)
           .groupby("locality")
           .agg(n=("price", "size"), ppsf=("ppsf", "median")))
    # Localities with a handful of rows produce unstable medians. Requiring a
    # minimum sample size is not cherry-picking -- it is refusing to report a
    # number we cannot support.
    g = g[g.n >= min_rows].sort_values("ppsf")
    band = pd.concat([g.head(10), g.tail(10)])

    fig, (ax,) = new_fig(9, 7)
    colors = [BLUE_RAMP[1]] * 10 + [BLUE_RAMP[4]] * 10
    ax.barh(band.index, band.ppsf, color=colors, height=0.68)
    for y, (v, n) in enumerate(zip(band.ppsf, band.n)):
        ax.text(v + 600, y, f"{v:,.0f}  (n={n})", va="center",
                color=INK_SOFT, fontsize=8.5)
    ax.set_xlabel("median price per sqft (INR)", color=INK_SOFT, fontsize=9.5)
    ax.margins(x=0.22)
    ratio = band.ppsf.iloc[-1] / band.ppsf.iloc[0]
    title(ax, "Location is the single biggest price driver",
          f"10 cheapest and 10 most expensive localities (min {min_rows} listings)  ·  "
          f"{ratio:.0f}x spread")

    save(fig, "04_locality.png")


# =========================================================================
# FIGURE 5 — the furnishing trap
# =========================================================================
def fig_furnishing_confound(d: pd.DataFrame) -> None:
    """The most important chart in this file.

    Median PRICE says furnished flats cost far more. Median PRICE PER SQFT --
    which removes the size difference -- says something quite different. Same
    data, opposite story."""
    fig, (ax1, ax2) = new_fig(12, 4.4, 2)
    order = ["Unfurnished", "Semi-Furnished", "Furnished"]
    g = d.assign(ppsf=d.price / d.area).groupby("furnished").agg(
        price=("price", "median"), ppsf=("ppsf", "median"), area=("area", "median"),
    ).reindex(order)

    bars = ax1.bar(order, g.price / LAKH, color=BLUE, width=0.6)
    for b, v, a in zip(bars, g.price / LAKH, g.area):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}L\n{a:,.0f} sqft",
                 ha="center", va="bottom", color=INK_SOFT, fontsize=9)
    ax1.set_ylabel("median price (lakh)", color=INK_SOFT, fontsize=9.5)
    ax1.margins(y=0.22)
    title(ax1, "Median price says furnishing matters a lot",
          "but furnished flats are also bigger")

    bars = ax2.bar(order, g.ppsf, color=ORANGE, width=0.6)
    for b, v in zip(bars, g.ppsf):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center",
                 va="bottom", color=INK_SOFT, fontsize=9)
    ax2.set_ylabel("median price per sqft (INR)", color=INK_SOFT, fontsize=9.5)
    ax2.margins(y=0.18)
    title(ax2, "Price per sqft tells a different story",
          "Semi-Furnished is CHEAPER per sqft than Unfurnished")

    save(fig, "05_furnishing_confound.png")


# =========================================================================
# FIGURE 6 — age
# =========================================================================
def fig_age(d: pd.DataFrame) -> None:
    nz = d[d.age > 0].assign(ppsf=lambda x: x.price / x.area)
    bands = pd.cut(nz.age, [0, 5, 10, 15, 20, 60],
                   labels=["1-5", "6-10", "11-15", "16-20", "21+"])
    g = nz.groupby(bands, observed=True).agg(n=("price", "size"),
                                             ppsf=("ppsf", "median"))

    fig, (ax,) = new_fig(7.5, 4.4)
    bars = ax.bar(g.index.astype(str), g.ppsf, color=BLUE, width=0.6)
    for b, v, n in zip(bars, g.ppsf, g.n):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\n(n={n:,})",
                ha="center", va="bottom", color=INK_SOFT, fontsize=8.5)
    ax.set_xlabel("property age (years)", color=INK_SOFT, fontsize=9.5)
    ax.set_ylabel("median price per sqft (INR)", color=INK_SOFT, fontsize=9.5)
    ax.margins(y=0.20)
    title(ax, "Older buildings are not cheaper per sqft",
          "excludes the 61% of rows where age = 0 (meaning unknown)")

    save(fig, "06_age.png")


# =========================================================================
def summary(d: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("EDA SUMMARY")
    print("=" * 62)
    print(f"Listings           : {len(d):,}")
    print(f"Median price       : Rs {d.price.median()/LAKH:,.0f} lakh")
    print(f"Mean price         : Rs {d.price.mean()/LAKH:,.0f} lakh")
    print(f"Skew (raw price)   : {d.price.skew():.2f}")
    print(f"Skew (log price)   : {np.log(d.price).skew():.2f}")
    print(f"corr(area, price)  : {d.price.corr(d.area):.3f}")
    print(f"Localities         : {d.locality.nunique()}")

    # eta-squared: the share of variance in log(price) that sits BETWEEN
    # localities rather than within them. A quick, assumption-light way to
    # rank how much a categorical variable matters.
    y = np.log(d.price)
    grand = y.mean()
    grp = d.assign(y=y).groupby("locality").y.agg(["mean", "size"])
    eta2 = ((grp["mean"] - grand) ** 2 * grp["size"]).sum() / ((y - grand) ** 2).sum()
    print(f"Variance of log(price) explained by locality alone : {eta2:.1%}")


def main(path: str) -> int:
    csv = Path(path)
    if not csv.exists():
        print(f"ERROR: file not found: {csv}")
        return 1
    d = pd.read_csv(csv)

    print("Generating figures...")
    fig_price_distribution(d)
    fig_area_vs_price(d)
    fig_bhk(d)
    fig_locality(d)
    fig_furnishing_confound(d)
    fig_age(d)
    summary(d)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
