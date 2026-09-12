"""
app.py — Phase 21: the Streamlit web application.

Run it with:
    streamlit run app/app.py

DESIGN RULES THIS APP FOLLOWS
-----------------------------
1. It imports `predict_price` and does NO machine learning itself. Every
   number, every warning, every threshold comes from `src/predict.py`. If the
   app recomputed anything, it would eventually disagree with the model.

2. It never shows a bare number. Phase 18 measured a 13.2% median error, so a
   single confident figure would be a lie. Every estimate comes with a range
   and, where it applies, a warning.

3. It shows the locality's own median rate beside the estimate. A number with
   nothing to compare it against is hard to judge.

4. The limitations are on the page, not buried in a README. Anyone who uses
   this should be able to see what it cannot do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# The app lives in app/ and the code lives in src/, so src has to be on the
# path before we can import from it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import (VALID_FURNISHING, VALID_PROPERTY_TYPES, InvalidInput,
                     format_inr, list_localities, load_model, predict_price)

st.set_page_config(page_title="Mumbai House Price Estimator",
                   page_icon="🏠", layout="centered")


# `cache_resource` loads the model once and keeps it in memory across reruns.
# Streamlit re-executes this whole file on every interaction, so without this
# the app would reload a 1.6 MB model on every click.
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_localities():
    return list_localities()


@st.cache_data
def get_model_card():
    path = ROOT / "models" / "model_card.json"
    return json.loads(path.read_text()) if path.exists() else {}


st.title("🏠 Mumbai House Price Estimator")
st.caption("A statistical estimate from 40,826 Mumbai property listings. "
           "Not a valuation.")

try:
    get_model()
    localities = get_localities()
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun `python src/train_final.py "
             f"data/processed/train.csv data/processed/test.csv` first.")
    st.stop()

card = get_model_card()

# ---------------------------------------------------------------------------
# INPUT FORM
#
# st.form batches the inputs: nothing runs until Submit is pressed. Without
# it, Streamlit reruns the whole script on every single keystroke, which
# makes the app feel twitchy and wastes compute.
# ---------------------------------------------------------------------------
with st.form("property"):
    st.subheader("Property details")

    col1, col2 = st.columns(2)
    with col1:
        # A dropdown, not a text box. It makes an invalid locality impossible
        # rather than something we have to validate and reject.
        locality = st.selectbox("Locality", localities,
                                help="Sorted by how many listings the model "
                                     "saw for each area")
        area = st.number_input("Carpet / built-up area (sqft)",
                               min_value=100, max_value=20_000, value=900,
                               step=50)
        bedrooms = st.number_input("Bedrooms (BHK)", min_value=0,
                                   max_value=15, value=2, step=1,
                                   help="Use 0 only for a studio apartment")
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=15,
                                    value=2, step=1)
    with col2:
        balconies = st.number_input("Balconies", min_value=0, max_value=10,
                                    value=0, step=1)
        age = st.number_input("Age of the building (years)", min_value=0,
                              max_value=100, value=0, step=1,
                              help="0 means new, under construction, or "
                                   "not known")
        prop_type = st.selectbox("Property type", VALID_PROPERTY_TYPES)
        furnishing = st.selectbox("Furnishing", VALID_FURNISHING)

    submitted = st.form_submit_button("Estimate price", type="primary",
                                      use_container_width=True)

# ---------------------------------------------------------------------------
# RESULT
# ---------------------------------------------------------------------------
if submitted:
    try:
        result = predict_price(
            locality=locality, area_sqft=float(area), bedrooms=int(bedrooms),
            bathrooms=int(bathrooms), balconies=int(balconies),
            age_years=int(age), property_type=prop_type,
            furnishing=furnishing)
    except InvalidInput as e:
        # A validation failure is the user's problem to fix, so it gets a
        # plain message, not a stack trace.
        st.error(f"{e}")
        st.stop()

    st.divider()
    st.subheader("Estimated price")
    st.markdown(f"## {result['estimate_text']}")
    st.markdown(f"**Likely range:** {result['range_text']}  \n"
                f"<span style='color:#52514e'>The range is ±"
                f"{result['typical_error_pct']}%, which is the typical error "
                f"this model showed on unseen properties like yours.</span>",
                unsafe_allow_html=True)

    # Context. The estimate on its own is hard for a person to judge; against
    # the locality's own median rate it becomes meaningful.
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated rate", f"₹{result['price_per_sqft']:,.0f}/sqft")
    c2.metric(f"{locality} median",
              f"₹{result['locality_median_rate']:,}/sqft",
              delta=f"{(result['price_per_sqft']/result['locality_median_rate']-1)*100:+.0f}%")
    c3.metric("Listings the model saw here",
              f"{result['locality_listings']:,}")

    for w in result["warnings"]:
        st.warning(w)

    st.info(result["disclaimer"])

# ---------------------------------------------------------------------------
# HONESTY SECTION — always visible, not hidden behind the result
# ---------------------------------------------------------------------------
st.divider()
with st.expander("How accurate is this, really?"):
    test = card.get("test_metrics", {})
    if test:
        st.markdown(f"""
Measured on **{card.get('trained_on', {}).get('rows', 0):,}** training rows and
tested on properties in **buildings the model had never seen**:

| | |
|---|---|
| Typical (median) error | **{test.get('MedAPE_pct')}%** |
| Estimates within 10% of the listed price | {test.get('within_10pct')}% |
| Estimates within 20% | {test.get('within_20pct')}% |
| R² | {test.get('R2')} |

The test set was split by **building**, not by row, so the model was never
graded on a flat in a building it had already studied. That matters: with a
plain random split the same model looked 12% better than it really is.
""")
    segs = card.get("test_by_segment", {})
    if segs:
        st.markdown("**Where it is weaker:**\n")
        rows = ["| Segment | Properties tested | Typical error |",
                "|---|---:|---:|"]
        pretty = {"all": "Everything", "under_2cr": "Under ₹2 crore",
                  "2cr_to_10cr": "₹2–10 crore", "over_10cr": "Over ₹10 crore",
                  "area_over_2500": "Over 2,500 sqft",
                  "bhk_4_plus": "4 BHK and larger"}
        for k, v in segs.items():
            rows.append(f"| {pretty.get(k, k)} | {v['n']:,} | "
                        f"{v['MedAPE_pct']}% |")
        st.markdown("\n".join(rows))

with st.expander("What this model cannot do"):
    for item in card.get("limitations", []):
        st.markdown(f"- {item}")
    st.markdown("""
It also has no idea about the things that actually separate two similar
flats: which floor, whether there is a sea view, how good the building's
maintenance is, how the society is managed, or what the parking situation is.
None of those columns exist in the data.
""")

st.caption("Built as a learning project. Not financial advice, and not a "
           "substitute for a professional valuation.")
