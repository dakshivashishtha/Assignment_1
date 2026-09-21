"""
Sales Prediction - Streamlit Interface
=======================================
Interactive prediction interface for the Sales Prediction System.

Run with:
    streamlit run app.py

Requirements:
    pip install streamlit pandas numpy scikit-learn joblib

IMPORTANT: this file must sit in the same folder as 'sales_prediction_model.joblib'
(the file your notebook saves in Phase 7). If you keep it elsewhere, update
MODEL_PATH below.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "sales_prediction_model.joblib"

st.set_page_config(page_title="Sales Prediction System", page_icon="\U0001F4C8", layout="centered")


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


saved = load_model(MODEL_PATH)

st.title("\U0001F4C8 Sales Prediction System")
st.caption("Machine Learning-based monthly sales forecasting")

if saved is None:
    st.error(
        f"Could not find **{MODEL_PATH}** in this folder. "
        "Run the notebook's Phase 7 cell (joblib.dump(...)) first, "
        "then place the resulting .joblib file next to this script."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Recover the exact Sub-Category / Market options the model was trained on,
# straight from the one-hot-encoded training columns - no hardcoding, so this
# always matches whatever dataset the notebook was actually trained on.
# ---------------------------------------------------------------------------
def extract_options(columns, prefix):
    return sorted(c[len(prefix):] for c in columns if c.startswith(prefix))


sub_categories = extract_options(saved["columns"], "Sub-Category_")
markets = extract_options(saved["columns"], "Market_")

if not sub_categories or not markets:
    st.error(
        "Couldn't find Sub-Category / Market columns in the saved model. "
        "Make sure sales_prediction_model.joblib was produced by the updated notebook."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Prediction function - mirrors predict_sales() from the notebook
# ---------------------------------------------------------------------------
def predict_sales(sub_category, market, month, prev_sales, avg_price, discount_pct,
                   avg_prev_3=None, avg_prev_6=None, avg_prev_12=None, sales_last_year=None):
    if avg_prev_3 is None:
        avg_prev_3 = prev_sales
    if avg_prev_6 is None:
        avg_prev_6 = avg_prev_3
    if avg_prev_12 is None:
        avg_prev_12 = avg_prev_6
    if sales_last_year is None:
        sales_last_year = prev_sales

    row = pd.DataFrame([{
        "Month": month, "Quarter": (month - 1) // 3 + 1,
        "Prev_Sales": prev_sales, "Avg_Prev_3": avg_prev_3,
        "Avg_Prev_6": avg_prev_6, "Avg_Prev_12": avg_prev_12,
        "Sales_LastYear_SameMonth": sales_last_year,
        "Discount_Pct": discount_pct, "Avg_Price": avg_price,
        "Sub-Category": sub_category, "Market": market,
    }])

    row = pd.get_dummies(row, columns=["Sub-Category", "Market"])
    row = row.reindex(columns=saved["columns"], fill_value=0)
    pred = saved["model"].predict(row)[0]

    if saved["uses_ratio"]:
        pred = pred * max(avg_prev_12, saved["ref_floor"])
    return max(pred, 0)


MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# ---------------------------------------------------------------------------
# Sidebar: model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model info")
    st.write("**Model used:**", saved["name"])
    st.write("**Target type:**", "Ratio-based" if saved["uses_ratio"] else "Direct sales value")
    st.write("**Input features:**", len(saved["columns"]))
    st.caption(
        "This app predicts total monthly Sales for one Sub-Category "
        "in one Market, based on the trained model from the notebook."
    )

# ---------------------------------------------------------------------------
# Main input form
# ---------------------------------------------------------------------------
st.subheader("Enter prediction inputs")

col1, col2 = st.columns(2)
with col1:
    sub_category = st.selectbox("Product Sub-Category", sub_categories)
    market = st.selectbox("Market / Region", markets)
    month_name = st.selectbox("Month to predict", MONTH_NAMES, index=8)
    month = MONTH_NAMES.index(month_name) + 1

with col2:
    prev_sales = st.number_input("Previous month's sales", min_value=0.0, value=9000.0, step=100.0)
    avg_price = st.number_input("Average selling price (per unit)", min_value=0.0, value=180.0, step=5.0)
    discount_pct = st.slider("Average discount (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)

with st.expander("Optional: more history (improves accuracy if you have it)"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_3 = st.checkbox("3-month avg")
        avg_prev_3 = st.number_input("Value", min_value=0.0, value=prev_sales, key="p3", disabled=not use_3)
    with c2:
        use_6 = st.checkbox("6-month avg")
        avg_prev_6 = st.number_input("Value", min_value=0.0, value=prev_sales, key="p6", disabled=not use_6)
    with c3:
        use_12 = st.checkbox("12-month avg")
        avg_prev_12 = st.number_input("Value", min_value=0.0, value=prev_sales, key="p12", disabled=not use_12)
    with c4:
        use_ly = st.checkbox("Same month last year")
        sales_last_year = st.number_input("Value", min_value=0.0, value=prev_sales, key="ly", disabled=not use_ly)

st.divider()

if st.button("Predict Sales", type="primary", use_container_width=True):
    pred = predict_sales(
        sub_category=sub_category,
        market=market,
        month=month,
        prev_sales=prev_sales,
        avg_price=avg_price,
        discount_pct=discount_pct,
        avg_prev_3=avg_prev_3 if use_3 else None,
        avg_prev_6=avg_prev_6 if use_6 else None,
        avg_prev_12=avg_prev_12 if use_12 else None,
        sales_last_year=sales_last_year if use_ly else None,
    )

    st.success("Prediction complete")
    st.metric(
        label=f"Predicted Sales - {sub_category} ({market}, {month_name})",
        value=f"{pred:,.0f}",
    )

    change = (pred - prev_sales) / prev_sales * 100 if prev_sales > 0 else 0
    st.caption(f"That's a {change:+.1f}% change versus last month's sales of {prev_sales:,.0f}.")

st.divider()
st.caption(
    "Note: predictions are estimates based on historical patterns and should be used "
    "as one input among several for business decisions, not a guaranteed figure."
)
