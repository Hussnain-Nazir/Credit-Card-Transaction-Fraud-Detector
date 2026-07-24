"""
Credit Card Transaction Fraud Detector - Streamlit App
"""

import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Transaction Fraud Detector",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# CUSTOM CSS - dark navy blue theme, white text
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Top header bar */
        header[data-testid="stHeader"] {
            background-color: #050d1a;
        }

        /* Main app background */
        .stApp {
            background-color: #050d1a;
            color: #ffffff;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #030811;
            border-right: 1px solid #14243a;
        }
        section[data-testid="stSidebar"] * {
            color: #f5f5f5 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #14243a;
        }

        /* Global text */
        html, body, p, span, label, div {
            color: #ffffff;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #ffffff !important;
            font-weight: 700;
        }

        /* Markdown text in main area */
        .stMarkdown, .stMarkdown p {
            color: #f0f0f0 !important;
        }

        /* Buttons (Predict button included) */
        div.stButton > button,
        div.stFormSubmitButton > button {
            background-color: #1a4fc4;
            color: #ffffff !important;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.4rem;
            font-weight: 700;
            width: 100%;
            transition: background-color 0.2s ease, color 0.2s ease;
        }
        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            background-color: #2563eb;
            color: #ffffff !important;
        }

        /* Form container */
        div[data-testid="stForm"] {
            background-color: #081527;
            border: 1px solid #14243a;
            border-radius: 10px;
            padding: 1.5rem;
        }

        /* Text / number inputs */
        .stNumberInput input, .stTextInput input {
            background-color: #0d1f38 !important;
            color: #ffffff !important;
            border: 1px solid #1e355a !important;
            border-radius: 6px !important;
        }

        /* Selectbox */
        div[data-baseweb="select"] > div {
            background-color: #0d1f38 !important;
            color: #ffffff !important;
            border: 1px solid #1e355a !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] span {
            color: #ffffff !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #0d1f38 !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            color: #ffffff !important;
        }

        /* Number input +/- buttons */
        button[data-testid="stNumberInputStepUp"],
        button[data-testid="stNumberInputStepDown"] {
            background-color: #0d1f38 !important;
            color: #ffffff !important;
            border: 1px solid #1e355a !important;
        }

        /* Labels above inputs */
        label, .stNumberInput label, .stTextInput label, .stSelectbox label {
            color: #cccccc !important;
        }

        /* Result cards */
        .result-card {
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-top: 1rem;
        }
        .result-fraud {
            background-color: #24110f;
            border: 1px solid #e74c3c;
        }
        .result-safe {
            background-color: #0d2a1e;
            border: 1px solid #2ecc71;
        }
        .result-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
            color: #ffffff;
        }
        .result-sub {
            font-size: 0.95rem;
            color: #dddddd;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            background-color: #081527;
            border: 1px solid #14243a;
            border-radius: 8px;
        }
        div[data-testid="stExpander"] summary {
            color: #ffffff !important;
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            background-color: #081527;
        }

        /* Error box */
        div[data-testid="stAlert"] {
            background-color: #14243a;
            color: #ffffff;
        }

        /* Remove extra top padding */
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    preprocessor = joblib.load("models/scaler.pkl")
    metadata = joblib.load("models/metadata.pkl")
    return model, preprocessor, metadata


try:
    model, preprocessor, metadata = load_artifacts()
    ARTIFACTS_LOADED = True
except FileNotFoundError:
    ARTIFACTS_LOADED = False


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/long points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Credit Card Transaction Fraud Detector")
    st.markdown("Real-time credit card transaction screening powered by machine learning.")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app uses a trained **XGBoost** classifier to estimate the "
        "probability that a credit card transaction is fraudulent, based on "
        "transaction amount, merchant category, location, and cardholder details."
    )
    st.markdown("---")
    if ARTIFACTS_LOADED:
        st.markdown("### Model Info")
        st.markdown(f"**Model:** {metadata['best_model_name']}")
        f1 = metadata["metrics"][metadata["best_model_name"]]["F1 Score"]
        auc = metadata["metrics"][metadata["best_model_name"]]["ROC-AUC"]
        st.markdown(f"**F1 Score:** {f1:.3f}")
        st.markdown(f"**ROC-AUC:** {auc:.3f}")
    st.markdown("---")
    st.markdown("Built with Streamlit")

# ---------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------
st.markdown("# Transaction Fraud Check")
st.markdown("Enter the transaction details below to check whether it looks fraudulent.")
st.write("")

if not ARTIFACTS_LOADED:
    st.error(
        "Model artifacts not found. Please run the notebook "
        "(`notebook/fraud_detection_pipeline.ipynb`) first to generate "
        "`models/model.pkl` and `models/scaler.pkl`."
    )
    st.stop()

with st.form("transaction_form"):
    st.markdown("### Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=1.0)
        category = st.selectbox("Merchant Category", options=metadata["categories"])
        gender = st.selectbox("Cardholder Gender", options=["Female", "Male"])
        age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35, step=1)

    with col2:
        city_pop = st.number_input("Cardholder City Population", min_value=0, value=50000, step=100)
        trans_datetime = st.text_input(
            "Transaction Date & Time (YYYY-MM-DD HH:MM)",
            value=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    st.markdown("### Location Details")
    col3, col4 = st.columns(2)
    with col3:
        cust_lat = st.number_input("Cardholder Latitude", value=40.7128, format="%.4f")
        cust_long = st.number_input("Cardholder Longitude", value=-74.0060, format="%.4f")
    with col4:
        merch_lat = st.number_input("Merchant Latitude", value=40.7300, format="%.4f")
        merch_long = st.number_input("Merchant Longitude", value=-73.9950, format="%.4f")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        dt = pd.to_datetime(trans_datetime)
    except Exception:
        st.error("Invalid date/time format. Please use YYYY-MM-DD HH:MM.")
        st.stop()

    distance_km = haversine_distance(cust_lat, cust_long, merch_lat, merch_long)
    gender_encoded = 1 if gender == "Male" else 0

    input_df = pd.DataFrame(
        [{
            "amt": amt,
            "city_pop": city_pop,
            "distance_km": distance_km,
            "age": age,
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "month": dt.month,
            "gender_encoded": gender_encoded,
            "category": category,
        }]
    )

    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    st.write("")
    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card result-fraud">
                <div class="result-title">⚠️ Likely Fraudulent Transaction</div>
                <div class="result-sub">Estimated fraud probability: {probability * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-safe">
                <div class="result-title">✅ Transaction Looks Genuine</div>
                <div class="result-sub">Estimated fraud probability: {probability * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View computed features"):
        st.dataframe(input_df.assign(distance_km=round(distance_km, 2)))
