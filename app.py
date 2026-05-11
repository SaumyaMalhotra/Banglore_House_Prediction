import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 { color: #e94560; font-size: 2rem; margin: 0; }
.hero p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1rem; }

.result-card {
    background: linear-gradient(135deg, #0f3460, #e94560);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(233,69,96,0.35);
    margin-top: 1.5rem;
}
.result-card .label { color: #fff; font-size: 0.95rem; opacity: 0.85; }
.result-card .price { color: #fff; font-size: 2.4rem; font-weight: 700; }

div[data-testid="stNumberInput"] > div > input { border-radius: 8px; }
div[data-testid="stSelectbox"] > div     { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("banglore_home_prices_model.pickle", "rb") as f:
        return pickle.load(f)

model = load_model()

data_columns = list(model.feature_names_in_)

NUMERIC_COLS = {'total_sqft', 'bath', 'balcony', 'bhk'}
locations = sorted([c for c in data_columns if c not in NUMERIC_COLS])

st.markdown("""
<div class="hero">
    <h1>🏠 Bangalore House Price Predictor</h1>
    <p>Enter property details below to get an instant price estimate</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_sqft = st.number_input("📐 Total Square Feet", min_value=300, max_value=30000,
                                  value=500, step=50)
    bath = st.number_input("🚿 Bathrooms", min_value=1, max_value=15, value=2, step=1)

with col2:
    bhk = st.number_input("🛏️ BHK", min_value=1, max_value=15, value=2, step=1)
    balcony = st.number_input("🌿 Balconies", min_value=0, max_value=5, value=1, step=1)

location = st.selectbox("📍 Location", locations)

def predict_price(location, sqft, bath, bhk, balcony):
    cols = pd.Index(data_columns)

    loc_index = np.where(cols == location)[0]
    loc_index = loc_index[0] if len(loc_index) > 0 else -1

    x = np.zeros(len(cols))
    x[cols.get_loc('total_sqft')] = sqft
    x[cols.get_loc('bath')]       = bath
    x[cols.get_loc('balcony')]    = balcony
    x[cols.get_loc('bhk')]        = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict(pd.DataFrame([x], columns=cols))[0]

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Predict Price", use_container_width=True):
    try:
        price = predict_price(location, total_sqft, bath, bhk, balcony)
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Property Value</div>
            <div class="price">₹ {round(price, 2)} Lakhs</div>
            <div class="label" style="margin-top:0.5rem;font-size:0.8rem;">
                ≈ ₹ {round(price * 100000):,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
