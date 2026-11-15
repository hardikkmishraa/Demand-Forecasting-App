import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Demand Forecasting App", page_icon="📈", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    
    return model, encoders

model, label_encoders = load_artifacts()

st.title("📈 Demand Forecasting App")
st.write("Predict future product demand based on pricing, promotions, and inventory levels.")
st.divider()

st.header("Input Features")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price ($)", min_value=0.0, value=50.0, step=1.0)
    discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=10, step=1)
    inventory_level = st.number_input("Inventory Level", min_value=0, value=100, step=1)

with col2:
    promotion = st.selectbox("Promotion", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    competitor_pricing = st.number_input("Competitor Price ($)", min_value=0.0, value=50.0, step=1.0)
    category = st.selectbox("Category", options=label_encoders["Category"].classes_.tolist())

# Encode the categorical feature
encoded_category = label_encoders["Category"].transform([category])[0]

# Prepare DataFrame matching the exact training feature names and order
input_data = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory Level": [inventory_level],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category": [encoded_category]
})

st.divider()

# Predict button
if st.button("🔮 Predict Demand", use_container_width=True, type="primary"):
    prediction = model.predict(input_data)[0]
    predicted_demand = max(0, round(float(prediction)))

    st.subheader("Forecast Result")
    st.metric(label="Predicted Demand (Units)", value=f"{predicted_demand:,}")
    
    with st.expander("View Input Data"):
        display_df = pd.DataFrame({
            "Price": [price],
            "Discount (%)": [discount],
            "Inventory Level": [inventory_level],
            "Promotion": ["Yes" if promotion == 1 else "No"],
            "Competitor Price": [competitor_pricing],
            "Category": [category]
        })
        st.dataframe(display_df, use_container_width=True)
