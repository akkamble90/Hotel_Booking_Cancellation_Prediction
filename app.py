import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re

# 1. ASSET LOADING
@st.cache_resource
def load_all_assets():
    m = joblib.load('hotel_model.pkl')
    s = joblib.load('scaler.pkl')
    e = joblib.load('encoder.pkl')
    return m, s, e

# 2. USER INTERFACE (UI) & CONFIGURATION
st.set_page_config(page_title="Hotel Booking Risk Analyzer", layout="wide")

# Sidebar Metrics
st.sidebar.header("Model Performance")
st.sidebar.caption("Stacking Ensemble (ExtraTrees + RF)")
st.sidebar.metric("Test Accuracy", "89.1%")
st.sidebar.metric("ROC-AUC Score", "0.961")
st.sidebar.metric("Recall (Detection)", "87.2%")
st.sidebar.metric("Precision", "81.0%")
st.sidebar.divider()
st.sidebar.info("Trained on reservations balanced with SMOTE & Optuna Bayesian Tuning.")

# Initialize assets
try:
    model, scaler, encoder = load_all_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.info("Ensure hotel_model.pkl, scaler.pkl, and encoder.pkl are in the project directory.")

st.title("Hotel Booking Cancellation Predictor")
st.markdown("""
This AI system predicts the probability of a reservation being canceled using a Stacking Ensemble model.
*Developed for Revenue Management Optimization.*
""")

# Input Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" Timing & Market")
    lead_time = st.number_input("Lead Time (Days in Advance)", min_value=0, value=30)
    market_segment = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'])
    avg_price = st.number_input("Average Price per Night ($)", min_value=0.0, value=100.0)

with col2:
    st.subheader(" Guest Details")
    adults = st.number_input("Number of Adults", min_value=1, value=2)
    children = st.number_input("Number of Children", min_value=0, value=0)
    meal_type = st.selectbox("Type of Meal", ['Meal Plan 1', 'Meal Plan 2', 'Not Selected', 'Meal Plan 3'])

with col3:
    st.subheader(" Room & Stay")
    weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
    week_nights = st.number_input("Week Nights", min_value=0, value=2)
    room_type = st.selectbox("Room Type", [f'Room_Type {i}' for i in range(1, 8)])

st.divider()
special_requests = st.slider("Number of Special Requests (Guest Engagement)", 0, 5, 1)

# 3. PREDICTION LOGIC
if st.button("Analyze Cancellation Risk", type="primary", use_container_width=True):
    # Create input dictionary
    input_dict = {
        'number of adults': adults,
        'number of children': children,
        'number of weekend nights': weekend_nights,
        'number of week nights': week_nights,
        'type of meal': meal_type,
        'room type': room_type,
        'lead time': lead_time,
        'market segment type': market_segment,
        'average price ': avg_price,
        'special requests': special_requests,
        'car parking space': 0,
        'repeated': 0,
        'P-C': 0,
        'P-not-C': 0,
        'date of reservation': '2025-10-10'  # Placeholder for feature engineering
    }

    input_df = pd.DataFrame([input_dict])

    # 1. Clean names to match training format
    input_df.columns = [re.sub(r'\s+', '_', col.strip()).lower() for col in input_df.columns]

    # 2. Feature Engineering
    input_df['total_guests'] = input_df['number_of_adults'] + input_df['number_of_children']
    input_df['total_stay'] = input_df['number_of_weekend_nights'] + input_df['number_of_week_nights']
    input_df['price_per_person'] = input_df['average_price'] / (input_df['total_guests'] + 0.1)
    input_df['res_month'] = 10
    input_df['res_dayofweek'] = 4

    # 3. Drop unneeded raw columns
    cols_to_drop = ['date_of_reservation', 'booking_id', 'booking_status']
    input_df.drop(columns=[c for c in cols_to_drop if c in input_df.columns], inplace=True, errors='ignore')

    # 4. Transform & Predict
    try:
        # Create a working copy for encoding
        encoded_df = input_df.copy()
        
        # Identify categorical columns used during training
        cat_cols = ['type_of_meal', 'room_type', 'market_segment_type']
        
        # Target Encode the 3 categorical columns
        encoded_df[cat_cols] = encoder.transform(encoded_df[cat_cols])
        
        # Align all 19 columns exactly with the order expected by the StandardScaler
        encoded_df = encoded_df[scaler.feature_names_in_]

        # Standard Scale all 19 features
        scaled_data = scaler.transform(encoded_df)

        # Get probability of cancellation (Class 1)
        prob = model.predict_proba(scaled_data)[0][1]

        # 5. Display Result
        st.markdown("---")
        st.header(f"Risk Assessment: {prob:.1%}")

        if prob > 0.75:
            st.error(" **CRITICAL RISK:** High probability of cancellation. Recommend mandatory non-refundable deposit.")
        elif prob > 0.40:
            st.warning(" **MODERATE RISK:** Standard risk level. Send a confirmation reminder email or offer an add-on.")
        else:
            st.success(" **LOW RISK:** Reliable booking. High confidence in guest arrival.")

    except Exception as e:
        st.error(f"Prediction Error: {e}. Please ensure inputs match training data format.")

# Footer
st.caption("Developed by Akhil Kamble | COEP Technological University")
