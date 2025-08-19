import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

#Custom inline CSS for theme
st.markdown(
    """
    <style>
    /* Main app */
    .stApp {
        background-color: #FFFFFF;
        color: #222222 !important;
        font-size: 16px;
    }

    /* Input labels */
    label, .stNumberInput label, .stSelectbox label {
        color: #222222 !important;
        font-weight: 600;
    }

    /* Input boxes (number inputs, dropdowns, text fields) */
    input, select, textarea {
        background-color: #F0FFF0 !important;  /* Honeydew light green */
        color: #222222 !important;  /* Dark text */
        border: 1px solid #228B22 !important;  /* Green border */
        border-radius: 6px;
        padding: 0.3em;
    }

    /* Buttons */
    .stButton>button {
        background-color: #228B22 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1em;
        font-weight: bold;
        font-size: 15px;
    }
    .stButton>button:hover {
        background-color: #196619 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True

)
# Load models and encoders
clf = joblib.load('models/fertilizer_type_model.pkl')
reg = joblib.load('models/fertilizer_amount_model.pkl')
crop_encoder = joblib.load('models/crop_encoder.pkl')
fertilizer_encoder = joblib.load('models/fertilizer_encoder.pkl')

# Page title and description
st.set_page_config(page_title="🌾 Optimal Fertilizer Prediction", layout="wide")
st.title("🌾 Optimal Fertilizer Prediction App")
st.markdown("Predict the best fertilizer type and application rate based on soil and weather conditions.")

# Sidebar for background info
st.sidebar.header("📌 About This Project")
st.sidebar.write("""
This tool uses Random Forest Machine Learning Models to recommend 
the optimal fertilizer type and amount for a given crop based on soil and climate data.
""")
st.sidebar.write("Dataset size: 900+ rows")
st.sidebar.write("Models: Random Forest Classifier & Regressor for fertilizer type and fertilizer amount respectively.")
st.sidebar.write("Made with: Streamlit, scikit-learn, pandas.")

# User inputs
st.subheader("Enter Soil & Climate Data")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=70)
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=20)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=150)
with col3:
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=30)
    crop_list = list(crop_encoder.classes_)
    crop = st.selectbox("Select Crop", crop_list)

# Encode crop input
crop_enc = crop_encoder.transform([crop])[0]

# Prepare input for prediction
features = np.array([[N, P, K, pH, temperature, humidity, rainfall, crop_enc]])

if st.button("🔍 Predict Optimal Fertilizer"):
    # Predict fertilizer type (encoded)
    pred_type_probs = clf.predict_proba(features)[0]
    pred_type_enc = np.argmax(pred_type_probs)
    pred_type = fertilizer_encoder.inverse_transform([pred_type_enc])[0]
    confidence = pred_type_probs[pred_type_enc] * 100

    # Predict fertilizer amount
    pred_amount = reg.predict(features)[0]

    # Display predictions
    st.subheader("Prediction Results")
    st.success(f"Recommended Fertilizer: {pred_type} (Confidence: {confidence:.1f}%)")
    st.success(f"Recommended Amount: {pred_amount:.2f} kg/ha")

    # Farming tips
    st.subheader("🌱 Farming Tip")
    tips = {
        "Urea": "Apply in moist soil and avoid direct sunlight exposure to reduce nitrogen loss.",
        "NPK 20-10-10": "Great for boosting vegetative growth; ensure balanced watering.",
        "NPK 15-15-15": "Good general-purpose fertilizer; apply before rainfall for best results.",
        "Ammonium Sulphate": "Helps in acidic soils; avoid overuse to prevent acidification.",
        "Asaase Wura": "Local organic blend; improves soil health over time.",
        "Organic Cocoa Blend": "Perfect for cocoa; improves bean size and yield.",
        "NPK 12-15-17": "Best for root crops; promotes tuber growth.",
        "Single Super Phosphate": "Boosts flowering and seed formation.",
        "Generic NPK": "Balanced nutrient source; apply evenly.",
        "NPK 12-10-18": "Supports strong stems and root systems."
    }
    st.info(tips.get(pred_type, "Use recommended dosage and follow local agricultural extension advice."))

    # Feature importance chart
    st.subheader("📊 Feature Importance (Model Insights)")
    importances = clf.feature_importances_
    feature_names = ["N", "P", "K", "pH", "Temperature", "Humidity", "Rainfall", "Crop"]
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Influence on Fertilizer Type Prediction")
    st.pyplot(fig)










