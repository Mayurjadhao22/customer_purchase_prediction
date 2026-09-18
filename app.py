# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Ensure scikit-learn is imported so unpickling can find all modules
import sklearn
import sklearn.ensemble

# Page Configuration
st.set_page_config(
    page_title="Model Predictor",
    page_icon="🔮",
    layout="centered"
)

# Custom Styling & Animations
st.markdown("""
<style>
    /* Main App Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Card Containers */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
    }

    /* Animated Prediction Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5);
        color: #ffffff;
    }

    /* Result Pulse Animation */
    @keyframes pulse {
        0% { transform: scale(0.98); opacity: 0.8; }
        50% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }

    .result-container {
        animation: pulse 0.6s ease-out forwards;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 2px solid #a855f7;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open('gradient.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `gradient.pkl`: {e}")
    st.info("Ensure `scikit-learn` is installed in your Python environment via `pip install scikit-learn`.")
    st.stop()

# Header Section
st.title("🔮 AI Prediction Portal")
st.caption("Gradient Boosting Classifier Inference Pipeline")
st.markdown("---")

# Input Form
with st.container():
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📋 Enter Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", options=["Male", "Female"])

    with col2:
        review = st.selectbox("Review Rating", options=["Poor", "Average", "Good"])
        education = st.selectbox("Education Level", options=["School", "UG", "PG"])

    st.markdown('</div>', unsafe_allow_html=True)

# Feature Encoding Map
gender_map = {"Female": 0, "Male": 1}
review_map = {"Average": 0, "Good": 1, "Poor": 2}
education_map = {"PG": 0, "School": 1, "UG": 2}

# Prediction Logic & Effects
if st.button("🚀 Predict Outcome"):
    # Trigger Confetti Celebration Effect
    st.balloons()

    # Progress bar effect
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i + 1)
        status_text.text(f"Analyzing inputs... {i+1}%")

    status_text.empty()
    progress_bar.empty()

    # Prepare input array
    encoded_inputs = np.array([[
        age,
        gender_map[gender],
        review_map[review],
        education_map[education]
    ]])

    # Make Prediction
    prediction = model.predict(encoded_inputs)[0]
    probabilities = model.predict_proba(encoded_inputs)[0]

    # Display Results with Animation
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    if str(prediction).lower() in ["yes", "1"]:
        st.success(f"### Prediction Result: **{prediction}** 🎉")
    else:
        st.info(f"### Prediction Result: **{prediction}**")

    # Display class probabilities
    classes = getattr(model, "classes_", ["Class 0", "Class 1"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"Probability ({classes[0]})", f"{probabilities[0]*100:.1f}%")
    with col_b:
        st.metric(f"Probability ({classes[1]})", f"{probabilities[1]*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)
