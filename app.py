import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for modern design & shadow effects
st.markdown("""
    <style>
    /* Main container background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Section with Soft Shadow */
    .header-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4), 0 8px 10px -6px rgba(79, 70, 229, 0.2);
        margin-bottom: 2rem;
    }
    .header-card h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Form Container Card with Soft Shadow */
    .form-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03);
        border: 1px solid #edf2f7;
        margin-bottom: 2rem;
    }
    
    /* Input Field Styling */
    div[data-baseweb="select"], div[data-baseweb="input"] {
        border-radius: 10px !important;
    }
    
    /* Predict Button Styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5);
        color: white;
    }
    
    /* Output Result Box */
    .result-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4F46E5;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING WITH CACHING
# ==========================================
MODEL_PATH = "gradient.pkl"

@st.cache_resource
def load_model():
    """Loads the pickled model and caches it in memory."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ==========================================
# 3. UI LAYOUT & INPUT FIELDS
# ==========================================
st.markdown("""
    <div class="header-card">
        <h1>Model Inference Studio</h1>
        <p>Customer Purchase Prediction Portal</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.subheader("📋 Enter Input Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    
    # Categorical Column 1: Gender
    gender_map = {"Female": 0, "Male": 1}
    gender_selected = st.selectbox("Gender", options=list(gender_map.keys()))

with col2:
    # Categorical Column 2: Review Grade
    review_map = {"Poor": 0, "Average": 1, "Good": 2}
    review_selected = st.selectbox("Review Grade", options=list(review_map.keys()))
    
    # Categorical Column 3: Education Level
    education_map = {"School / High School": 0, "UG / Graduate": 1, "PG / Post Graduate": 2}
    education_selected = st.selectbox("Education Level", options=list(education_map.keys()))

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 4. PREDICTION INFERENCE
# ==========================================
if st.button("🚀 Run Prediction"):
    try:
        model = load_model()
        
        # Structure features into DataFrame expected by model
        input_data = pd.DataFrame([{
            "age": float(age),
            "gender": int(gender_map[gender_selected]),
            "review": int(review_map[review_selected]),
            "education": int(education_map[education_selected])
        }])
        
        # Execute prediction
        prediction = model.predict(input_data)[0]
        
        # Extract probability if model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            prob_no = f"{probabilities[0]*100:.1f}%"
            prob_yes = f"{probabilities[1]*100:.1f}%"
            prob_text = f"Confidence (No): <b>{prob_no}</b> | Confidence (Yes): <b>{prob_yes}</b>"
        else:
            prob_text = "Probability scores not available for this model."

        # Display Result Card
        st.markdown(f"""
            <div class="result-card">
                <h3 style="margin:0; color:#1E293B;">Prediction Result: <span style="color:#4F46E5;">{prediction}</span></h3>
                <p style="margin-top:8px; color:#64748B;">{prob_text}</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
        
    except FileNotFoundError:
        st.error(f"Could not find `{MODEL_PATH}`. Make sure `gradient.pkl` is pushed to your GitHub repository.")
    except Exception as err:
        st.error(f"Error processing prediction: {err}")
