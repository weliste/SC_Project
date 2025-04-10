import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Configure page
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:18px !important;
        color: #2e86de;
    }
    .risk-high {
        color: #ee5253;
        font-weight: bold;
        font-size: 24px;
    }
    .risk-medium {
        color: #ff9f43;
        font-weight: bold;
        font-size: 24px;
    }
    .risk-low {
        color: #1dd1a1;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Check if model exists
if not os.path.exists('diabetes_risk_nn.h5'):
    st.error("Model file 'diabetes_risk_nn.h5' not found in current directory")
    st.stop()

# Load model with error handling
@st.cache_resource
def load_model_safe():
    try:
        return load_model('diabetes_risk_nn.h5')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model_safe()

# App header
st.title("Diabetes Risk Assessment")
st.markdown("This tool predicts diabetes risk based on health metrics.", unsafe_allow_html=True)

# Input form
with st.form("diabetes_form"):
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    
    submitted = st.form_submit_button("Calculate Risk", type="primary")

# When form is submitted
if submitted:
    st.markdown("---")
    st.subheader("Results")
    
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0][0]
        risk_percentage = round(prediction * 100, 1)
        
        # Display progress bar
        st.progress(int(risk_percentage))
        
        # Display risk level with color coding
        if risk_percentage < 30:
            risk_class = "low"
            rec = "Low risk - Maintain healthy lifestyle"
        elif risk_percentage < 70:
            risk_class = "medium"
            rec = "Moderate risk - Consider lifestyle changes"
        else:
            risk_class = "high"
            rec = "High risk - Consult a healthcare provider"
        
        st.markdown(f"""
        <div style='text-align: center; margin: 20px 0;'>
            <p class='big-font'>Diabetes Risk Score</p>
            <p class='risk-{risk_class}'>{risk_percentage}%</p>
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed interpretation
        with st.expander("Detailed Interpretation"):
            st.markdown(f"""
            - **Risk Score**: {risk_percentage}%
            - **Interpretation**: {rec}
            - **Recommendation**: {rec}
            
            **Clinical Parameters**:
            - Glucose: {glucose} mg/dL (Normal range: 70-140)
            - BMI: {bmi} (Normal range: 18.5-24.9)
            - Blood Pressure: {blood_pressure} mmHg (Normal: <120/80)
            """)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*For educational purposes only. Not medical advice.*")