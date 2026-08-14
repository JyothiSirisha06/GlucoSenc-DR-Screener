"""
frontend.py
===========
Interactive Streamlit Web Application for GlucoSense: Diabetes Risk Screener.

Provides telemetry input controls (sliders, numeric inputs, preset sample profiles),
queries the FastAPI backend (`http://localhost:8000/predict`), and renders dynamic
risk metrics, visual indicators, and personalized clinical guidance.
"""

import requests
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Page Config & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GlucoSense | Diabetes Risk Screener",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern telemetry cards
st.markdown("""
<style>
    /* Main container tweaks */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Header hero section */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-left: 6px solid #3B82F6;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        color: white;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #F8FAFC;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Badge Container */
    .risk-banner {
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Input section cards */
    .section-header {
        font-weight: 600;
        font-size: 1.1rem;
        color: #1E293B;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Configuration & Backend Connectivity Check
# -----------------------------------------------------------------------------
DEFAULT_BACKEND_URL = "http://localhost:8000"

st.sidebar.markdown("### ⚙️ System Settings")
backend_url = st.sidebar.text_input("Backend API Endpoint URL", value=DEFAULT_BACKEND_URL)


@st.cache_data(ttl=5)
def check_backend_status(url: str) -> dict:
    try:
        response = requests.get(f"{url}/", timeout=2.5)
        if response.status_code == 200:
            return {"online": True, "data": response.json()}
    except Exception:
        pass
    return {"online": False, "data": None}


server_status = check_backend_status(backend_url)

if server_status["online"]:
    model_name = server_status["data"].get("model_loaded", "Active")
    st.sidebar.success(f"🟢 **FastAPI Online**\n\nModel: `{model_name}`")
else:
    st.sidebar.error("🔴 **FastAPI Offline**\n\nStart server: `uvicorn backend:app --port 8000`")


# -----------------------------------------------------------------------------
# 3. Preset Session State Management
# -----------------------------------------------------------------------------
if "pregnancies" not in st.session_state:
    st.session_state.pregnancies = 1
if "glucose" not in st.session_state:
    st.session_state.glucose = 115.0
if "blood_pressure" not in st.session_state:
    st.session_state.blood_pressure = 72.0
if "skin_thickness" not in st.session_state:
    st.session_state.skin_thickness = 22.0
if "insulin" not in st.session_state:
    st.session_state.insulin = 75.0
if "bmi" not in st.session_state:
    st.session_state.bmi = 28.5
if "diabetes_pedigree" not in st.session_state:
    st.session_state.diabetes_pedigree = 0.38
if "age" not in st.session_state:
    st.session_state.age = 38


def apply_preset(preset_type: str):
    if preset_type == "healthy":
        st.session_state.pregnancies = 0
        st.session_state.glucose = 88.0
        st.session_state.blood_pressure = 68.0
        st.session_state.skin_thickness = 18.0
        st.session_state.insulin = 45.0
        st.session_state.bmi = 22.4
        st.session_state.diabetes_pedigree = 0.18
        st.session_state.age = 26
    elif preset_type == "moderate":
        st.session_state.pregnancies = 2
        st.session_state.glucose = 118.0
        st.session_state.blood_pressure = 78.0
        st.session_state.skin_thickness = 26.0
        st.session_state.insulin = 110.0
        st.session_state.bmi = 29.8
        st.session_state.diabetes_pedigree = 0.45
        st.session_state.age = 44
    elif preset_type == "high":
        st.session_state.pregnancies = 5
        st.session_state.glucose = 168.0
        st.session_state.blood_pressure = 88.0
        st.session_state.skin_thickness = 34.0
        st.session_state.insulin = 210.0
        st.session_state.bmi = 36.2
        st.session_state.diabetes_pedigree = 0.82
        st.session_state.age = 56


st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Load Test Profiles")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)
if col_p1.button("🟢 Low", help="Load normal healthy telemetry"):
    apply_preset("healthy")
if col_p2.button("🟡 Med", help="Load elevated telemetry"):
    apply_preset("moderate")
if col_p3.button("🔴 High", help="Load diabetic telemetry"):
    apply_preset("high")


# -----------------------------------------------------------------------------
# 4. Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🩺 GlucoSense | Diabetes Risk Screener</div>
    <div class="hero-subtitle">Production AI Clinical Screening Platform powered by Machine Learning & REST API Inference</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Patient Telemetry Input Controls
# -----------------------------------------------------------------------------
st.subheader("📋 Patient Clinical Telemetry")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">🩸 Metabolic & Glycemic Telemetry</div>', unsafe_allow_html=True)
    
    glucose = st.slider(
        "Fasting / Random Plasma Glucose (mg/dL)",
        min_value=40.0, max_value=250.0,
        value=float(st.session_state.glucose), step=1.0,
        help="Normal: <100 mg/dL, Pre-diabetic: 100-125 mg/dL, Diabetic: ≥126 mg/dL"
    )
    
    bmi = st.number_input(
        "Body Mass Index - BMI (kg/m²)",
        min_value=10.0, max_value=65.0,
        value=float(st.session_state.bmi), step=0.1,
        help="Normal: 18.5-24.9, Overweight: 25.0-29.9, Obese: ≥30.0"
    )
    
    insulin = st.number_input(
        "2-Hour Serum Insulin (μU/mL)",
        min_value=0.0, max_value=700.0,
        value=float(st.session_state.insulin), step=1.0,
        help="Normal fasting range: 16-166 μU/mL"
    )
    
    skin_thickness = st.slider(
        "Triceps Skin Fold Thickness (mm)",
        min_value=5.0, max_value=80.0,
        value=float(st.session_state.skin_thickness), step=1.0,
        help="Subcutaneous fat marker"
    )

with col_right:
    st.markdown('<div class="section-header">👤 Demographics & Vital Indicators</div>', unsafe_allow_html=True)
    
    age = st.slider(
        "Patient Age (Years)",
        min_value=18, max_value=95,
        value=int(st.session_state.age), step=1
    )
    
    blood_pressure = st.slider(
        "Diastolic Blood Pressure (mmHg)",
        min_value=40.0, max_value=140.0,
        value=float(st.session_state.blood_pressure), step=1.0,
        help="Normal: <80 mmHg, Pre-hypertension: 80-89 mmHg, Stage 2: ≥90 mmHg"
    )
    
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function Index",
        min_value=0.05, max_value=2.80,
        value=float(st.session_state.diabetes_pedigree), step=0.01,
        help="Genetic score based on family history & relative incidence"
    )
    
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=18,
        value=int(st.session_state.pregnancies), step=1
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Screening Submission & Backend API Call
# -----------------------------------------------------------------------------
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    run_screening = st.button(
        "🚀 RUN CLINICAL DIABETES RISK SCREENING",
        use_container_width=True,
        type="primary"
    )

if run_screening:
    payload = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "age": age
    }
    
    with st.spinner("Analyzing patient telemetry and querying FastAPI prediction server..."):
        try:
            response = requests.post(f"{backend_url}/predict", json=payload, timeout=5.0)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("## 📊 Screening Results & Assessment")
                
                risk_pct = result["risk_score_percent"]
                status_text = result["status"]
                color_code = result["color_code"]
                model_used = result["best_model_used"]
                guidance_list = result["clinical_guidance"]
                elevated_list = result["elevated_biomarkers"]
                model_metrics = result["model_metrics"]
                
                # Risk Banner
                st.markdown(f"""
                <div class="risk-banner" style="background-color: {color_code};">
                    <h2 style="margin: 0; color: white;">SCREENING RESULT: {status_text.upper()}</h2>
                    <h3 style="margin: 5px 0 0 0; color: white; opacity: 0.95;">Estimated Probability: {risk_pct}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress Bar
                st.progress(min(int(risk_pct), 100))
                
                # Summary Metric Cards
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    st.metric("Risk Score", f"{risk_pct}%")
                with m_col2:
                    st.metric("Risk Category", result["risk_level"])
                with m_col3:
                    st.metric("ML Classifier", model_used)
                with m_col4:
                    f1_score_val = model_metrics.get("f1_score", "N/A")
                    st.metric("Model F1-Score", f"{f1_score_val}")
                
                st.markdown("---")
                
                # Detailed Insights
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("### ⚠️ Biomarker & Risk Factors")
                    if elevated_list:
                        for bio in elevated_list:
                            st.warning(f"• {bio}")
                    else:
                        st.success("✅ All evaluated clinical telemetry parameters fall within standard physiological reference bounds.")
                
                with res_col2:
                    st.markdown("### 📋 Clinical Guidance & Next Steps")
                    for guide in guidance_list:
                        st.info(f"💡 {guide}")
                        
                with st.expander("🔍 Inspect Full Telemetry Payload & Model Metrics"):
                    st.json({
                        "input_telemetry": payload,
                        "prediction_output": result
                    })

            else:
                st.error(f"❌ Backend returned error standard {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"⚠️ **Could not connect to FastAPI server at `{backend_url}`**.\n\nPlease launch the backend server in your terminal:\n```bash\nuvicorn backend:app --reload --port 8000\n```")
        except Exception as e:
            st.error(f"❌ An error occurred during screening: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "GlucoSense Diabetes Screener Prototype | Standard Clinical Telemetry Engine | Built with Streamlit, FastAPI, and Scikit-Learn / XGBoost"
    "</div>",
    unsafe_allow_html=True
)
