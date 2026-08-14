"""
backend.py
==========
FastAPI REST API Server for Diabetes Risk Screener.

Loads serialized `diabetes_model.joblib` artifact at startup, validates patient
telemetry payloads using Pydantic schemas, runs inference, and returns risk scores,
categorical risk statuses, and tailored clinical guidance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Pydantic Schemas
# -----------------------------------------------------------------------------
class PatientData(BaseModel):
    pregnancies: int = Field(
        default=1, ge=0, le=20,
        description="Number of pregnancies"
    )
    glucose: float = Field(
        default=115.0, ge=30.0, le=300.0,
        description="Fasting / random plasma glucose concentration (mg/dL)"
    )
    blood_pressure: float = Field(
        default=72.0, ge=30.0, le=200.0,
        description="Diastolic blood pressure (mmHg)"
    )
    skin_thickness: float = Field(
        default=22.0, ge=0.0, le=100.0,
        description="Triceps skin fold thickness (mm)"
    )
    insulin: float = Field(
        default=75.0, ge=0.0, le=900.0,
        description="2-Hour serum insulin (mu U/ml)"
    )
    bmi: float = Field(
        default=28.5, ge=10.0, le=75.0,
        description="Body Mass Index (weight in kg / (height in m)^2)"
    )
    diabetes_pedigree: float = Field(
        default=0.38, ge=0.05, le=3.0,
        description="Diabetes pedigree function score (genetic likelihood)"
    )
    age: int = Field(
        default=38, ge=1, le=120,
        description="Age in years"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "pregnancies": 2,
                "glucose": 140.0,
                "blood_pressure": 82.0,
                "skin_thickness": 28.0,
                "insulin": 120.0,
                "bmi": 32.4,
                "diabetes_pedigree": 0.65,
                "age": 45
            }
        }
    }


class PredictionResponse(BaseModel):
    risk_score_percent: float
    probability: float
    status: str  # "High Risk" or "Low Risk"
    risk_level: str  # "Low", "Moderate", "High"
    color_code: str
    best_model_used: str
    clinical_guidance: List[str]
    elevated_biomarkers: List[str]
    model_metrics: Dict[str, float]


# -----------------------------------------------------------------------------
# 2. Global State & App Setup
# -----------------------------------------------------------------------------
MODEL_FILE = "diabetes_model.joblib"
model_artifact: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads ML model artifact into memory upon startup."""
    global model_artifact
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(
            f"Model file '{MODEL_FILE}' not found! Run 'python model_pipeline.py' first to train and save the model."
        )

    try:
        model_artifact = joblib.load(MODEL_FILE)
        print(f"[+] Loaded ML model artifact successfully from '{MODEL_FILE}'.")
        print(f"    Selected Model: {model_artifact.get('best_model_name')}")
        print(f"    Features Expected: {model_artifact.get('feature_names')}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model file '{MODEL_FILE}': {str(e)}")

    yield
    model_artifact.clear()


app = FastAPI(
    title="GlucoSense API - Diabetes Risk Engine",
    description="Production REST API providing inference and clinical risk stratification for diabetes screening.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# 3. Clinical Guidance Helper
# -----------------------------------------------------------------------------
def generate_clinical_guidance(patient: PatientData, probability: float) -> tuple[List[str], List[str]]:
    guidance = []
    elevated = []

    # Biomarker analysis
    if patient.glucose >= 126.0:
        elevated.append(f"Elevated Fasting Glucose: {patient.glucose} mg/dL (Diabetic Range >= 126 mg/dL)")
        guidance.append("Schedule a formal diagnostic HbA1c test and Fasting Plasma Glucose test with a physician.")
    elif patient.glucose >= 100.0:
        elevated.append(f"Impaired Glucose: {patient.glucose} mg/dL (Pre-diabetic Range 100-125 mg/dL)")
        guidance.append("Adopt low-glycemic dietary modifications and limit refined sugars to manage glucose levels.")

    if patient.bmi >= 30.0:
        elevated.append(f"Class I/II Obesity: BMI {patient.bmi} kg/m^2 (Threshold >= 30.0)")
        guidance.append("Incorporate 150+ minutes of moderate aerobic exercise weekly to enhance insulin sensitivity.")
    elif patient.bmi >= 25.0:
        elevated.append(f"Overweight: BMI {patient.bmi} kg/m^2")
        guidance.append("Maintain routine physical activity and monitor weight trends.")

    if patient.blood_pressure >= 90.0:
        elevated.append(f"Elevated Diastolic BP: {patient.blood_pressure} mmHg (Stage 2 Hypertension >= 90 mmHg)")
        guidance.append("Monitor blood pressure regularly and consult a doctor regarding cardiovascular risk.")
    elif patient.blood_pressure >= 80.0:
        elevated.append(f"Pre-hypertensive Diastolic BP: {patient.blood_pressure} mmHg")

    if patient.diabetes_pedigree >= 0.5:
        elevated.append(f"Strong Familial Risk Factor: Pedigree Index {patient.diabetes_pedigree:.2f}")
        guidance.append("Inform your primary healthcare provider of your strong familial predisposition.")

    if patient.age >= 45:
        guidance.append("Annual routine metabolic panel screening is recommended for individuals aged 45+.")

    if not guidance:
        guidance.append("Telemetry values fall within normal physiological reference ranges. Continue healthy lifestyle habits.")

    return guidance, elevated


# -----------------------------------------------------------------------------
# 4. Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "GlucoSense Diabetes Risk Screener API",
        "model_loaded": model_artifact.get("best_model_name", "Unknown"),
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": model_artifact.get("training_timestamp"),
        "model_name": model_artifact.get("best_model_name"),
        "metrics": model_artifact.get("metrics", {}).get(model_artifact.get("best_model_name", ""), {})
    }


@app.get("/predict", tags=["Inference Info"])
def predict_info():
    return {
        "message": "The /predict endpoint requires an HTTP POST request containing patient telemetry data.",
        "interactive_docs": "/docs",
        "example_payload": {
            "pregnancies": 2,
            "glucose": 140.0,
            "blood_pressure": 82.0,
            "skin_thickness": 28.0,
            "insulin": 120.0,
            "bmi": 32.4,
            "diabetes_pedigree": 0.65,
            "age": 45
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_risk(patient: PatientData):
    if not model_artifact:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Model not initialized."
        )

    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_names = model_artifact["feature_names"]
    best_model_name = model_artifact["best_model_name"]
    all_metrics = model_artifact.get("metrics", {})

    # Construct input dataframe matching exact feature order
    input_dict = {
        "pregnancies": [patient.pregnancies],
        "glucose": [patient.glucose],
        "blood_pressure": [patient.blood_pressure],
        "skin_thickness": [patient.skin_thickness],
        "insulin": [patient.insulin],
        "bmi": [patient.bmi],
        "diabetes_pedigree": [patient.diabetes_pedigree],
        "age": [patient.age]
    }
    input_df = pd.DataFrame(input_dict)[feature_names]

    # Preprocessing
    input_scaled = scaler.transform(input_df)

    # Model inference
    proba = float(model.predict_proba(input_scaled)[0, 1])
    risk_score_pct = round(proba * 100.0, 2)

    # Categorization
    if proba >= 0.60:
        risk_status = "High Risk"
        risk_level = "High"
        color_code = "#EF4444"  # Vibrant Red
    elif proba >= 0.35:
        risk_status = "Moderate Risk"
        risk_level = "Moderate"
        color_code = "#F59E0B"  # Amber/Yellow
    else:
        risk_status = "Low Risk"
        risk_level = "Low"
        color_code = "#10B981"  # Emerald Green

    guidance, elevated_biomarkers = generate_clinical_guidance(patient, proba)

    best_metrics = all_metrics.get(best_model_name, {})

    return PredictionResponse(
        risk_score_percent=risk_score_pct,
        probability=round(proba, 4),
        status=risk_status,
        risk_level=risk_level,
        color_code=color_code,
        best_model_used=best_model_name,
        clinical_guidance=guidance,
        elevated_biomarkers=elevated_biomarkers,
        model_metrics=best_metrics
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
