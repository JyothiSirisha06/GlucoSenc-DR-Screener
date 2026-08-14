# 🩺 GlucoSense: Diabetes Risk Screener

A production-ready, full-stack Machine Learning application for predicting patient diabetes risk based on physiological clinical telemetry. 

Built with **scikit-learn**, **XGBoost**, **FastAPI**, **Pydantic**, and **Streamlit**.

---

## 🏗️ Architecture & Project Structure

```
Diabetes-Risk-screener/
├── model_pipeline.py     # Training layer: Data simulation, multi-model evaluation & serialization
├── backend.py            # REST API layer: FastAPI server with Pydantic validation & model inference
├── frontend.py           # UI layer: Interactive Streamlit web application & clinical dashboard
├── diabetes_model.joblib # Serialized model bundle (Model, Scaler, Feature Order, Metrics)
├── requirements.txt      # Dependencies specification
└── README.md             # Documentation & running instructions
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
Make sure Python 3.9+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train Machine Learning Models
Execute `model_pipeline.py` to generate clinical data, train Random Forest, SVM, and XGBoost models, evaluate metrics, and save the best model artifact to `diabetes_model.joblib`:
```bash
python model_pipeline.py
```

### 3. Launch Backend API Server
Start the FastAPI server on port `8000`:
```bash
uvicorn backend:app --reload --port 8000
```
* **Swagger API Docs**: Explore interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 4. Launch Streamlit UI Dashboard
In a separate terminal tab, launch the Streamlit frontend app on port `8501`:
```bash
streamlit run frontend.py
```
* **Web UI Dashboard**: Access the interactive screener at [http://localhost:8501](http://localhost:8501)

---

## 📊 Machine Learning Pipeline Summary

`model_pipeline.py` trains three distinct binary classification algorithms using 5-fold metric evaluation:
1. **Support Vector Machine (SVM)** (RBF Kernel)
2. **XGBoost Classifier**
3. **Random Forest Classifier**

Each model is evaluated across **Accuracy**, **Recall**, **Precision**, **F1-Score**, and **ROC-AUC Score**. The pipeline automatically picks the top performer and serializes the complete bundle (`scaler`, `model`, `feature_names`, `metrics`) to disk via `joblib`.

### Telemetry Features Used:
* **Pregnancies**: Total number of pregnancies
* **Glucose**: Fasting / Random Plasma Glucose concentration (mg/dL)
* **Blood Pressure**: Diastolic blood pressure (mmHg)
* **Skin Thickness**: Triceps skin fold thickness (mm)
* **Insulin**: 2-Hour serum insulin (μU/mL)
* **BMI**: Body Mass Index (kg/m²)
* **Diabetes Pedigree**: Genetic likelihood function
* **Age**: Patient age in years

---

## 📡 API Endpoint Overview (`POST /predict`)

### Request Payload (`PatientData` Schema)
```json
{
  "pregnancies": 2,
  "glucose": 140.0,
  "blood_pressure": 82.0,
  "skin_thickness": 28.0,
  "insulin": 120.0,
  "bmi": 32.4,
  "diabetes_pedigree": 0.65,
  "age": 45
}
```

### Response Payload (`PredictionResponse` Schema)
```json
{
  "risk_score_percent": 88.5,
  "probability": 0.885,
  "status": "High Risk",
  "risk_level": "High",
  "color_code": "#EF4444",
  "best_model_used": "Support Vector Machine",
  "clinical_guidance": [
    "Schedule a formal diagnostic HbA1c test and Fasting Plasma Glucose test with a physician.",
    "Incorporate 150+ minutes of moderate aerobic exercise weekly to enhance insulin sensitivity.",
    "Inform your primary healthcare provider of your strong familial predisposition.",
    "Annual routine metabolic panel screening is recommended for individuals aged 45+."
  ],
  "elevated_biomarkers": [
    "Elevated Fasting Glucose: 140.0 mg/dL (Diabetic Range >= 126 mg/dL)",
    "Class I/II Obesity: BMI 32.4 kg/m² (Threshold >= 30.0)",
    "Strong Familial Risk Factor: Pedigree Index 0.65"
  ],
  "model_metrics": {
    "accuracy": 0.9717,
    "precision": 0.9819,
    "recall": 0.9673,
    "f1_score": 0.9745,
    "roc_auc": 0.9975
  }
}
```

---

## 🧪 Interactive Features in Streamlit UI

- **Clinical Input Sliders & Number Controls**: Live adjustment of metabolic and demographic telemetry.
- **Preset Profile Buttons**: Instant filling for **Low Risk**, **Moderate Risk**, and **High Risk** sample profiles for effortless testing.
- **Dynamic Risk Gauge & Badges**: Color-coded visual badges (#10B981 Green, #F59E0B Amber, #EF4444 Red).
- **Personalized Recommendations**: Context-aware clinical guidance based on specific biomarker anomalies.
- **JSON Telemetry Inspector**: Expandable technical view displaying full REST payloads and model evaluation metrics.
