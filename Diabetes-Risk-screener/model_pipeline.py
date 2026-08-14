"""
model_pipeline.py
=================
Training & Storage Layer for Diabetes Risk Screener.

Simulates structured clinical telemetry dataset, trains 3 models (Random Forest,
SVM, XGBoost Classifier), evaluates accuracy/recall/F1 metrics, selects the best model,
and serializes the artifact into `diabetes_model.joblib`.
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age"
]

TARGET_COLUMN = "diabetes_risk"


def generate_synthetic_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic clinical telemetry dataset based on
    clinical distributions and physiological risk correlations.
    """
    np.random.seed(random_state)

    # Telemetry feature distributions
    pregnancies = np.random.poisson(lam=3.5, size=n_samples)
    pregnancies = np.clip(pregnancies, 0, 17)

    age = np.random.gamma(shape=5, scale=8, size=n_samples).astype(int) + 18
    age = np.clip(age, 18, 88)

    # Glucose mg/dL (Normal ~70-100, Impaired ~100-125, Diabetic >126)
    glucose_base = np.random.normal(loc=115, scale=28, size=n_samples)
    glucose = glucose_base + (age - 35) * 0.35
    glucose = np.clip(glucose, 55, 240)

    # BMI kg/m^2 (Normal 18.5-24.9, Overweight 25-29.9, Obese >30)
    bmi = np.random.normal(loc=31.5, scale=6.8, size=n_samples)
    bmi = np.clip(bmi, 15.0, 58.0)

    # Blood Pressure mmHg (Diastolic)
    blood_pressure = np.random.normal(loc=72, scale=12, size=n_samples) + (bmi - 25) * 0.3
    blood_pressure = np.clip(blood_pressure, 45, 130)

    # Skin Thickness mm
    skin_thickness = np.random.normal(loc=24, scale=9, size=n_samples) + (bmi - 25) * 0.4
    skin_thickness = np.clip(skin_thickness, 5, 80)

    # Insulin mu U/ml
    insulin = np.random.exponential(scale=70, size=n_samples) + (glucose - 90) * 0.8
    insulin = np.clip(insulin, 10, 650)

    # Diabetes Pedigree Function (Genetic risk factor: 0.08 to 2.4)
    diabetes_pedigree = np.random.beta(a=2, b=5, size=n_samples) * 2.2 + 0.08
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.08, 2.45)

    # Non-linear probability logit for Diabetes Risk
    logit = (
        0.045 * (glucose - 100) +
        0.075 * (bmi - 25) +
        0.030 * (age - 30) +
        0.900 * (diabetes_pedigree - 0.4) +
        0.012 * (blood_pressure - 70) +
        0.005 * (insulin - 80) +
        0.080 * pregnancies - 2.8
    )

    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (prob > 0.48).astype(int)

    df = pd.DataFrame({
        "pregnancies": pregnancies,
        "glucose": np.round(glucose, 1),
        "blood_pressure": np.round(blood_pressure, 1),
        "skin_thickness": np.round(skin_thickness, 1),
        "insulin": np.round(insulin, 1),
        "bmi": np.round(bmi, 1),
        "diabetes_pedigree": np.round(diabetes_pedigree, 3),
        "age": age,
        TARGET_COLUMN: target
    })

    return df


def train_and_evaluate_models():
    """
    Executes the end-to-end ML model pipeline:
    1. Dataset creation & splitting
    2. Feature standardization
    3. Multi-model training (Random Forest, SVM, XGBoost)
    4. Evaluation metric comparison
    5. Best model serialization to joblib
    """
    print("=" * 60)
    print(" GlucoSense ML Pipeline - Training & Selection")
    print("=" * 60)

    # 1. Load / Generate Data
    df = generate_synthetic_dataset(n_samples=3000, random_state=42)
    print(f"[+] Dataset loaded: {df.shape[0]} patient records, {len(FEATURE_COLUMNS)} clinical features.")
    print(f"    Target distribution: {dict(df[TARGET_COLUMN].value_counts())}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Definitions
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            min_samples_split=4
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.5, gamma="scale", random_state=42),
            cv=5
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=42
        )
    }

    # 4. Evaluation Loop
    results = {}
    print("\n" + "-" * 60)
    print(f"{'Model Name':<25} | {'Accuracy':<9} | {'Recall':<9} | {'Precision':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 60)

    best_score = -1.0
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }

        print(f"{name:<25} | {acc:<9.4f} | {rec:<9.4f} | {prec:<9.4f} | {f1:<9.4f} | {auc:<9.4f}")

        # Selection criterion: primary metric F1-Score combined with ROC-AUC
        composite_score = 0.6 * f1 + 0.4 * auc
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name
            best_model_obj = model

    print("-" * 60)
    print(f"\n[*] Selected Best Model: {best_model_name} (Composite F1/AUC Score: {best_score:.4f})")

    # 5. Serialize Artifact
    artifact = {
        "model": best_model_obj,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "best_model_name": best_model_name,
        "metrics": results,
        "training_timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    }

    output_filename = "diabetes_model.joblib"
    joblib.dump(artifact, output_filename)
    print(f"[+] Model artifact successfully serialized to '{os.path.abspath(output_filename)}'.")

    return artifact


if __name__ == "__main__":
    train_and_evaluate_models()
