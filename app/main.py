from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from explainability.shap_analysis import explain_prediction

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Thyroid Disease Prediction & Risk Assessment API",
    description="Machine learning based thyroid disease prediction system",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "models/thyroid_random_forest.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Original model feature names
# The model was trained using columns 0 to 20
# --------------------------------------------------

MODEL_FEATURES = [
    str(i)
    for i in range(21)
]


# --------------------------------------------------
# Disease labels
# --------------------------------------------------

CLASS_NAMES = {
    1: "Normal",
    2: "Hyperthyroidism",
    3: "Hypothyroidism"
}


# --------------------------------------------------
# Risk calculation
# --------------------------------------------------

def calculate_risk(confidence):

    if confidence >= 0.80:
        return "Low"

    elif confidence >= 0.50:
        return "Medium"

    else:
        return "High"


# --------------------------------------------------
# Patient input schema
# --------------------------------------------------

class PatientData(BaseModel):

    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "Thyroid Disease Prediction and Risk Assessment",
        "status": "API running",
        "model": "Random Forest"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(patient: PatientData):

    # Convert request to dictionary
    data = patient.model_dump()

    # Preserve the order of feature_1 to feature_21
    values = [
        data[f"feature_{i}"]
        for i in range(1, 22)
    ]

    # Create dataframe using the ORIGINAL
    # feature names expected by the trained model
    input_data = pd.DataFrame(
        [values],
        columns=MODEL_FEATURES
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    # --------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------

    probabilities = model.predict_proba(
        input_data
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    # --------------------------------------------------
    # Disease name
    # --------------------------------------------------

    disease = CLASS_NAMES.get(
        prediction,
        "Unknown"
    )

    # --------------------------------------------------
    # Risk level
    # --------------------------------------------------

    risk_level = calculate_risk(
        confidence
    )

    # --------------------------------------------------
    # Probability results
    # --------------------------------------------------

    probability_result = {}

    for class_value, probability in zip(
        model.classes_,
        probabilities
    ):

        probability_result[
            CLASS_NAMES.get(
                int(class_value),
                str(class_value)
            )
        ] = round(
            float(probability),
            4
        )

    # --------------------------------------------------
    # Final response
    # --------------------------------------------------

    return {
        "prediction": disease,
        "class": prediction,
        "confidence": round(
            confidence,
            4
        ),
        "risk_level": risk_level,
        "probabilities": probability_result
    }
# --------------------------------------------------
# SHAP explanation endpoint
# --------------------------------------------------

@app.post("/explain")
def explain(patient: PatientData):

    data = patient.model_dump()

    values = [
        data[f"feature_{i}"]
        for i in range(1, 22)
    ]

    result = explain_prediction(values)

    return result