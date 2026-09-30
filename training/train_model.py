import pandas as pd
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = "data/processed"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

# --------------------------------------------------
# Load processed data
# --------------------------------------------------

X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")

y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv")["target"]
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv")["target"]

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

# --------------------------------------------------
# Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# Train model
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Save model
# --------------------------------------------------

model_file = f"{MODEL_PATH}/thyroid_random_forest.joblib"

joblib.dump(model, model_file)

print(f"\nModel saved to: {model_file}")

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = {
    "model": "Random Forest",
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "features": int(X_train.shape[1])
}

with open(
    f"{MODEL_PATH}/metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved to: models/metrics.json")

print("\nModel training pipeline completed successfully.")