import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load UCI thyroid training data
train = pd.read_csv(
    f"{RAW_PATH}/ann-train.data",
    sep=r"\s+",
    header=None
)

# Load UCI thyroid testing data
test = pd.read_csv(
    f"{RAW_PATH}/ann-test.data",
    sep=r"\s+",
    header=None
)

# Separate features and target
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Convert everything to numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Replace missing/infinite values
X_train = X_train.replace([float("inf"), float("-inf")], pd.NA)
X_test = X_test.replace([float("inf"), float("-inf")], pd.NA)

# Fill missing values using training medians
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Save processed datasets
X_train.to_csv(
    f"{PROCESSED_PATH}/X_train.csv",
    index=False
)

X_test.to_csv(
    f"{PROCESSED_PATH}/X_test.csv",
    index=False
)

y_train.to_csv(
    f"{PROCESSED_PATH}/y_train.csv",
    index=False,
    header=["target"]
)

y_test.to_csv(
    f"{PROCESSED_PATH}/y_test.csv",
    index=False,
    header=["target"]
)

print("Preprocessing completed successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
print("Processed files saved to data/processed/")