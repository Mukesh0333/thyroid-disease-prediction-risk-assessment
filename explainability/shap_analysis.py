import joblib
import pandas as pd
import shap


MODEL_PATH = "models/thyroid_random_forest.joblib"

# Load model
model = joblib.load(MODEL_PATH)

# Model was trained with columns 0 to 20
FEATURE_NAMES = [
    str(i)
    for i in range(21)
]


def explain_prediction(values):

    # Create input dataframe using the
    # exact feature names expected by model
    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES
    )

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        input_data
    )

    # Get prediction
    prediction = int(
        model.predict(input_data)[0]
    )

    # --------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------

    if isinstance(shap_values, list):

        class_index = list(
            model.classes_
        ).index(prediction)

        values_for_class = shap_values[
            class_index
        ][0]

    else:

        values_array = shap_values

        if len(values_array.shape) == 3:

            class_index = list(
                model.classes_
            ).index(prediction)

            values_for_class = values_array[
                0,
                :,
                class_index
            ]

        else:

            values_for_class = values_array[0]

    # --------------------------------------------------
    # Create contribution list
    # --------------------------------------------------

    contributions = []

    for feature, value in zip(
        FEATURE_NAMES,
        values_for_class
    ):

        contributions.append(
            {
                "feature": feature,
                "contribution": round(
                    float(value),
                    6
                )
            }
        )

    # Sort by absolute contribution
    contributions.sort(
        key=lambda x: abs(
            x["contribution"]
        ),
        reverse=True
    )

    return {
        "prediction_class": prediction,
        "shap_values": contributions
    }