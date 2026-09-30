// ============================================================
// Thyroid Disease Prediction & Risk Assessment
// Frontend JavaScript
// ============================================================

const API_URL = "http://127.0.0.1:8000";

let lastPatientData = null;


// ============================================================
// DOM ELEMENTS
// ============================================================

const predictionForm = document.getElementById("predictionForm");

const loading = document.getElementById("loading");

const results = document.getElementById("results");

const errorBox = document.getElementById("errorBox");

const predictionElement = document.getElementById("prediction");

const confidenceElement = document.getElementById("confidence");

const riskElement = document.getElementById("riskLevel");

const probabilitiesElement =
    document.getElementById("probabilities");

const explainButton =
    document.getElementById("explainBtn");

const explainLoading =
    document.getElementById("explainLoading");

const shapResults =
    document.getElementById("shapResults");


// ============================================================
// GET INPUT DATA
// ============================================================

function getPatientData() {

    const patientData = {};

    for (let i = 1; i <= 21; i++) {

        const input =
            document.getElementById(`feature_${i}`);

        patientData[`feature_${i}`] =
            Number(input.value);
    }

    return patientData;
}


// ============================================================
// SHOW ERROR
// ============================================================

function showError(message) {

    errorBox.textContent = message;

    errorBox.classList.remove("hidden");

    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
    });
}


// ============================================================
// HIDE ERROR
// ============================================================

function hideError() {

    errorBox.classList.add("hidden");

    errorBox.textContent = "";
}


// ============================================================
// PREDICTION FORM
// ============================================================

predictionForm.addEventListener("submit", async function(event) {

    event.preventDefault();

    hideError();

    loading.classList.remove("hidden");

    results.classList.add("hidden");

    shapResults.innerHTML = "";

    try {

        // Collect patient values
        const patientData = getPatientData();

        // Save for explainability request
        lastPatientData = patientData;


        // ----------------------------------------------------
        // CALL /predict
        // ----------------------------------------------------

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(patientData)
            }
        );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                `Prediction failed (${response.status}): ${errorText}`
            );
        }


        const data =
            await response.json();


        console.log("Prediction response:", data);


        // ----------------------------------------------------
        // DISPLAY PREDICTION
        // ----------------------------------------------------

        predictionElement.textContent =
            data.prediction || "Unknown";


        // ----------------------------------------------------
        // DISPLAY CONFIDENCE
        // ----------------------------------------------------

        if (data.confidence !== undefined) {

            const confidence =
                Number(data.confidence) * 100;

            confidenceElement.textContent =
                `${confidence.toFixed(2)}%`;

        } else {

            confidenceElement.textContent =
                "--";
        }


        // ----------------------------------------------------
        // DISPLAY RISK
        // ----------------------------------------------------

        riskElement.textContent =
            data.risk_level || "Unknown";


        // ----------------------------------------------------
        // DISPLAY PROBABILITIES
        // ----------------------------------------------------

        displayProbabilities(
            data.probabilities || {}
        );


        // Show results
        results.classList.remove("hidden");


        // Scroll to results
        results.scrollIntoView({
            behavior: "smooth"
        });

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        showError(
            error.message ||
            "Unable to connect to prediction API."
        );
    }

    finally {

        loading.classList.add("hidden");
    }

});


// ============================================================
// DISPLAY PROBABILITIES
// ============================================================

function displayProbabilities(probabilities) {

    probabilitiesElement.innerHTML = "";

    const entries =
        Object.entries(probabilities);


    if (entries.length === 0) {

        probabilitiesElement.innerHTML =
            "<p>No probability data available.</p>";

        return;
    }


    entries.forEach(
        ([name, value]) => {

            const percentage =
                Number(value) * 100;


            const row =
                document.createElement("div");

            row.className =
                "probability-row";


            row.innerHTML = `

                <div class="probability-name">
                    ${name}
                </div>

                <div class="progress">

                    <div
                        class="progress-bar"
                        style="width: ${percentage}%">
                    </div>

                </div>

                <div class="probability-value">
                    ${percentage.toFixed(2)}%
                </div>

            `;


            probabilitiesElement.appendChild(row);

        }
    );
}


// ============================================================
// EXPLAINABILITY BUTTON
// ============================================================

explainButton.addEventListener(
    "click",
    async function() {

        hideError();

        if (!lastPatientData) {

            showError(
                "Please run a prediction first."
            );

            return;
        }


        explainLoading.classList.remove(
            "hidden"
        );

        shapResults.innerHTML = "";


        try {

            // ------------------------------------------------
            // CALL /explain
            // ------------------------------------------------

            const response = await fetch(
                `${API_URL}/explain`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify(
                        lastPatientData
                    )
                }
            );


            if (!response.ok) {

                const errorText =
                    await response.text();

                throw new Error(
                    `Explainability failed (${response.status}): ${errorText}`
                );
            }


            const data =
                await response.json();


            console.log(
                "SHAP response:",
                data
            );


            // ------------------------------------------------
            // DISPLAY SHAP
            // ------------------------------------------------

            displaySHAP(
                data.shap_values || []
            );

        }

        catch (error) {

            console.error(
                "Explainability error:",
                error
            );

            showError(
                error.message ||
                "Unable to calculate SHAP explanation."
            );
        }

        finally {

            explainLoading.classList.add(
                "hidden"
            );
        }

    }
);


// ============================================================
// DISPLAY SHAP VALUES
// ============================================================

function displaySHAP(shapValues) {

    shapResults.innerHTML = "";


    if (!shapValues ||
        shapValues.length === 0) {

        shapResults.innerHTML =
            "<p>No SHAP explanation available.</p>";

        return;
    }


    // Find largest absolute contribution
    const maxValue =
        Math.max(
            ...shapValues.map(
                item =>
                    Math.abs(
                        Number(
                            item.contribution
                        )
                    )
            )
        );


    shapValues.forEach(
        item => {

            const contribution =
                Number(
                    item.contribution
                );


            const absoluteValue =
                Math.abs(contribution);


            let width = 0;

            if (maxValue > 0) {

                width =
                    (absoluteValue /
                    maxValue) * 100;
            }


            const row =
                document.createElement("div");

            row.className =
                "shap-row";


            const barClass =
                contribution >= 0
                    ? "positive"
                    : "negative";


            const sign =
                contribution >= 0
                    ? "+"
                    : "";


            row.innerHTML = `

                <div class="shap-feature">
                    Feature ${item.feature}
                </div>

                <div class="shap-bar-area">

                    <div
                        class="shap-bar ${barClass}"
                        style="width: ${width}%">
                    </div>

                </div>

                <div class="shap-value">
                    ${sign}${contribution.toFixed(6)}
                </div>

            `;


            shapResults.appendChild(row);

        }
    );
}