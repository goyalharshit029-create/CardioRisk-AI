// Same-origin API: works locally through FastAPI and after Vercel deployment.
const API = window.location.origin;

function token() {
    return localStorage.getItem("token");
}

function patient() {
    try {
        return JSON.parse(localStorage.getItem("patient") || "null");
    } catch {
        return null;
    }
}


/* =========================
   TOAST / NOTIFICATIONS
========================= */

function toast(msg, type = "info") {
    const e = document.getElementById("toast");

    if (!e) {
        alert(msg);
        return;
    }

    e.className = "toast " + type;
    e.textContent = msg;
    e.style.display = "block";

    clearTimeout(window._toast);

    window._toast = setTimeout(() => {
        e.style.display = "none";
    }, 5000);
}


/* =========================
   AUTHENTICATION
========================= */

function authHeaders() {
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token()
    };
}

function requireAuth() {
    if (!token()) {
        toast("Please login first.", "error");

        setTimeout(() => {
            location.href = "login.html";
        }, 500);

        return false;
    }

    return true;
}

function logout() {
    // Capture user-specific keys before clearing login state.
    const chatKey = aiUserKey(AI_CHAT_STORAGE_KEY);
    const pendingKey = aiUserKey(AI_PENDING_STORAGE_KEY);
    localStorage.clear();
    sessionStorage.removeItem(chatKey);
    sessionStorage.removeItem(pendingKey);
    sessionStorage.removeItem("cardiorisk_ai_open");

    const widget = document.getElementById("aiAssistantWidget");
    if (widget) widget.remove();

    location.href = "login.html";
}

function nav() {
    const p = patient();
    const el = document.getElementById("patientName");

    if (el) {
        el.textContent = p ? `Hi, ${p.full_name}` : "";
    }
}


/* =========================
   HELPER
========================= */

function v(id) {
    return document.getElementById(id)?.value ?? "";
}


/* =========================
   REGISTER
========================= */

async function register() {

    const body = {
        full_name: v("name"),
        email: v("email"),
        phone: v("phone"),
        password: v("password")
    };

    if (
        body.full_name.length < 2 ||
        !body.email ||
        body.password.length < 6
    ) {
        toast("Enter valid registration details.", "error");
        return;
    }

    try {

        const r = await fetch(API + "/register", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body)
        });

        const d = await r.json();

        if (!r.ok) {
            throw Error(
                typeof d.detail === "string"
                    ? d.detail
                    : "Registration failed"
            );
        }

        localStorage.setItem("token", d.token);

        /*
         * Registration endpoint returns patient_id,
         * but not the complete patient object.
         * Store a basic object so navigation works.
         */
        localStorage.setItem(
            "patient",
            JSON.stringify({
                patient_id: d.patient_id,
                full_name: body.full_name,
                email: body.email
            })
        );

        toast(
            "Registration successful. You are logged in.",
            "ok"
        );

        setTimeout(() => {
            location.href = "assessment.html";
        }, 600);

    } catch (e) {

        toast(
            e.message || "Registration failed.",
            "error"
        );
    }
}


/* =========================
   LOGIN
========================= */

async function login() {

    const body = {
        email: v("email"),
        password: v("password")
    };

    try {

        const r = await fetch(API + "/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body)
        });

        const d = await r.json();

        if (!r.ok) {
            throw Error(
                typeof d.detail === "string"
                    ? d.detail
                    : "Login failed"
            );
        }

        localStorage.setItem("token", d.patient.token);

        localStorage.setItem(
            "patient",
            JSON.stringify(d.patient)
        );

        toast("Login successful.", "ok");

        setTimeout(() => {
            location.href = "assessment.html";
        }, 500);

    } catch (e) {

        toast(
            e.message || "Login failed.",
            "error"
        );
    }
}

async function analyzeECG() {

    const fileInput =
        document.getElementById("ecg_file");

    // ECG optional
    if (!fileInput || !fileInput.files.length) {

        return {
            status: "Not provided",
            prediction: null,
            abnormal_probability: null,
            normal_probability: null,
            analyzed: false
        };
    }

    const file = fileInput.files[0];

    const formData = new FormData();

    formData.append("file", file);

    const resultBox =
        document.getElementById("ecg_result");

    if (resultBox) {
        resultBox.textContent =
            "⏳ Analyzing ECG using CNN...";
    }

    try {

        const response = await fetch(
            API + "/ecg-predict",
            {
                method: "POST",

                headers: {
                    Authorization:
                        "Bearer " + token()
                },

                body: formData
            }
        );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "ECG analysis failed"
            );

        }

        const prediction = data.prediction;

        const abnormalProbability =
            Number(data.abnormal_probability);

        const normalProbability =
            Number(data.normal_probability);

        const isAbnormal =
            prediction === "Abnormal ECG";

        const status =
            isAbnormal
                ? "Abnormal"
                : "Normal";

        if (resultBox) {

            if (isAbnormal) {

                resultBox.textContent =
                    `⚠️ ECG analysis complete: Possible abnormal pattern detected (${(
                        abnormalProbability * 100
                    ).toFixed(1)}%).`;

            } else {

                resultBox.textContent =
                    `✅ ECG analysis complete: No abnormal pattern detected. Normal probability: ${(
                        normalProbability * 100
                    ).toFixed(1)}%.`;

            }

        }

        return {

            status:
                isAbnormal
                    ? "Abnormal"
                    : "Normal",

            prediction: prediction,

            abnormal_probability:
                abnormalProbability,

            normal_probability:
                normalProbability,

            analyzed: true

        };

    }

    catch (error) {

        console.error(
            "ECG analysis error:",
            error
        );

        if (resultBox) {

            resultBox.textContent =
                "⚠️ ECG analysis failed. Continuing without ECG data.";

        }

        return {

            status: "Not provided",

            prediction: null,

            abnormal_probability: null,

            normal_probability: null,

            analyzed: false

        };

    }

}
/* =========================
   ASSESSMENT
========================= */

async function analyze() {

    const btn = document.getElementById("analyzeBtn");

    /*
     * Read values
     */
    const age = Number(v("age"));
    const gender = Number(v("gender"));
    const height = Number(v("height"));
    const weight = Number(v("weight"));
    const ap_hi = Number(v("ap_hi"));
    const ap_lo = Number(v("ap_lo"));
    const cholesterol = Number(v("cholesterol"));
    const gluc = Number(v("gluc"));
    const smoke = Number(v("smoke"));
    const alco = Number(v("alco"));
    const active = Number(v("active"));


    /*
     * Frontend validation
     *
     * This prevents obviously invalid values
     * before sending them to FastAPI.
     */
    const checks = [

        [
            age,
            18,
            120,
            "Age must be between 18 and 120 years."
        ],

        [
            height,
            100,
            250,
            "Height must be between 100 and 250 cm."
        ],

        [
            weight,
            30,
            250,
            "Weight must be between 30 and 250 kg."
        ],

        [
            ap_hi,
            70,
            250,
            "Systolic BP must be between 70 and 250."
        ],

        [
            ap_lo,
            40,
            150,
            "Diastolic BP must be between 40 and 150."
        ],

        [
            gender,
            1,
            2,
            "Please select a valid gender."
        ],

        [
            cholesterol,
            1,
            3,
            "Please select a valid cholesterol level."
        ],

        [
            gluc,
            1,
            3,
            "Please select a valid glucose level."
        ],

        [
            smoke,
            0,
            1,
            "Please select a valid smoking value."
        ],

        [
            alco,
            0,
            1,
            "Please select a valid alcohol value."
        ],

        [
            active,
            0,
            1,
            "Please select a valid physical activity value."
        ]
    ];


    /*
     * Check ranges
     */
    for (const [value, min, max, message] of checks) {

        if (
            !Number.isFinite(value) ||
            value < min ||
            value > max
        ) {

            toast(message, "error");
            return;
        }
    }


    /*
     * Blood pressure relationship
     */
    if (ap_lo > ap_hi) {

        toast(
            "Diastolic BP cannot be greater than systolic BP.",
            "error"
        );

        return;
    }


    /*
     * Authentication
     */
    const authToken = token();

    if (!authToken) {

        toast(
            "Please login first.",
            "error"
        );

        return;
    }


    /*
     * Data sent to FastAPI
     */
    const family_history = Number(v("family_history"));
    const ecgAnalysis = await analyzeECG();

    const ecg_status = ecgAnalysis.status;

    if (!Number.isFinite(family_history) || family_history < 0 || family_history > 1) {
        toast("Please select a valid family history value.", "error");
        return;
    }

    const data = {
        age: age,
        gender: gender,
        height: height,
        weight: weight,
        ap_hi: ap_hi,
        ap_lo: ap_lo,
        cholesterol: cholesterol,
        gluc: gluc,
        smoke: smoke,
        alco: alco,
        active: active,
        family_history: family_history,

        ecg_status: ecg_status,

        ecg_prediction:
            ecgAnalysis.prediction,

        ecg_abnormal_probability:
            ecgAnalysis.abnormal_probability,

        ecg_normal_probability:
            ecgAnalysis.normal_probability,

        ecg_analyzed:
            ecgAnalysis.analyzed
    };


    /*
     * Loading state
     */
    if (btn) {
        btn.disabled = true;
        btn.textContent = "Analyzing...";
    }


    try {

        const response = await fetch(
            API + "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${authToken}`
                },

                body: JSON.stringify(data)
            }
        );


        /*
         * Read response
         */
        const result = await response.json();


        /*
         * Backend error
         */
        if (!response.ok) {

            /*
             * Pydantic validation errors
             */
            if (Array.isArray(result.detail)) {

                const messages = result.detail.map(error => {

                    const field =
                        error.loc &&
                            error.loc.length
                            ? error.loc[error.loc.length - 1]
                            : "Input";


                    /*
                     * Convert backend field names
                     * into user-friendly names.
                     */
                    const fieldNames = {

                        age: "Age",

                        gender: "Gender",

                        height: "Height",

                        weight: "Weight",

                        ap_hi: "Systolic BP",

                        ap_lo: "Diastolic BP",

                        cholesterol: "Cholesterol",

                        gluc: "Glucose",

                        smoke: "Smoking",

                        alco: "Alcohol",

                        active: "Physical Activity"
                    };


                    const friendlyField =
                        fieldNames[field] || field;


                    return `${friendlyField}: ${error.msg}`;
                });


                toast(
                    messages.join(" | "),
                    "error"
                );
            }


            /*
             * Normal FastAPI HTTPException
             */
            else if (
                typeof result.detail === "string"
            ) {

                toast(
                    result.detail,
                    "error"
                );
            }


            /*
             * Unknown error format
             */
            else {

                toast(
                    "Invalid input. Please check your values.",
                    "error"
                );
            }


            return;
        }


        /*
         * Successful prediction
         *
         * IMPORTANT:
         * Use "lastResult" because results page
         * reads "lastResult".
         */
        const completeResult = {

            ...result,

            ecg_status:
                ecgAnalysis.status,

            ecg_prediction:
                ecgAnalysis.prediction,

            ecg_abnormal_probability:
                ecgAnalysis.abnormal_probability,

            ecg_normal_probability:
                ecgAnalysis.normal_probability,

            ecg_analyzed:
                ecgAnalysis.analyzed

        };


        localStorage.setItem(
            "lastResult",
            JSON.stringify(completeResult)
        );


        localStorage.setItem(
            "latestResult",
            JSON.stringify(completeResult)
        );


        /* Latest entered assessment is used by the dynamic Guidance page. */
        localStorage.setItem(
            "lastAssessment",
            JSON.stringify(data)
        );


        /*
         * Go to Results page
         */
        window.location.href = "results.html";

    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        toast(
            "Unable to connect to the backend.",
            "error"
        );

    } finally {

        if (btn) {
            btn.disabled = false;
            btn.textContent =
                "Analyze Cardiovascular Risk";
        }
    }
}

/* =========================
   SHAP EXPLAINABILITY
========================= */

/* =========================
   SHAP EXPLAINABILITY
========================= */

async function loadShapExplanation(assessment) {

    const container =
        document.getElementById("shapExplanation");


    if (!container) {
        return;
    }


    // Agar assessment available nahi hai
    if (!assessment) {

        container.innerHTML = `

            <div class="shap-empty">

                <h3>
                    Explainability Unavailable
                </h3>

                <p>
                    Assessment data is not available for explanation.
                </p>

            </div>

        `;

        return;
    }


    container.innerHTML = `

        <div class="shap-loading">

            <div class="shap-spinner"></div>

            <p>
                Analyzing the factors behind this prediction...
            </p>

        </div>

    `;


    try {

        const response = await fetch(

            API + "/predict-explanation",

            {

                method: "POST",

                headers: {

                    "Content-Type": "application/json",

                    "Authorization":
                        "Bearer " + token()

                },


                // IMPORTANT:
                // Selected assessment is sent here
                body: JSON.stringify(assessment)

            }

        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(

                typeof data.detail === "string"

                    ? data.detail

                    : "Unable to generate prediction explanation"

            );

        }


        const factors =
            Array.isArray(data.shap_explanation)

                ? data.shap_explanation

                : [];


        if (!factors.length) {

            container.innerHTML = `

                <div class="shap-empty">

                    <div class="shap-empty-icon">
                        🧠
                    </div>

                    <h3>
                        Explainability Unavailable
                    </h3>

                    <p>
                        No detailed factor explanation is available
                        for this assessment.
                    </p>

                </div>

            `;

            return;

        }


        /* =========================
           FEATURE LABELS
        ========================= */

        const featureLabels = {

            age: "Age",
            gender: "Gender",
            height: "Height",
            weight: "Body Weight",

            bmi: "Body Mass Index (BMI)",
            bmi_squared: "BMI-Related Risk",
            underweight: "Low Body Weight",
            overweight: "Overweight",
            obese: "Obesity",
            severely_obese: "Severe Obesity",

            ap_hi: "Systolic Blood Pressure",
            ap_lo: "Diastolic Blood Pressure",

            high_bp: "High Blood Pressure",
            very_high_bp: "Severely High Blood Pressure",
            normal_bp: "Normal Blood Pressure",

            pulse_pressure: "Pulse Pressure",
            mean_arterial_pressure: "Overall Blood Pressure",
            bp_ratio: "Blood Pressure Balance",
            bp_sum: "Overall Blood Pressure",
            bp_squared: "Blood Pressure Severity",

            cholesterol: "Cholesterol Level",
            high_cholesterol: "High Cholesterol",
            very_high_cholesterol: "Very High Cholesterol",

            cholesterol_age:
                "Combined Effect of Cholesterol and Age",

            gluc: "Blood Glucose Level",
            high_glucose: "High Blood Glucose",
            very_high_glucose: "Very High Blood Glucose",

            glucose_bmi:
                "Combined Effect of Glucose and BMI",

            smoke: "Smoking",
            alco: "Alcohol Consumption",
            active: "Physical Activity",
            inactive: "Physical Inactivity",

            lifestyle_risk_score:
                "Combined Lifestyle Risk Factors",

            age_years: "Age",
            age_squared: "Age-Related Risk",
            age_cubed: "Age-Related Risk",

            age_over_40: "Age Above 40",
            age_over_50: "Age Above 50",
            age_over_60: "Age Above 60",

            age_bp_interaction:
                "Combined Effect of Age and Blood Pressure",

            age_diastolic_interaction:
                "Combined Effect of Age and Diastolic Pressure",

            age_bmi_interaction:
                "Combined Effect of Age and BMI",

            bmi_bp_interaction:
                "Combined Effect of BMI and Blood Pressure",

            clinical_risk_score:
                "Overall Clinical Risk Factors"

        };


        /* =========================
           CREATE SHAP CARDS
        ========================= */

        const factorsHTML = factors.map(item => {


            const shapValue =
                Number(item.shap_value || 0);


            const increasesRisk =
                shapValue > 0;


            const direction =
                increasesRisk

                    ? "Increases Model Risk Estimate"

                    : "Decreases Model Risk Estimate";


            const directionClass =
                increasesRisk

                    ? "increase"

                    : "decrease";


            const featureName =

                featureLabels[item.feature]

                ||

                String(item.feature || "Unknown Factor")

                    .replace(/_/g, " ")

                    .replace(

                        /\b\w/g,

                        char => char.toUpperCase()

                    );


            const impact =
                Math.abs(shapValue);


            return `

                <div class="shap-factor-card ${directionClass}">


                    <div class="shap-factor-top">


                        <div class="shap-factor-icon">

                            ${increasesRisk ? "↑" : "↓"}

                        </div>


                        <div class="shap-factor-info">


                            <h3>

                                ${escapeHtml(featureName)}

                            </h3>


                            <span
                                class="shap-direction ${directionClass}"
                            >

                                ${direction}

                            </span>


                        </div>


                    </div>



                    <div class="shap-impact">


                        <span>
                            Model Impact
                        </span>


                        <strong>

                            ${impact.toFixed(4)}

                        </strong>


                    </div>



                    <div class="shap-impact-bar">


                        <div

                            class="
                                shap-impact-fill
                                ${directionClass}
                            "

                            style="
                                width:
                                ${Math.min(
                impact * 100,
                100
            )}%
                            "

                        ></div>


                    </div>


                </div>

            `;


        }).join("");


        /* =========================
           RENDER
        ========================= */

        container.innerHTML = `


            <div class="shap-method-badge">

                ✨ SHAP Explainability

            </div>


            <div class="shap-factors-grid">

                ${factorsHTML}

            </div>


            <div class="shap-note">

                <strong>
                    How to interpret this:
                </strong>

                A factor marked as increasing or decreasing risk
                shows how it influenced the AI model's prediction
                for this specific assessment.

            </div>


        `;


    }

    catch (error) {


        console.error(
            "SHAP explanation error:",
            error
        );


        container.innerHTML = `

            <div class="shap-empty">


                <div class="shap-empty-icon">
                    ⚠️
                </div>


                <h3>
                    Explainability Unavailable
                </h3>


                <p>

                    ${escapeHtml(

            error.message ||

            "Unable to generate the prediction explanation."

        )}

                </p>


            </div>

        `;

    }

}


function formatFeatureName(feature) {

    const names = {

        age: "Age",

        gender: "Gender",

        height: "Height",

        weight: "Weight",

        ap_hi: "Systolic Blood Pressure",

        ap_lo: "Diastolic Blood Pressure",

        cholesterol: "Cholesterol",

        gluc: "Glucose",

        smoke: "Smoking",

        alco: "Alcohol Consumption",

        active: "Physical Activity",

        bmi: "Body Mass Index",

        pulse_pressure: "Pulse Pressure",

        mean_arterial_pressure:
            "Mean Arterial Pressure",

        bp_ratio:
            "Blood Pressure Ratio",

        family_history:
            "Family History"

    };

    return names[feature] ||
        feature
            .replace(/_/g, " ")
            .replace(/\b\w/g, char =>
                char.toUpperCase()
            );

}


/* =========================
   RESULTS
========================= */

function renderResult() {

    const r = JSON.parse(
        localStorage.getItem("lastResult") || "null"
    );

    const box = document.getElementById("result");

    if (!box) return;


    /* =========================
       NO RESULT
    ========================= */

    if (!r) {

        box.innerHTML = `
            <div class="card empty-result">

                <div class="empty-icon">📋</div>

                <h2>No Assessment Result Yet</h2>

                <p>
                    Complete a cardiovascular assessment to view
                    your personalized cardiovascular risk analysis.
                </p>

                <a class="btn" href="assessment.html">
                    Start Assessment
                </a>

            </div>
        `;

        return;
    }


    /* =========================
       RISK CLASS
    ========================= */

    const level = (r.risk_level || "").toUpperCase();

    const cls =
        level === "HIGH"
            ? "high"
            : level === "MODERATE" || level === "MEDIUM"
                ? "medium"
                : "low";


    /* =========================
       RISK ICON
    ========================= */

    const riskIcon =
        cls === "high"
            ? "⚠️"
            : cls === "medium"
                ? "⚡"
                : "💚";


    /* =========================
       CONTRIBUTING FACTORS
    ========================= */

    const factors = Array.isArray(r.contributing_factors)
        ? r.contributing_factors
        : [];


    const factorsHTML = factors.length > 0

        ? factors.map(factor => `

            <div class="factor-card">

                <div class="factor-icon">
                    ✓
                </div>

                <div class="factor-text">
                    ${factor}
                </div>

            </div>

        `).join("")

        : `

            <div class="no-data">

                <div class="no-data-icon">
                    💚
                </div>

                <p>
                    No major contributing factors were identified
                    from the available assessment information.
                </p>

            </div>

        `;


    /* =========================
       ECG ANALYSIS
    ========================= */

    let ecgHTML = "";

    if (r.ecg_analyzed) {

        const normalProbability =
            r.ecg_normal_probability !== undefined &&
                r.ecg_normal_probability !== null
                ? (Number(r.ecg_normal_probability) * 100).toFixed(1)
                : "—";


        const abnormalProbability =
            r.ecg_abnormal_probability !== undefined &&
                r.ecg_abnormal_probability !== null
                ? (Number(r.ecg_abnormal_probability) * 100).toFixed(1)
                : "—";


        ecgHTML = `

            <section class="result-section ecg-analysis-section">

                <div class="section-heading">

                    <div>

                        <p class="section-small-title">
                            AI ECG ANALYSIS
                        </p>

                        <h2>
                            🫀 ECG Analysis
                        </h2>

                    </div>

                </div>


                <p class="section-description">

                    Your uploaded ECG data was analyzed using
                    the AI ECG prediction model.

                </p>


                <div class="result-summary-grid">


                    <div class="summary-card">

                        <div class="summary-icon">
                            🫀
                        </div>

                        <div>

                            <div class="summary-label">
                                ECG Result
                            </div>

                            <div class="summary-value">
                                ${r.ecg_status || "—"}
                            </div>

                        </div>

                    </div>


                    <div class="summary-card">

                        <div class="summary-icon">
                            📈
                        </div>

                        <div>

                            <div class="summary-label">
                                Normal Probability
                            </div>

                            <div class="summary-value">
                                ${normalProbability}%
                            </div>

                        </div>

                    </div>


                    <div class="summary-card">

                        <div class="summary-icon">
                            ⚠️
                        </div>

                        <div>

                            <div class="summary-label">
                                Abnormal Probability
                            </div>

                            <div class="summary-value">
                                ${abnormalProbability}%
                            </div>

                        </div>

                    </div>


                </div>


                <div class="medical-disclaimer">

                    <span>ℹ️</span>

                    <p>
                        ECG analysis is generated by an AI model and
                        should not be considered a medical diagnosis.
                    </p>

                </div>

            </section>

        `;
    }


    /* =========================
       RENDER RESULT PAGE
    ========================= */

    box.innerHTML = `


        <!-- HERO -->

        <section class="result-hero ${cls}">


            <div class="model-badge">

                🤖 ML PREDICTION

                ${r.model ? `• ${r.model}` : ""}

            </div>


            <div class="result-hero-content">


                <div class="result-main-info">

                    <p class="result-label">
                        CARDIOVASCULAR RISK ASSESSMENT
                    </p>


                    <h1>
                        Your Assessment Result
                    </h1>


                    <p class="result-description">

                        Your cardiovascular risk has been calculated
                        using the health information provided during
                        this assessment.

                    </p>

                </div>


                <div class="risk-score-box">


                    <div class="risk-icon">
                        ${riskIcon}
                    </div>


                    <div class="risk-score">
                        ${Number(r.risk_probability || 0).toFixed(2)}%
                    </div>


                    <div class="risk-status ${cls}">
                        ${level || "UNKNOWN"} RISK
                    </div>


                </div>


            </div>


        </section>



        <!-- SUMMARY -->

        <section class="result-summary-grid">


            <div class="summary-card">

                <div class="summary-icon">
                    📊
                </div>

                <div>

                    <div class="summary-label">
                        Risk Level
                    </div>

                    <div class="summary-value ${cls}">
                        ${level || "—"}
                    </div>

                </div>

            </div>



            <div class="summary-card">

                <div class="summary-icon">
                    🎯
                </div>

                <div>

                    <div class="summary-label">
                        Risk Probability
                    </div>

                    <div class="summary-value">
                        ${Number(r.risk_probability || 0).toFixed(2)}%
                    </div>

                </div>

            </div>



            <div class="summary-card">

                <div class="summary-icon">
                    📅
                </div>

                <div>

                    <div class="summary-label">
                        Assessment Date
                    </div>

                    <div class="summary-value date-value">

                        ${r.created_at
            ? new Date(r.created_at).toLocaleString()
            : "Latest Assessment"}

                    </div>

                </div>

            </div>


        </section>



        <!-- ECG -->

        ${ecgHTML}



        <!-- CONTRIBUTING FACTORS -->

        <section class="result-section">


            <div class="section-heading">

                <div>

                    <p class="section-small-title">
                        PERSONALIZED ANALYSIS
                    </p>

                    <h2>
                        🔍 Contributing Factors
                    </h2>

                </div>

            </div>


            <p class="section-description">

                These factors are based specifically on the
                information from this assessment.

            </p>


            <div class="factors-grid">

                ${factorsHTML}

            </div>


        </section>



        <!-- SHAP EXPLANATION -->

        <section class="result-section">

            <div class="section-heading">

                <div>

                    <p class="section-small-title">
                        AI EXPLAINABILITY
                    </p>

                    <h2>
                        🧠 Why did the AI give this result?
                    </h2>

                </div>

            </div>


            <p class="section-description">

                The following explanation shows which factors had
                the greatest influence on this specific prediction.

            </p>


            <div id="shapExplanation"></div>


        </section>



        <!-- DOCTOR ESCALATION -->

        ${r.doctor_escalation ? `

            <section class="doctor-escalation ${cls}">


                <div class="doctor-icon">
                    🩺
                </div>


                <div>

                    <h3>
                        Professional Medical Follow-up Recommended
                    </h3>


                    <p>
                        ${r.doctor_escalation}
                    </p>

                </div>


            </section>

        ` : ""}



        <!-- DISCLAIMER -->

        ${r.disclaimer ? `

            <div class="medical-disclaimer">

                <span>ℹ️</span>

                <p>

                    <strong>Medical Disclaimer:</strong>

                    ${r.disclaimer}

                </p>

            </div>

        ` : ""}



        <!-- ACTIONS -->

        <div class="result-actions">


            <a class="btn" href="assessment.html">

                🔄 New Assessment

            </a>


            <a class="btn secondary"
               href="recommendations.html">

                📖 View Personalized Guidance

            </a>


        </div>


    `;


    /* =========================
       LOAD SHAP FOR THIS RESULT
    ========================= */


    const assessment = JSON.parse(
        localStorage.getItem("lastAssessment") || "null"
    );

    loadShapExplanation(assessment);

}

/* =========================
   HISTORY
========================= */

async function loadHistory() {

    if (!requireAuth()) {
        return;
    }

    try {

        const r = await fetch(
            API + "/assessments",
            {
                headers: {
                    "Authorization": "Bearer " + token()
                }
            }
        );

        const d = await r.json();

        if (!r.ok) {
            throw Error(
                typeof d.detail === "string"
                    ? d.detail
                    : "Could not load history"
            );
        }

        const assessments = Array.isArray(d.assessments)
            ? d.assessments
            : [];

        const rows = assessments.map(a => {

            const risk = Number(a.risk_probability).toFixed(2);

            const level = a.risk_level || "UNKNOWN";

            const cls =
                level === "HIGH"
                    ? "high"
                    : level === "MODERATE" || level === "MEDIUM"
                        ? "medium"
                        : "low";

            return `
                <tr>

                    <td class="history-date">
                        ${new Date(a.created_at).toLocaleString()}
                    </td>

                    <td>
                        <span class="history-risk ${cls}">
                            <strong>${risk}%</strong>
                            <span>${level}</span>
                        </span>
                    </td>

                </tr>
            `;

        }).join("");

        const history = document.getElementById("history");

        if (history) {

            history.innerHTML = rows || `
                <tr>
                    <td colspan="2" class="no-history">
                        📋 No assessments yet.
                    </td>
                </tr>
            `;
        }

    } catch (e) {

        toast(
            e.message || "Could not load history.",
            "error"
        );
    }
}

/* =========================
   AI ASSISTANT
========================= */

async function askAssistant() {

    if (!requireAuth()) {
        return;
    }


    const q = v("question");


    if (q.trim().length < 2) {

        toast(
            "Ask a question first.",
            "error"
        );

        return;
    }


    const out =
        document.getElementById("answer");


    if (out) {
        out.innerHTML = "Thinking...";
    }


    try {

        const r = await fetch(
            API + "/assistant",
            {
                method: "POST",
                headers: authHeaders(),

                body: JSON.stringify({
                    question: q
                })
            }
        );


        const d = await r.json();


        if (!r.ok) {

            throw Error(
                typeof d.detail === "string"
                    ? d.detail
                    : "Assistant unavailable"
            );
        }


        if (out) {

            out.innerHTML = `
                <div class="notice info">
                    ${d.answer}
                </div>
            `;
        }

    } catch (e) {

        if (out) {
            out.innerHTML = "";
        }

        toast(
            e.message ||
            "Assistant unavailable.",
            "error"
        );
    }
}


/* =========================
   MODEL METRICS
========================= */

function loadMetrics() {

    fetch(API + "/metrics")

        .then(r => r.json())

        .then(m => {

            const e =
                id =>
                    document.getElementById(id);


            if (e("metricModel")) {
                e("metricModel").textContent =
                    m.selected_model;
            }


            if (e("metricAcc")) {
                e("metricAcc").textContent =
                    (m.accuracy * 100)
                        .toFixed(2) + "%";
            }


            if (e("metricPrec")) {
                e("metricPrec").textContent =
                    (m.precision * 100)
                        .toFixed(2) + "%";
            }


            if (e("metricRec")) {
                e("metricRec").textContent =
                    (m.recall * 100)
                        .toFixed(2) + "%";
            }


            if (e("metricF1")) {
                e("metricF1").textContent =
                    (m.f1 * 100)
                        .toFixed(2) + "%";
            }


            if (e("metricAuc")) {
                e("metricAuc").textContent =
                    (m.roc_auc * 100)
                        .toFixed(2) + "%";
            }


            if (e("metricRows")) {
                e("metricRows").textContent =
                    m.dataset_records;
            }

        })

        .catch(() => {
            // Metrics are optional on pages
            // that do not display them.
        });
}




/* =========================
   GEMINI PERSONALIZED GUIDANCE
========================= */

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function guidanceItems(items, type, icon) {
    if (!Array.isArray(items) || !items.length) {
        return '<div class="guidance-empty-small">No additional actions are needed in this category right now.</div>';
    }

    return items.map((item, index) => `
        <article class="guidance-action-card ${type}">
            <div class="guidance-action-icon">${icon}</div>
            <div class="guidance-action-copy">
                <div class="guidance-step">${String(index + 1).padStart(2, "0")}</div>
                <h3>${escapeHtml(item.title)}</h3>
                <p>${escapeHtml(item.description)}</p>
            </div>
        </article>
    `).join("");
}

async function renderGuidance() {
    const box = document.getElementById("guidanceContent");
    if (!box) return;

    const result = JSON.parse(localStorage.getItem("lastResult") || "null");
    const assessment = JSON.parse(localStorage.getItem("lastAssessment") || "null");

    if (!result || !assessment) {
        box.innerHTML = `
            <section class="guidance-empty card">
                <div class="empty-icon">📋</div>
                <h2>No assessment available</h2>
                <p>Complete a cardiovascular assessment first to generate personalized guidance.</p>
                <a class="btn" href="assessment.html">Start Assessment</a>
            </section>`;
        return;
    }

    try {
        const response = await fetch(API + "/guidance", {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify({
                assessment,
                risk_probability: Number(result.risk_probability),
                risk_level: result.risk_level,
                bmi: Number(result.bmi),
                contributing_factors: Array.isArray(result.contributing_factors)
                    ? result.contributing_factors : []
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(typeof data.detail === "string" ? data.detail : "Guidance could not be generated");
        }

        const isGemini = data.source === "Gemini AI";
        const followUp = data.medical_follow_up || {};

        box.innerHTML = `
            <section class="guidance-overview">
                <div>
                    <span class="overview-label">LATEST ASSESSMENT</span>
                    <h2>${escapeHtml(result.risk_level)} risk • ${Number(result.risk_probability).toFixed(2)}%</h2>
                    <p>Your action plan is organized by priority so you can focus on the most relevant next steps.</p>
                </div>
                <div class="ai-source ${isGemini ? "gemini" : "fallback"}">
                    ${isGemini ? "✦ Gemini AI guidance" : "✦ Personalized guidance"}
                </div>
            </section>

            <section class="guidance-section-new priority-section">
                <div class="guidance-section-header">
                    <div class="section-icon-new">🎯</div>
                    <div><span class="section-kicker">START HERE</span><h2>Priority Actions</h2><p>Focus on these important next steps first.</p></div>
                </div>
                <div class="guidance-grid">${guidanceItems(data.priority_actions, "priority", "!")}</div>
            </section>

            <section class="guidance-section-new lifestyle-section">
                <div class="guidance-section-header">
                    <div class="section-icon-new">❤️</div>
                    <div><span class="section-kicker">LONG-TERM HABITS</span><h2>Lifestyle & Risk Reduction</h2><p>Practical habits selected for your current assessment.</p></div>
                </div>
                <div class="guidance-grid">${guidanceItems(data.lifestyle_risk_reduction, "lifestyle", "✓")}</div>
            </section>

            <section class="guidance-section-new prevention-section">
                <div class="guidance-section-header">
                    <div class="section-icon-new">🛡️</div>
                    <div><span class="section-kicker">STAY AWARE</span><h2>Prevention & Monitoring</h2><p>Keep track of the factors that matter for your cardiovascular health.</p></div>
                </div>
                <div class="guidance-grid">${guidanceItems(data.prevention_monitoring, "prevention", "→")}</div>
            </section>

            <section class="medical-followup-new ${followUp.recommended ? "recommended" : "routine"}">
                <div class="medical-followup-icon">🩺</div>
                <div>
                    <span class="section-kicker">MEDICAL FOLLOW-UP</span>
                    <h2>${followUp.recommended ? "Professional follow-up recommended" : "Continue routine preventive care"}</h2>
                    <p>${escapeHtml(followUp.message || "Use this assessment as educational decision-support information.")}</p>
                </div>
            </section>

            <div class="guidance-disclaimer-new"><strong>Important:</strong> CardioRisk AI is an educational decision-support prototype. It does not provide a medical diagnosis or replace professional medical advice.</div>

            <div class="guidance-buttons-new">
                <a href="results.html" class="btn secondary">← Back to Results</a>
                <a href="assessment.html" class="btn">↻ New Assessment</a>
            </div>
        `;
    } catch (error) {
        box.innerHTML = `
            <section class="guidance-empty card">
                <div class="empty-icon">⚠️</div>
                <h2>Guidance unavailable</h2>
                <p>${escapeHtml(error.message || "Unable to generate personalized guidance.")}</p>
                <button class="btn" onclick="renderGuidance()">Try Again</button>
            </section>`;
    }
}


/* =========================
   PAGE INITIALIZATION
========================= */

nav();
renderGuidance();


/* =========================
   GLOBAL FOOTER
========================= */

function createFooter() {

    // Prevent creating the footer twice
    if (document.querySelector(".site-footer")) {
        return;
    }

    const footer = document.createElement("footer");

    footer.className = "site-footer";

    footer.innerHTML = `
        <div class="footer-container">

            <div class="footer-brand">

                <div class="footer-logo">
                    ❤️ <span>CardioRisk AI</span>
                </div>

                <p>
                    AI-powered cardiovascular risk assessment and
                    personalized health guidance.
                </p>

            </div>


            <div class="footer-links">

                <span class="footer-heading">Quick Links</span>

                <a href="index.html">Dashboard</a>
                <a href="assessment.html">Assessment</a>
                <a href="results.html">Results</a>
                <a href="recommendations.html">Guidance</a>
                <a href="about.html">About</a>

            </div>


            <div class="footer-contact">

                <span>Connect with us</span>

                <div class="social-icons">

                    <a href="#" class="social instagram" title="Instagram">
                        <i class="fa-brands fa-instagram"></i>
                    </a>

                    <a href="mailto:cardioriskai@example.com"
                       class="social email"
                       title="Email">
                        <i class="fa-solid fa-envelope"></i>
                    </a>

                    <a href="#" class="social linkedin" title="LinkedIn">
                        <i class="fa-brands fa-linkedin-in"></i>
                    </a>

                </div>

            </div>

        </div>


        <div class="footer-bottom">

            <span>
                © ${new Date().getFullYear()} CardioRisk AI
            </span>

            <span class="footer-divider">•</span>

            <span>
                ML Healthcare Prototype
            </span>

            <span class="footer-divider">•</span>

            <span>
                Not a medical diagnosis
            </span>

        </div>
    `;

    document.body.appendChild(footer);
}


/* Create footer */
createFooter();

/* =========================================================
   FLOATING AI ASSISTANT
========================================================= */

/*
    IMPORTANT:

    We use sessionStorage instead of localStorage.

    sessionStorage:
    ✔ Chat remains while website/browser session is open
    ✔ Chat remains if you close and reopen the chat window
    ✔ Chat disappears when the browser/app session is closed
*/

const AI_CHAT_STORAGE_KEY = "cardiorisk_ai_chat";
const AI_PENDING_STORAGE_KEY = "cardiorisk_ai_pending";

function aiUserKey(base) {
    const p = patient();
    return p && p.patient_id ? `${base}_${p.patient_id}` : base;
}


/* =========================================================
   CREATE ASSISTANT
========================================================= */

function createFloatingAssistant() {

    // Prevent duplicate assistant
    if (document.getElementById("aiAssistantWidget")) {
        return;
    }


    const widget = document.createElement("div");

    widget.id = "aiAssistantWidget";


    widget.innerHTML = `

        <!-- ================= CHAT WINDOW ================= -->

        <div
            class="ai-chat-window"
            id="aiChatWindow"
        >

            <!-- HEADER -->

            <div class="ai-chat-header">

                <div class="ai-header-info">

                    <div class="ai-avatar">
                        🤖
                    </div>


                    <div class="ai-header-text">

                        <strong>
                            CardioRisk AI Assistant
                        </strong>


                        <span>

                            <span class="online-dot"></span>

                            Online • Health education

                        </span>

                    </div>

                </div>


                <!-- CLOSE BUTTON -->

                <button
                    class="ai-close-btn"
                    onclick="closeAIChat()"
                    title="Close chat"
                >
                    ×
                </button>

            </div>



            <!-- ================= MESSAGES ================= -->

            <div
                class="ai-chat-messages"
                id="aiChatMessages"
            >
            </div>



            <!-- ================= INPUT ================= -->

            <div class="ai-chat-input-area">

                <input
                    type="text"
                    id="floatingAIQuestion"
                    placeholder="Ask about your health assessment..."
                    onkeydown="handleAIEnter(event)"
                    autocomplete="off"
                >


                <button
                    class="ai-send-btn"
                    type="button"
                    onclick="askFloatingAssistant()"
                    title="Send message"
                >
                    ➤
                </button>

            </div>



            <!-- DISCLAIMER -->

            <div class="ai-chat-disclaimer">

                Educational support only • Not medical advice

            </div>

        </div>



        <!-- ================= FLOATING BUTTON ================= -->

        <button
            class="floating-ai-button"
            id="floatingAIButton"
            onclick="toggleAIChat()"
            title="Ask CardioRisk AI"
        >

            <span class="floating-ai-icon">
                🤖
            </span>


            <span class="floating-ai-pulse">
            </span>

        </button>

    `;


    document.body.appendChild(widget);


    /*
        Load previous messages.

        This is why messages remain when you close
        and reopen the chat.
    */
    loadAIChat();


    /*
        Scroll to bottom after loading
    */
    setTimeout(() => {

        scrollAIChatToBottom();

    }, 100);
}



/* =========================================================
   OPEN / CLOSE CHAT
========================================================= */

function toggleAIChat() {

    const chat =
        document.getElementById("aiChatWindow");

    const button =
        document.getElementById("floatingAIButton");


    if (!chat) {
        return;
    }


    /*
        Toggle chat
    */

    chat.classList.toggle("show");


    /*
        Change floating button state
    */

    if (button) {

        button.classList.toggle(
            "chat-open",
            chat.classList.contains("show")
        );

    }


    /*
        Focus input when chat opens
    */

    if (chat.classList.contains("show")) {

        setTimeout(() => {

            const input =
                document.getElementById(
                    "floatingAIQuestion"
                );

            if (input) {

                input.focus();

            }

        }, 200);

    }


    /*
        Save whether chat is open
    */

    sessionStorage.setItem(
        "cardiorisk_ai_open",
        chat.classList.contains("show")
            ? "true"
            : "false"
    );
}

function openAssistant() {

    // Agar widget kisi reason se create nahi hua ho
    if (!document.getElementById("aiAssistantWidget")) {
        createFloatingAssistant();
    }

    const chat = document.getElementById("aiChatWindow");
    const button = document.getElementById("floatingAIButton");

    if (!chat) return;

    // Always OPEN chat
    chat.classList.add("show");

    if (button) {
        button.classList.add("chat-open");
    }

    // Remember open state
    sessionStorage.setItem("cardiorisk_ai_open", "true");

    // Focus input
    setTimeout(() => {
        const input = document.getElementById("floatingAIQuestion");

        if (input) {
            input.focus();
        }
    }, 150);
}



/* =========================================================
   CLOSE CHAT
========================================================= */

function closeAIChat() {

    const chat =
        document.getElementById("aiChatWindow");

    const button =
        document.getElementById("floatingAIButton");


    if (chat) {

        chat.classList.remove("show");

    }


    if (button) {

        button.classList.remove(
            "chat-open"
        );

    }


    /*
        Remember that user closed the window.

        Messages are NOT deleted.
    */

    sessionStorage.setItem(
        "cardiorisk_ai_open",
        "false"
    );
}



/* =========================================================
   ENTER KEY
========================================================= */

function handleAIEnter(event) {

    if (event.key === "Enter") {

        event.preventDefault();

        askFloatingAssistant();

    }
}



/* =========================================================
   SAVE CHAT
========================================================= */

function saveAIChat() {

    const messages =
        document.getElementById(
            "aiChatMessages"
        );


    if (!messages) {
        return;
    }


    const chatData = [];


    /*
        Get all user and bot messages
    */

    messages
        .querySelectorAll(".ai-message")
        .forEach(message => {

            /*
                Don't save Thinking...
            */

            if (
                message.classList.contains(
                    "thinking-message"
                )
            ) {
                return;
            }


            const content =
                message.querySelector(
                    ".message-content"
                );


            if (!content) {
                return;
            }


            const text =
                content.textContent.trim();


            if (!text) {
                return;
            }


            chatData.push({

                sender:
                    message.classList.contains(
                        "user-message"
                    )
                        ? "user"
                        : "bot",

                message: text

            });

        });


    /*
        Save in sessionStorage

        This disappears after browser/app session ends.
    */

    sessionStorage.setItem(

        aiUserKey(AI_CHAT_STORAGE_KEY),

        JSON.stringify(chatData)

    );
}



/* =========================================================
   LOAD CHAT
========================================================= */

function loadAIChat() {

    const messages =
        document.getElementById(
            "aiChatMessages"
        );


    if (!messages) {
        return;
    }


    let chatData = [];


    try {

        chatData =
            JSON.parse(

                sessionStorage.getItem(
                    aiUserKey(AI_CHAT_STORAGE_KEY)
                ) || "[]"

            );

    } catch (error) {

        chatData = [];

    }


    /*
        If previous messages exist,
        restore them.
    */

    if (
        Array.isArray(chatData) &&
        chatData.length > 0
    ) {

        chatData.forEach(item => {

            addChatMessage(
                item.message,
                item.sender,
                false
            );

        });

    }

    else {

        /*
            First time chat message
        */

        addChatMessage(

            "Hi! I'm your CardioRisk AI Assistant 👋",

            "bot",

            false

        );


        addChatMessage(

            "Ask me about your cardiovascular assessment, risk score, BMI, blood pressure, cholesterol, lifestyle, or your latest results.",

            "bot",

            false

        );


        /*
            Save welcome message
        */

        saveAIChat();

    }


    /*
        Restore open/close state

        Usually closed initially.
    */

    const wasOpen =
        sessionStorage.getItem(
            "cardiorisk_ai_open"
        );


    if (wasOpen === "true") {

        const chat =
            document.getElementById(
                "aiChatWindow"
            );

        const button =
            document.getElementById(
                "floatingAIButton"
            );


        if (chat) {

            chat.classList.add("show");

        }


        if (button) {

            button.classList.add(
                "chat-open"
            );

        }

    }

}



/* =========================================================
   ADD MESSAGE
========================================================= */

function addChatMessage(

    message,

    sender = "bot",

    shouldSave = true

) {

    const messages =
        document.getElementById(
            "aiChatMessages"
        );


    if (!messages) {
        return null;
    }


    const messageDiv =
        document.createElement("div");


    /*
        Set class
    */

    messageDiv.className =
        sender === "user"

            ? "ai-message user-message"

            : "ai-message bot-message";



    /*
        USER MESSAGE
    */

    if (sender === "user") {

        messageDiv.innerHTML = `

            <div class="message-content">

                <p>
                    ${escapeHtml(message)}
                </p>

            </div>

        `;

    }


    /*
        BOT MESSAGE
    */

    else {

        messageDiv.innerHTML = `

            <div class="message-avatar">

                🤖

            </div>


            <div class="message-content">

                <p>
                    ${escapeHtml(message)}
                </p>

            </div>

        `;

    }


    /*
        Add message
    */

    messages.appendChild(
        messageDiv
    );


    /*
        Scroll down
    */

    scrollAIChatToBottom();


    /*
        Save chat

        Thinking messages are not saved.
    */

    if (shouldSave) {

        saveAIChat();

    }


    return messageDiv;
}



/* =========================================================
   SCROLL CHAT
========================================================= */

function scrollAIChatToBottom() {

    const messages =
        document.getElementById(
            "aiChatMessages"
        );


    if (!messages) {
        return;
    }


    messages.scrollTop =
        messages.scrollHeight;

}



/* =========================================================
   SHOW THINKING MESSAGE
========================================================= */

function showThinkingMessage() {

    const messages =
        document.getElementById(
            "aiChatMessages"
        );


    if (!messages) {
        return null;
    }


    const thinking =
        document.createElement("div");


    thinking.className =
        "ai-message bot-message thinking-message";


    thinking.innerHTML = `

        <div class="message-avatar">

            🤖

        </div>


        <div class="message-content">

            <div class="typing-indicator">

                <span></span>
                <span></span>
                <span></span>

            </div>

        </div>

    `;


    messages.appendChild(
        thinking
    );


    scrollAIChatToBottom();


    return thinking;
}



/* =========================================================
   ASK AI ASSISTANT
========================================================= */

let assistantSubmitting = false;

async function askFloatingAssistant() {
    if (assistantSubmitting) return;

    const input = document.getElementById("floatingAIQuestion");
    if (!input) return;

    const question = input.value.trim();
    if (question.length < 2) return;

    if (!token()) {
        addChatMessage("Please login first to use the CardioRisk AI Assistant.", "bot");
        return;
    }

    assistantSubmitting = true;
    input.disabled = true;
    const sendButton = document.querySelector(".ai-send-btn");
    if (sendButton) sendButton.disabled = true;

    addChatMessage(question, "user");
    input.value = "";
    const thinking = showThinkingMessage();

    try {
        const response = await fetch(API + "/assistant", {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify({ question })
        });

        let data = {};
        try { data = await response.json(); } catch { }
        if (!response.ok) {
            throw new Error(data.detail || "Assistant unavailable");
        }

        if (thinking) thinking.remove();
        addChatMessage(
            data.answer || "Sorry, I couldn't generate a response right now.",
            "bot"
        );
    } catch (error) {
        if (thinking) thinking.remove();
        console.error("Assistant error:", error);
        addChatMessage(
            error.message || "Sorry, I couldn't connect to the AI assistant. Please try again.",
            "bot"
        );
    } finally {
        assistantSubmitting = false;
        input.disabled = false;
        input.focus();
        if (sendButton) sendButton.disabled = false;
    }
}

/* =========================================================
   CREATE GLOBAL ASSISTANT
========================================================= */

if (token()) {
    createFloatingAssistant();
}

/* =========================
   PAGE ACCESS + NAVIGATION
========================= */

document.addEventListener("DOMContentLoaded", () => {
    const page = location.pathname.split("/").pop() || "index.html";
    const publicPages = ["login.html", "register.html", "auth.html"];

    // Logged-in users should not stay on the sign-in/register pages.
    if (token() && publicPages.includes(page)) {
        location.href = "assessment.html";
        return;
    }

    // Keep protected pages protected even after refresh/direct URL access.
    if (!token() && !publicPages.includes(page)) {
        location.href = "login.html";
        return;
    }

    // Logged-out/public pages must never show the floating chatbot.
    if (!token()) {
        const widget = document.getElementById("aiAssistantWidget");
        if (widget) widget.remove();
    }

    nav();
});

