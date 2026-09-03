from pathlib import Path
from typing import Any
import json
import os
import secrets
import uuid

from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd


from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import tempfile
import shutil

try:
    import shap
except ImportError:
    shap = None
    print("SHAP not installed: using lightweight feature-importance fallback for deployment.")

import time
import requests

try:
    from .database import (
        init_db,
        create_patient,
        get_login,
        get_assessments,
        conn
    )
except ImportError:
    from database import (
        init_db,
        create_patient,
        get_login,
        get_assessments,
        conn
    )


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "cardiovascular_model.pkl"
FEATURES_PATH = ROOT / "features.pkl"
METRICS_PATH = ROOT / "metrics.json"

# ============================================================
# RESEARCH MODEL PATHS
# ============================================================

PROJECT_ROOT = ROOT.parent

CLINICAL_MODEL_PATH = (
    PROJECT_ROOT
    / "research"
    / "models"
    / "clinical_xgboost.pkl"
)

ECG_MODEL_PATH = (
    PROJECT_ROOT
    / "research"
    / "models"
    / "ecg_cnn.keras"
)

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = ROOT / ".env"

if ENV_PATH.exists():

    for line in ENV_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if (
            line
            and not line.startswith("#")
            and "=" in line
        ):

            key, value = line.split("=", 1)

            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'")
            )


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    ""
).strip() or "gemini-3.6-flash"


# ============================================================
# GEMINI AI
# ============================================================

# ============================================================
# GEMINI AI ASSISTANT
# ============================================================

def ask_gemini(question: str) -> str | None:
    """Call Gemini through the REST API with a strict timeout."""
    if not GEMINI_API_KEY:
        print("❌ Gemini API key is missing.")
        return None

    system_instruction = """
You are CardioRisk AI Assistant, a friendly multilingual AI assistant inside a healthcare application.
Answer the user's ACTUAL question directly and naturally. Detect the user's language and reply in the same language whenever possible, including English, Hindi, Hinglish and other languages.

You can have normal conversations and answer harmless general questions. You are especially helpful with cardiovascular health, risk scores, blood pressure, cholesterol, BMI, diabetes, exercise, diet, sleep, stress and lifestyle.

If the user mentions low, moderate or high cardiovascular risk, explain what it generally means and give practical general guidance relevant to the question. Never invent patient measurements or claim to know medical data that was not provided.

Do not diagnose a disease or prescribe medicines or medication doses. For severe chest pain, severe breathing difficulty, fainting, or stroke symptoms, advise immediate emergency medical care.

Keep answers concise and conversational unless the user asks for detail. Do not give repetitive generic introductions. Directly answer the specific question.
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400}
    }

    try:
        print(f"\n🤖 Sending Gemini REST request: {GEMINI_MODEL}")
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=(8, 25)
        )
        print(f"📡 Gemini status: {response.status_code}")
        if not response.ok:
            print("❌ Gemini API error:", response.text[:1000])
            return None

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            print("❌ Gemini returned no candidates.")
            return None
        parts = (candidates[0].get("content") or {}).get("parts") or []
        answer = "".join(part.get("text", "") for part in parts).strip()
        if answer:
            print("✅ Gemini response received successfully.")
            return answer
        print("❌ Gemini returned an empty answer.")
        return None
    except requests.Timeout:
        print("❌ Gemini request timed out.")
        return None
    except Exception as error:
        print("❌ Gemini assistant error:", repr(error))
        return None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="CardioRisk AI API",

    version="1.1"

)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

init_db()


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = None
model_explainer = None

print(f"Main model path: {MODEL_PATH}")
print(f"Main model exists: {MODEL_PATH.exists()}")
print(f"Features path: {FEATURES_PATH}")
print(f"Features file exists: {FEATURES_PATH.exists()}")

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing deployment model file: {MODEL_PATH}")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Machine learning model loaded successfully."
    )

    try:

        # The saved main model is a sklearn Pipeline. TreeExplainer must
        # receive the underlying tree model, while SHAP values must be
        # calculated on the pipeline's preprocessed feature matrix.
        if shap is not None:
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                model_explainer = shap.TreeExplainer(model.named_steps["model"])
            else:
                model_explainer = shap.TreeExplainer(model)

        print(
            "Main model SHAP explainer loaded successfully."
        )

    except Exception as error:

        print(
            "Main model SHAP explainer load error:",
            error
        )

        model_explainer = None


except Exception as error:

    print(
        "Model load error:",
        error
    )

    model = None
    model_explainer = None

# ============================================================
# OPTIONAL RESEARCH MODELS
# ============================================================
# The production cardiovascular model above is required for the
# assessment feature. Research models are optional and must never
# prevent the API from starting.

clinical_model = None
clinical_features = []
clinical_explainer = None

try:
    if CLINICAL_MODEL_PATH.exists():
        clinical_bundle = joblib.load(CLINICAL_MODEL_PATH)
        clinical_model = clinical_bundle.get("model")
        clinical_features = clinical_bundle.get("features", [])

        if clinical_model is not None and shap is not None:
            try:
                clinical_explainer = shap.TreeExplainer(clinical_model)
            except Exception as error:
                print("Clinical SHAP explainer unavailable:", error)

        print("Clinical XGBoost model loaded successfully.")
        print("Clinical features loaded:", len(clinical_features))
    else:
        print("Clinical research model not included in this deployment.")
except Exception as error:
    print("Clinical model load error:", error)
    clinical_model = None
    clinical_features = []
    clinical_explainer = None


# ============================================================
# ECG CNN
# ============================================================
# TensorFlow is intentionally excluded from the Vercel deployment to
# keep the serverless bundle below the function-size limit.
ecg_model = None
print("ECG CNN model is disabled on the Vercel serverless deployment.")

# ============================================================
# LOAD TRAINED FEATURES
# ============================================================

try:

    trained_features = joblib.load(
        FEATURES_PATH
    )

    print(
        "Features loaded:",
        len(trained_features)
    )

    print("\n========== TRAINED MODEL FEATURES ==========")

    for i, feature in enumerate(trained_features, 1):
        print(f"{i}. {feature}")

    print("===========================================\n")


except Exception as error:

    print(
        "Feature load error:",
        error
    )

    trained_features = []


# ============================================================
# AUTH TOKEN FUNCTIONS
# Stateless signed tokens work across multiple serverless instances.
# ============================================================

import base64
import hashlib
import hmac

APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "change-this-before-production").encode("utf-8")


def issue_token(patient_id):
    payload = {
        "patient_id": int(patient_id),
        "expires": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    signature = hmac.new(
        APP_SECRET_KEY,
        encoded.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return f"{encoded}.{signature}"


def patient_id_from_auth(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Login required")

    token = authorization.split(" ", 1)[1].strip()

    try:
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(
            APP_SECRET_KEY,
            encoded.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")

        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(padded.encode("utf-8"))
        )

        expires = datetime.fromisoformat(payload["expires"])
        if expires < datetime.utcnow():
            raise ValueError("Expired token")

        return int(payload["patient_id"])

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session. Please login again."
        )


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics():

    try:

        if METRICS_PATH.exists():

            return json.loads(
                METRICS_PATH.read_text()
            )


    except Exception as error:

        print(
            "Metrics load error:",
            error
        )


    return {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def yn(value):

    if isinstance(
        value,
        (int, float, bool)
    ):

        return int(value)


    return int(

        str(value)
        .strip()
        .lower()

        in [

            "1",
            "true",
            "yes",
            "y"

        ]

    )



def cat(value):

    if isinstance(
        value,
        (int, float)
    ):

        return int(value)


    mapping = {

        "normal": 1,

        "above normal": 2,

        "above_normal": 2,

        "high": 3

    }


    return mapping.get(

        str(value)
        .lower()
        .strip(),

        1

    )



def gender(value):

    if isinstance(
        value,
        (int, float)
    ):

        return int(value)


    if str(value).lower() in [
        "female",
        "f"
    ]:

        return 2


    return 1


# ============================================================
# REQUEST MODELS
# ============================================================

class AuthRequest(BaseModel):

    full_name: str | None = None

    name: str | None = None

    email: str

    phone: str = ""

    password: str



class AssessmentRequest(BaseModel):

    age: float

    gender: Any

    height: float

    weight: float

    ap_hi: float

    ap_lo: float

    cholesterol: Any

    gluc: Any

    smoke: Any

    alco: Any

    active: Any

    family_history: Any = 0

    ecg_summary: Any = 0

class ClinicalPredictionRequest(BaseModel):

    age: float
    sex: int
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float



class AssistantRequest(BaseModel):

    question: str



class GuidanceRequest(BaseModel):

    assessment: dict

    risk_probability: float | None = None

    risk_level: str | None = None

    bmi: float | None = None

    contributing_factors: list = []


# ============================================================
# HOME
# ============================================================

@app.get("/", include_in_schema=False)
def home():
    frontend_index = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)

    return {
        "message": "CardioRisk AI Backend Running",
        "model_loaded": model is not None,
        "feature_count": len(trained_features)
    }


@app.get("/api-status")
def api_status():
    return {
        "message": "CardioRisk AI Backend Running",
        "model_loaded": model is not None,
        "feature_count": len(trained_features)
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "features_loaded": bool(trained_features),
        "features_file_exists": FEATURES_PATH.exists()
    }


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")

def metrics():

    return load_metrics()


# ============================================================
# REGISTER
# ============================================================

@app.post("/register")

def register(r: AuthRequest):

    name = (

        r.full_name
        or r.name
        or ""

    ).strip()


    if len(name) < 2:

        raise HTTPException(

            400,

            "Please enter your full name."

        )


    if len(r.password) < 6:

        raise HTTPException(

            400,

            "Password must contain at least 6 characters."

        )


    try:

        patient_id = create_patient(

            name,

            r.email,

            r.phone,

            r.password

        )


    except Exception:

        raise HTTPException(

            400,

            "An account with this email already exists."

        )


    return {

        "token":
            issue_token(patient_id),

        "patient_id":
            patient_id

    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")

def login(r: AuthRequest):

    patient = get_login(

        r.email,

        r.password

    )


    if not patient:

        raise HTTPException(

            401,

            "Invalid email or password."

        )


    patient["token"] = issue_token(

        patient["patient_id"]

    )


    return {

        "patient":
            patient

    }


# ============================================================
# FEATURE CREATION
# ============================================================

# ============================================================
# FEATURE CREATION
# ============================================================

def make_row(data):

    age = float(data["age"])
    height = float(data["height"])
    weight = float(data["weight"])

    ap_hi = float(data["ap_hi"])
    ap_lo = float(data["ap_lo"])

    cholesterol = cat(data["cholesterol"])
    gluc = cat(data["gluc"])

    smoke = yn(data["smoke"])
    alco = yn(data["alco"])
    active = yn(data["active"])

    gender_value = gender(data["gender"])


    # ========================================================
    # BASIC CALCULATIONS
    # ========================================================

    bmi = weight / ((height / 100) ** 2)

    pulse_pressure = ap_hi - ap_lo

    mean_arterial_pressure = (
        ap_lo + pulse_pressure / 3
    )

    bp_ratio = (
        ap_hi / ap_lo
        if ap_lo != 0
        else 0
    )


    # ========================================================
    # ENGINEERED FEATURES
    # THESE MUST MATCH THE TRAINED MODEL FEATURES
    # ========================================================

    row = {

        # ----------------------------------------------------
        # ORIGINAL FEATURES
        # ----------------------------------------------------

        "age": age,
        "gender": gender_value,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,


        # ----------------------------------------------------
        # AGE FEATURES
        # ----------------------------------------------------

        "age_years": age,

        "age_squared": age ** 2,

        "age_cubed": age ** 3,

        "age_over_40": int(age > 40),

        "age_over_50": int(age > 50),

        "age_over_60": int(age > 60),


        # ----------------------------------------------------
        # BMI FEATURES
        # ----------------------------------------------------

        "bmi": bmi,

        "bmi_squared": bmi ** 2,

        "underweight": int(bmi < 18.5),

        "overweight": int(25 <= bmi < 30),

        "obese": int(30 <= bmi < 35),

        "severely_obese": int(bmi >= 35),


        # ----------------------------------------------------
        # BLOOD PRESSURE FEATURES
        # ----------------------------------------------------

        "pulse_pressure": pulse_pressure,

        "mean_arterial_pressure":
            mean_arterial_pressure,

        "bp_ratio": bp_ratio,

        "high_bp":
            int(ap_hi >= 140 or ap_lo >= 90),

        "very_high_bp":
            int(ap_hi >= 160 or ap_lo >= 100),

        "normal_bp":
            int(ap_hi < 120 and ap_lo < 80),

        "bp_sum":
            ap_hi + ap_lo,

        "bp_squared":
            ap_hi ** 2 + ap_lo ** 2,


        # ----------------------------------------------------
        # CHOLESTEROL FEATURES
        # ----------------------------------------------------

        "high_cholesterol":
            int(cholesterol >= 2),

        "very_high_cholesterol":
            int(cholesterol == 3),


        # ----------------------------------------------------
        # GLUCOSE FEATURES
        # ----------------------------------------------------

        "high_glucose":
            int(gluc >= 2),

        "very_high_glucose":
            int(gluc == 3),


        # ----------------------------------------------------
        # ACTIVITY / LIFESTYLE
        # ----------------------------------------------------

        "inactive":
            int(active == 0),

        "lifestyle_risk_score":
            (
                smoke
                + alco
                + int(active == 0)
            ),


        # ----------------------------------------------------
        # INTERACTION FEATURES
        # ----------------------------------------------------

        "age_bp_interaction":
            age * ap_hi,

        "age_diastolic_interaction":
            age * ap_lo,

        "age_bmi_interaction":
            age * bmi,

        "bmi_bp_interaction":
            bmi * ap_hi,

        "cholesterol_age":
            cholesterol * age,

        "glucose_bmi":
            gluc * bmi,


        # ----------------------------------------------------
        # COMBINED CLINICAL RISK
        # ----------------------------------------------------

        "clinical_risk_score":

            int(age >= 50)

            + int(ap_hi >= 140)

            + int(ap_lo >= 90)

            + int(cholesterol >= 2)

            + int(gluc >= 2)

            + smoke

            + int(active == 0)

    }


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    missing_features = [

        feature

        for feature in trained_features

        if feature not in row

    ]


    if missing_features:

        raise ValueError(

            "Missing trained model features: "

            + ", ".join(missing_features)

        )


    # ========================================================
    # CREATE EXACT MODEL INPUT
    # ========================================================

    X = pd.DataFrame(

        [

            {

                feature: row[feature]

                for feature in trained_features

            }

        ],

        columns=trained_features

    )


    return X, row

# ============================================================
# CLINICAL XGBOOST FEATURE CREATION
# ============================================================

def make_clinical_features(data):

    age = float(data["age"])
    sex = float(data["sex"])
    cp = float(data["cp"])
    trestbps = float(data["trestbps"])
    chol = float(data["chol"])
    fbs = float(data["fbs"])
    restecg = float(data["restecg"])
    thalach = float(data["thalach"])
    exang = float(data["exang"])
    oldpeak = float(data["oldpeak"])
    slope = float(data["slope"])
    ca = float(data["ca"])
    thal = float(data["thal"])

    features = {

        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,

        "age_risk": int(age >= 55),
        "bp_risk": int(trestbps >= 140),
        "chol_risk": int(chol >= 240),

        "age_chol": age * chol,
        "age_bp": age * trestbps,

        "age_hr_ratio": (
            age / thalach
            if thalach != 0
            else 0
        ),

        "st_risk": oldpeak * exang

    }

    return pd.DataFrame([features])

# ============================================================
# RISK FACTORS
# ============================================================

def factors(data, row):

    output = []

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if float(data["age"]) >= 55:

        output.append(
            "Older age-related cardiovascular risk"
        )


    # --------------------------------------------------------
    # BMI / WEIGHT
    # --------------------------------------------------------

    if row["bmi"] < 18.5:

        output.append(
            "Below healthy weight range"
        )

    elif row["bmi"] >= 30:

        output.append(
            "Obesity"
        )

    elif row["bmi"] >= 25:

        output.append(
            "Above healthy weight range"
        )


    # --------------------------------------------------------
    # BLOOD PRESSURE
    # --------------------------------------------------------

    if (

        float(data["ap_hi"]) >= 160

        or

        float(data["ap_lo"]) >= 100

    ):

        output.append(
            "Significantly elevated blood pressure"
        )

    elif (

        float(data["ap_hi"]) >= 140

        or

        float(data["ap_lo"]) >= 90

    ):

        output.append(
            "Elevated blood pressure"
        )


    # --------------------------------------------------------
    # CHOLESTEROL
    # --------------------------------------------------------

    cholesterol_value = cat(
        data["cholesterol"]
    )

    if cholesterol_value == 3:

        output.append(
            "High cholesterol"
        )

    elif cholesterol_value == 2:

        output.append(
            "Above-normal cholesterol"
        )


    # --------------------------------------------------------
    # GLUCOSE
    # --------------------------------------------------------

    glucose_value = cat(
        data["gluc"]
    )

    if glucose_value == 3:

        output.append(
            "High glucose"
        )

    elif glucose_value == 2:

        output.append(
            "Above-normal glucose"
        )


    # --------------------------------------------------------
    # SMOKING
    # --------------------------------------------------------

    if yn(data["smoke"]):

        output.append(
            "Smoking"
        )


    # --------------------------------------------------------
    # ALCOHOL
    # --------------------------------------------------------

    if yn(data["alco"]):

        output.append(
            "Alcohol consumption"
        )


    # --------------------------------------------------------
    # PHYSICAL ACTIVITY
    # --------------------------------------------------------

    if not yn(data["active"]):

        output.append(
            "Low physical activity"
        )


    # --------------------------------------------------------
    # FAMILY HISTORY
    # --------------------------------------------------------

    if yn(data.get("family_history", 0)):

        output.append(
            "Family history of cardiovascular disease"
        )


    # --------------------------------------------------------
    # ECG SUMMARY
    # --------------------------------------------------------

    ecg_value = data.get(
        "ecg_summary",
        0
    )

    if str(ecg_value).strip().lower() not in [

        "",
        "0",
        "normal",
        "not available",
        "none",
        "no"

    ]:

        output.append(
            "Reported ECG abnormality or non-normal ECG finding"
        )


    return output



# ============================================================
# ECG CNN PREDICTION
# ============================================================

@app.post("/ecg-predict")
async def ecg_predict(
    file: UploadFile = File(...)
):

    if ecg_model is None:
        raise HTTPException(
            status_code=503,
            detail="ECG CNN model is unavailable."
        )

    try:

        import tempfile
        import wfdb

        # ----------------------------------------------------
        # SAVE UPLOADED FILE TEMPORARILY
        # ----------------------------------------------------

        suffix = Path(file.filename).suffix.lower()

        if suffix not in [".csv", ".txt"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported ECG file format. "
                    "Please upload a CSV or TXT ECG signal file."
                )
            )

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded ECG file is empty."
            )

        # ----------------------------------------------------
        # LOAD ECG SIGNAL
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(content)

            temp_path = temp_file.name


        try:

            # Try CSV first
            if suffix == ".csv":

                signal = np.loadtxt(
                    temp_path,
                    delimiter=","
                )

            else:

                signal = np.loadtxt(
                    temp_path
                )

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        signal = np.array(
            signal,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # HANDLE 1D ECG
        # ----------------------------------------------------

        if signal.ndim == 1:

            signal = signal.reshape(
                -1,
                1
            )


        # ----------------------------------------------------
        # VALIDATE SIGNAL
        # ----------------------------------------------------

        if signal.ndim != 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid ECG signal format."
                )
            )


        # ----------------------------------------------------
        # PTB-XL MODEL EXPECTS 1000 SAMPLES
        # ----------------------------------------------------

        SIGNAL_LENGTH = 1000


        # Pad short signals
        if signal.shape[0] < SIGNAL_LENGTH:

            padding = SIGNAL_LENGTH - signal.shape[0]

            signal = np.pad(
                signal,
                (
                    (0, padding),
                    (0, 0)
                ),
                mode="constant"
            )


        # Trim long signals
        elif signal.shape[0] > SIGNAL_LENGTH:

            signal = signal[
                :SIGNAL_LENGTH
            ]


        # ----------------------------------------------------
        # CHECK EXPECTED ECG CHANNELS
        # ----------------------------------------------------

        expected_channels = (
            ecg_model.input_shape[-1]
        )


        # Single channel input
        if signal.shape[1] == 1 and expected_channels > 1:

            signal = np.repeat(
                signal,
                expected_channels,
                axis=1
            )


        # Too many channels
        elif signal.shape[1] > expected_channels:

            signal = signal[
                :,
                :expected_channels
            ]


        # Too few channels
        elif signal.shape[1] < expected_channels:

            padding = (
                expected_channels -
                signal.shape[1]
            )

            signal = np.pad(
                signal,
                (
                    (0, 0),
                    (0, padding)
                ),
                mode="constant"
            )


        # ----------------------------------------------------
        # NORMALIZATION
        # SAME AS TRAINING
        # ----------------------------------------------------

        mean = np.mean(
            signal,
            keepdims=True
        )

        std = np.std(
            signal,
            keepdims=True
        )

        signal = (
            signal - mean
        ) / (
            std + 1e-8
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        X = np.expand_dims(
            signal,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        probability = float(
            ecg_model.predict(
                X,
                verbose=0
            )[0][0]
        )


        prediction = (
            "Abnormal ECG"
            if probability >= 0.5
            else
            "Normal ECG"
        )


        confidence = (
            probability
            if probability >= 0.5
            else
            1 - probability
        )


        return {

    "success": True,

    # Numeric value for frontend logic
    "prediction": int(probability >= 0.5),

    # Human-readable result
    "prediction_label": prediction,

    "abnormal_probability":
        round(probability, 4),

    "normal_probability":
        round(1 - probability, 4),

    "confidence":
        round(confidence, 4),

    "model":
        "PTB-XL ECG CNN",

    "signal_length":
        SIGNAL_LENGTH,

    "channels":
        int(expected_channels)

}


    except HTTPException:

        raise


    except Exception as error:

        print(
            "ECG prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"ECG prediction failed: {str(error)}"
            )
        )


# ============================================================
# CLINICAL XGBOOST PREDICTION
# ============================================================

@app.post("/clinical-predict")

def clinical_predict(

    r: ClinicalPredictionRequest,

    authorization: str | None = Header(None)

):

    patient_id_from_auth(authorization)

    if clinical_model is None:

        raise HTTPException(
            500,
            "Clinical XGBoost model is unavailable."
        )

    try:

        data = r.model_dump()

        X = make_clinical_features(data)

        probability = float(
            clinical_model.predict_proba(X)[0][1]
        )

        percentage = round(
            probability * 100,
            2
        )

        if probability < 0.30:
            risk_level = "LOW"

        elif probability < 0.60:
            risk_level = "MODERATE"

        else:
            risk_level = "HIGH"

                # ====================================================
        # SHAP EXPLAINABILITY
        # ====================================================

        shap_explanation = []

        if clinical_explainer is not None:

            try:

                shap_values = clinical_explainer.shap_values(X)

                # Different SHAP/XGBoost versions can return
                # different output formats
                if isinstance(shap_values, list):

                    values = np.array(shap_values[-1])[0]

                else:

                    values = np.array(shap_values)

                    if values.ndim == 3:
                        values = values[0, :, -1]

                    elif values.ndim == 2:
                        values = values[0]


                feature_names = list(X.columns)


                explanations = []

                for feature, value, feature_value in zip(
                    feature_names,
                    values,
                    X.iloc[0].values
                ):

                    explanations.append({

                        "feature": str(feature),

                        "feature_value": float(feature_value),

                        "shap_value": round(float(value), 6),

                        "impact":
                            "Increases Risk"
                            if value > 0
                            else "Decreases Risk"

                    })


                # Sort by absolute importance
                explanations.sort(
                    key=lambda item: abs(item["shap_value"]),
                    reverse=True
                )


                shap_explanation = explanations[:8]


            except Exception as error:

                print(
                    "SHAP calculation error:",
                    error
                )

        return {

            "success": True,

            "model":
                "Clinical XGBoost + SHAP",

            "prediction":
                int(probability >= 0.5),

            "probability":
                round(probability, 6),

            "risk_probability":
                percentage,

            "risk_level":
                risk_level,

            "risk_category":
                risk_level.title() + " Risk",

            "shap_explanation":
                shap_explanation,

            "explainability_method":
                "SHAP"

        }

    except Exception as error:

        print(
            "Clinical prediction error:",
            error
        )

        raise HTTPException(
            500,
            f"Clinical prediction failed: {error}"
        )

# ============================================================
# MAIN MODEL SHAP EXPLAINABILITY
# ============================================================

@app.post("/predict-explanation")
def predict_explanation(
    r: AssessmentRequest,
    authorization: str | None = Header(None)
):
    patient_id_from_auth(authorization)
    if model is None or not trained_features:
        raise HTTPException(status_code=500, detail="Machine learning model is unavailable.")
    try:
        data = r.model_dump()
        X, _ = make_row(data)
        probability = float(model.predict_proba(X)[0][1])

        # Use SHAP when available; otherwise use a lightweight model-feature fallback.
        explanations = []
        method = "SHAP"
        if shap is not None:
            if hasattr(model, "named_steps") and "preprocess" in model.named_steps and "model" in model.named_steps:
                preprocess = model.named_steps["preprocess"]
                tree_model = model.named_steps["model"]
                X_for_shap = preprocess.transform(X)
                explainer = model_explainer or shap.TreeExplainer(tree_model)
            else:
                X_for_shap = X
                explainer = model_explainer or shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_for_shap)
            values = np.asarray(shap_values)
            if isinstance(shap_values, list): values = np.asarray(shap_values[-1])
            if values.ndim == 3: values = values[0, :, -1]
            elif values.ndim == 2: values = values[0]
            elif values.ndim != 1: raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
            feature_values = np.asarray(X.iloc[0].values, dtype=float)
            for feature, value, feature_value in zip(trained_features, values, feature_values):
                sv=float(value)
                explanations.append({"feature":str(feature),"feature_value":float(feature_value),"shap_value":round(sv,6),"impact":"Increases Risk" if sv>0 else "Decreases Risk"})
        else:
            method = "Model feature importance (deployment fallback)"
            estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
            importances = getattr(estimator, "feature_importances_", None)
            if importances is None:
                importances = np.ones(len(trained_features))
            vals = np.asarray(importances).reshape(-1)
            for feature, value, imp in zip(trained_features, np.asarray(X.iloc[0].values, dtype=float), vals[:len(trained_features)]):
                score=float(imp)
                explanations.append({"feature":str(feature),"feature_value":float(value),"shap_value":round(score,6),"impact":"Higher importance"})

        explanations.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        return {"success":True,"model":"Cardiovascular Risk Model","risk_probability":round(probability*100,2),"shap_explanation":explanations[:8],"explainability_method":method}
    except Exception as error:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction explanation failed: {str(error)}")

# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")

def predict(

    r: AssessmentRequest,

    authorization: str | None = Header(None)

):

    patient_id = patient_id_from_auth(
        authorization
    )


    if (

        model is None

        or

        not trained_features

    ):

        raise HTTPException(

            500,

            "Machine learning model is unavailable."

        )


    data = r.model_dump()


    # Validation

    if not (

        18 <= data["age"] <= 120

        and

        100 <= data["height"] <= 250

        and

        30 <= data["weight"] <= 250

        and

        70 <= data["ap_hi"] <= 250

        and

        40 <= data["ap_lo"] <= 150

        and

        data["ap_lo"] < data["ap_hi"]

    ):

        raise HTTPException(

            400,

            "Please enter valid assessment values."

        )


    try:

        X, row = make_row(data)


        probability = float(

            model.predict_proba(X)[0][1]

        )


        percentage = round(

            probability * 100,

            2

        )


    except Exception as error:

        import traceback

        print("========== PREDICT ERROR ==========")
        print(repr(error))
        traceback.print_exc()
        print("===================================")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


    # Risk Level

    if probability < 0.30:

        level = "LOW"

    elif probability < 0.60:

        level = "MODERATE"

    else:

        level = "HIGH"


    model_name = load_metrics().get(

        "selected_model",

        "XGBoost"

    )


    contributing_factors = factors(
        data,
        row
    )


    recommendations = [

        "Continue regular preventive health checkups."

    ]


    if contributing_factors:

        recommendations = [

            f"Focus on: {factor}."

            for factor
            in contributing_factors

        ]


    # Save assessment

    c = conn()


    c.execute(

        """
        INSERT INTO assessments
        (
            patient_id,
            age,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            family_history,
            ecg_status,
            bmi,
            risk_probability,
            risk_level,
            model,
            created_at
        )

        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,

        (

            patient_id,

            data["age"],

            gender(data["gender"]),

            data["height"],

            data["weight"],

            data["ap_hi"],

            data["ap_lo"],

            cat(data["cholesterol"]),

            cat(data["gluc"]),

            yn(data["smoke"]),

            yn(data["alco"]),

            yn(data["active"]),

            yn(data["family_history"]),

            str(data["ecg_summary"]),

            row["bmi"],

            percentage,

            level,

            model_name,

            datetime.now().isoformat()

        )

    )


    c.commit()

    c.close()


    return {

        "success":
            True,

        "prediction":
            int(probability >= 0.5),

        "risk_probability":
            percentage,

        "probability":
            round(probability, 6),

        "risk_category":
            level.title() + " Risk",

        "risk_level":
            level,

        "bmi":
            round(row["bmi"], 2),

        "model":
            model_name,

        "contributing_factors":
            contributing_factors,

        "recommendations":
            recommendations,

        "calculated_values": {

            "bmi":
                round(row["bmi"], 2),

            "pulse_pressure":
                round(
                    row["pulse_pressure"],
                    2
                ),

            "mean_arterial_pressure":
                round(
                    row["mean_arterial_pressure"],
                    2
                ),

            "bp_ratio":
                round(
                    row["bp_ratio"],
                    3
                )

        },

        "message":
            "Cardiovascular risk assessment completed successfully."

    }


# ============================================================
# ASSESSMENT HISTORY
# ============================================================

@app.get("/assessments")

def assessments(

    authorization: str | None = Header(None)

):

    patient_id = patient_id_from_auth(
        authorization
    )


    return {

        "assessments":

            get_assessments(
                patient_id
            )

    }


# ============================================================
# PERSONALIZED GUIDANCE
# ============================================================

@app.post("/guidance")

def guidance(

    r: GuidanceRequest,

    authorization: str | None = Header(None)

):

    patient_id_from_auth(
        authorization
    )

    assessment = r.assessment


    # ========================================================
    # BASIC VALUES
    # ========================================================

    bmi = (

        r.bmi

        or

        assessment.get("bmi")

        or

        (

            float(
                assessment.get(
                    "weight",
                    70
                )
            )

            /

            (

                float(
                    assessment.get(
                        "height",
                        170
                    )
                )

                / 100

            ) ** 2

        )

    )


    risk_probability = (

        r.risk_probability

        if r.risk_probability is not None

        else 0

    )


    risk_level = str(

        r.risk_level

        or

        ""

    ).upper()


    priority = []

    lifestyle = []

    prevention = []


    # ========================================================
    # RISK LEVEL BASED GUIDANCE
    # ========================================================

    if risk_level in ["HIGH", "VERY HIGH"] or risk_probability >= 60:

        priority.append({

            "title":
                "Arrange medical follow-up",

            "description":
                "Your assessment indicates a higher cardiovascular risk. Discuss the results with a qualified healthcare professional."

        })

    elif risk_level == "MODERATE" or risk_probability >= 30:

        priority.append({

            "title":
                "Monitor cardiovascular risk factors",

            "description":
                "Focus on controlling modifiable risk factors and continue regular preventive health monitoring."

        })

    else:

        priority.append({

            "title":
                "Maintain healthy cardiovascular habits",

            "description":
                "Continue healthy lifestyle habits and regular preventive checkups."

        })


    # ========================================================
    # BLOOD PRESSURE
    # ========================================================

    ap_hi = float(
        assessment.get(
            "ap_hi",
            0
        )
    )

    ap_lo = float(
        assessment.get(
            "ap_lo",
            0
        )
    )


    if ap_hi >= 160 or ap_lo >= 100:

        priority.append({

            "title":
                "Address significantly elevated blood pressure",

            "description":
                "Repeatedly high blood pressure readings should be discussed with a qualified healthcare professional."

        })

    elif ap_hi >= 140 or ap_lo >= 90:

        priority.append({

            "title":
                "Monitor elevated blood pressure",

            "description":
                "Track blood pressure regularly and discuss persistently elevated readings with a healthcare professional."

        })


    # ========================================================
    # CHOLESTEROL
    # ========================================================

    cholesterol_value = cat(
        assessment.get(
            "cholesterol",
            1
        )
    )


    if cholesterol_value == 3:

        priority.append({

            "title":
                "Improve cholesterol management",

            "description":
                "High cholesterol is an important cardiovascular risk factor. Follow a heart-healthy lifestyle and discuss testing or management with a healthcare professional."

        })

    elif cholesterol_value == 2:

        lifestyle.append({

            "title":
                "Support healthy cholesterol",

            "description":
                "Choose more fiber-rich and minimally processed foods and continue monitoring cholesterol."

        })


    # ========================================================
    # GLUCOSE
    # ========================================================

    glucose_value = cat(
        assessment.get(
            "gluc",
            1
        )
    )


    if glucose_value >= 2:

        priority.append({

            "title":
                "Monitor glucose levels",

            "description":
                "Above-normal glucose can contribute to cardiovascular risk and should be monitored appropriately."

        })


    # ========================================================
    # BMI / WEIGHT
    # ========================================================

    if bmi >= 30:

        priority.append({

            "title":
                "Work toward a healthier weight",

            "description":
                "Gradual improvements in nutrition and physical activity can support weight management and cardiovascular health."

        })

    elif bmi >= 25:

        lifestyle.append({

            "title":
                "Maintain a healthy weight",

            "description":
                "Balanced nutrition and regular activity can help support a healthier weight."

        })


    # ========================================================
    # SMOKING
    # ========================================================

    if yn(
        assessment.get(
            "smoke",
            0
        )
    ):

        priority.append({

            "title":
                "Stop smoking",

            "description":
                "Stopping smoking is one of the most important actions for reducing cardiovascular risk."

        })


    # ========================================================
    # ALCOHOL
    # ========================================================

    if yn(
        assessment.get(
            "alco",
            0
        )
    ):

        lifestyle.append({

            "title":
                "Review alcohol consumption",

            "description":
                "Limiting alcohol consumption can support overall health and reduce avoidable health risks."

        })


    # ========================================================
    # PHYSICAL ACTIVITY
    # ========================================================

    if not yn(
        assessment.get(
            "active",
            0
        )
    ):

        priority.append({

            "title":
                "Increase physical activity gradually",

            "description":
                "Regular physical activity can support heart health. Choose activities appropriate to your health and fitness level."

        })

    else:

        lifestyle.append({

            "title":
                "Maintain regular physical activity",

            "description":
                "Continue regular activity appropriate to your health and fitness level."

        })


    # ========================================================
    # FAMILY HISTORY
    # ========================================================

    if yn(
        assessment.get(
            "family_history",
            0
        )
    ):

        priority.append({

            "title":
                "Consider family history in preventive care",

            "description":
                "A family history of cardiovascular disease can increase long-term risk, so regular preventive monitoring is especially important."

        })


    # ========================================================
    # ECG STATUS
    # ========================================================

    ecg_value = assessment.get(
        "ecg_summary",
        0
    )


    if str(ecg_value).strip().lower() not in [

        "",
        "0",
        "normal",
        "none",
        "no",
        "not available"

    ]:

        priority.append({

            "title":
                "Follow up on ECG findings",

            "description":
                "A reported non-normal ECG finding should be interpreted by a qualified healthcare professional in the context of your symptoms and clinical history."

        })


    # ========================================================
    # GENERAL LIFESTYLE GUIDANCE
    # ========================================================

    lifestyle.append({

        "title":
            "Choose heart-healthy foods",

        "description":
            "Emphasize vegetables, fruits, whole grains, legumes and minimally processed foods."

    })


    lifestyle.append({

        "title":
            "Support healthy sleep and stress management",

        "description":
            "Adequate sleep and healthy stress-management habits can support overall cardiovascular health."

    })


    # ========================================================
    # PREVENTION AND MONITORING
    # ========================================================

    prevention = [

        {

            "title":
                "Track important health indicators",

            "description":
                "Monitor blood pressure, weight, cholesterol and glucose over time when appropriate."

        },

        {

            "title":
                "Keep regular preventive checkups",

            "description":
                "Regular healthcare visits can help identify and manage cardiovascular risk factors early."

        }

    ]


    # ========================================================
    # MEDICAL FOLLOW-UP
    # ========================================================

    high_risk = (

        risk_level in [

            "HIGH",
            "VERY HIGH"

        ]

        or

        risk_probability >= 60

    )


    return {

        "source":
            "dynamic rule-based personalized guidance",

        "risk_summary": {

            "risk_level":
                risk_level or "NOT AVAILABLE",

            "risk_probability":
                risk_probability,

            "bmi":
                round(
                    float(bmi),
                    2
                )

        },

        "priority_actions":
            priority,

        "lifestyle_risk_reduction":
            lifestyle,

        "prevention_monitoring":
            prevention,

        "medical_follow_up": {

            "recommended":
                high_risk,

            "message":

                (

                    "A discussion with a qualified healthcare professional is recommended based on your cardiovascular risk assessment."

                    if high_risk

                    else

                    "Continue preventive care and discuss concerning symptoms or persistent abnormal readings with a healthcare professional."

                )

        }

    }

# ============================================================
# LOCAL AI FALLBACK
# ============================================================

def assistant_answer(question: str):

    q = question.lower().strip()


    # --------------------------------------------------------
    # MEDICAL DIAGNOSIS / MEDICINE
    # --------------------------------------------------------

    if any(

        word in q

        for word in [

            "diagnose",

            "prescription",

            "medicine dose",

            "medication dose"

        ]

    ):

        return (

            "I can provide general healthcare education, but I cannot diagnose a medical condition "
            "or prescribe medication. Please consult a qualified healthcare professional for personalised advice."

        )


    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    if q in [

        "hi",

        "hello",

        "hey",

        "hii",

        "hiii"

    ]:

        return (

            "Hi! 👋 I'm CardioRisk AI Assistant. "
            "I can help you understand health, fitness, cardiovascular disease, "
            "BMI, blood pressure, cholesterol and healthy lifestyle habits. ❤️"

        )


    # --------------------------------------------------------
    # WHO ARE YOU
    # --------------------------------------------------------

    if any(

        phrase in q

        for phrase in [

            "who are you",

            "who r u",

            "what are you",

            "your name",

            "what is your name"

        ]

    ):

        return (

            "I'm CardioRisk AI Assistant 🤖❤️. "
            "I'm a healthcare education assistant inside the CardioRisk AI application. "
            "I can help explain cardiovascular health, fitness, BMI, blood pressure, "
            "cholesterol, risk scores and healthy lifestyle habits."

        )


    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    if (

        "what is disease" in q

        or

        "what is a disease" in q

        or

        q == "disease"

    ):

        return (

            "A disease is a condition that affects how the body or mind normally functions. "
            "Diseases can have different causes, symptoms and levels of severity."

        )


    # --------------------------------------------------------
    # FITNESS
    # --------------------------------------------------------

    if any(

        phrase in q

        for phrase in [

            "remain fit",

            "stay fit",

            "how to stay fit",

            "how can i stay fit",

            "how to remain fit",

            "how can i remain fit",

            "fitness"

        ]

    ):

        return (

            "To remain fit, focus on regular physical activity, balanced nutrition, "
            "adequate sleep and stress management. 🚶‍♂️🥗💤 "
            "Try to exercise regularly, eat more fruits and vegetables, stay hydrated "
            "and avoid smoking."

        )


    # --------------------------------------------------------
    # CARDIOVASCULAR DISEASE
    # --------------------------------------------------------

    if any(

        phrase in q

        for phrase in [

            "cardiovascular disease",

            "heart disease",

            "cvd"

        ]

    ):

        return (

            "Cardiovascular disease refers to conditions that affect the heart and blood vessels. "
            "Common risk factors include high blood pressure, high cholesterol, smoking, diabetes, "
            "obesity and physical inactivity."

        )


    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    if any(

        phrase in q

        for phrase in [

            "risk score",

            "cardiovascular risk",

            "risk level"

        ]

    ):

        return (

            "A cardiovascular risk score estimates the likelihood of cardiovascular problems "
            "based on factors such as age, blood pressure, cholesterol, lifestyle and other clinical information. "
            "It is an estimate and not a medical diagnosis."

        )


    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if "bmi" in q:

        return (

            "BMI stands for Body Mass Index. It is calculated using height and weight "
            "and is commonly used as a general indicator of body weight status."

        )


    # --------------------------------------------------------
    # BLOOD PRESSURE
    # --------------------------------------------------------

    if any(

        phrase in q

        for phrase in [

            "blood pressure",

            "high bp",

            "low bp"

        ]

    ):

        return (

            "Blood pressure measures the force of blood against artery walls. "
            "It is commonly expressed using systolic and diastolic values. "
            "Maintaining healthy blood pressure is important for cardiovascular health."

        )


    # --------------------------------------------------------
    # CHOLESTEROL
    # --------------------------------------------------------

    if "cholesterol" in q:

        return (

            "Cholesterol is a fatty substance found in the blood. "
            "High cholesterol can contribute to cardiovascular risk, especially when combined "
            "with other risk factors such as high blood pressure or smoking."

        )


    # --------------------------------------------------------
    # EXERCISE
    # --------------------------------------------------------

    if any(

        word in q

        for word in [

            "exercise",

            "workout",

            "walking",

            "running"

        ]

    ):

        return (

            "Regular exercise can support heart health, fitness and weight management. "
            "Walking, cycling, swimming and other activities can be helpful. "
            "Choose activities appropriate for your health and fitness level."

        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (

        "I'm CardioRisk AI Assistant 🤖❤️. "
        "I can help answer general questions and explain health, fitness, "
        "cardiovascular disease, BMI, blood pressure, cholesterol, risk scores "
        "and healthy lifestyle habits."

    )


# ============================================================
# AI ASSISTANT
# ============================================================

# ============================================================
# AI ASSISTANT
# ============================================================

@app.post("/assistant")
def assistant(

    r: AssistantRequest,

    authorization: str | None = Header(None)

):

    # --------------------------------------------------------
    # AUTHENTICATE USER
    # --------------------------------------------------------

    patient_id = patient_id_from_auth(
        authorization
    )


    # --------------------------------------------------------
    # GET QUESTION
    # --------------------------------------------------------

    question = r.question.strip()


    if len(question) < 2:

        raise HTTPException(

            status_code=400,

            detail="Please enter a question."

        )


    # --------------------------------------------------------
    # ASK GEMINI
    # --------------------------------------------------------

    answer = ask_gemini(
        question
    )


    # --------------------------------------------------------
    # GEMINI SUCCESS
    # --------------------------------------------------------

    if answer:
        source = "gemini"
    else:
        answer = (
            "Sorry, the AI assistant is temporarily unavailable right now. "
            "Please try again in a moment."
        )
        source = "unavailable"


    # --------------------------------------------------------
    # CREATE JOB ID
    # --------------------------------------------------------

    job_id = str(
        uuid.uuid4()
    )


    # --------------------------------------------------------
    # SAVE CHAT
    # --------------------------------------------------------

    c = conn()


    try:

        c.execute(

            """
            INSERT INTO assistant_jobs
            (
                id,
                patient_id,
                question,
                answer,
                status,
                source,
                created_at,
                completed_at
            )

            VALUES
            (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            )
            """,

            (

                job_id,

                patient_id,

                question,

                answer,

                "completed",

                source

            )

        )


        c.commit()


    finally:

        c.close()


    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {

        "job_id":
            job_id,

        "answer":
            answer,

        "status":
            "completed",

        "source":
            source

    }


# ============================================================
# ASSISTANT PENDING
# ============================================================

@app.get("/assistant/pending")

def pending(

    authorization: str | None = Header(None)

):

    patient_id_from_auth(
        authorization
    )


    return []


# ============================================================
# GET ASSISTANT RESPONSE
# ============================================================

@app.get("/assistant/{job_id}")

def assistant_job(

    job_id: str,

    authorization: str | None = Header(None)

):

    patient_id = patient_id_from_auth(
        authorization
    )


    c = conn()


    row = c.execute(

        """
        SELECT *
        FROM assistant_jobs
        WHERE id=?
        AND patient_id=?
        """,

        (

            job_id,

            patient_id

        )

    ).fetchone()


    c.close()


    if not row:

        raise HTTPException(

            404,

            "Assistant request not found."

        )


    return {

        "job_id":
            row["id"],

        "status":
            row["status"],

        "answer":
            row["answer"]

    }


# ============================================================
# FRONTEND STATIC FILES
# ============================================================
# Register this after every API route so API endpoints keep priority.
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend"
    )


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
