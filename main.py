
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "academic_performance_classifier.joblib"

# Load the trained pipeline, including preprocessing
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Smart E-Learning Assistant API",
    description="Academic performance prediction using the OULAD-trained model.",
    version="1.0.0",
)


class StudentData(BaseModel):
    code_module: str = Field(examples=["AAA"])
    code_presentation: str = Field(examples=["2013J"])
    gender: str = Field(examples=["M"])
    region: str = Field(examples=["East Anglian Region"])
    highest_education: str = Field(examples=["A Level or Equivalent"])
    imd_band: str = Field(examples=["20-30%"])
    age_band: str = Field(examples=["0-35"])
    disability: str = Field(examples=["N"])

    date_registration: float = Field(examples=[-30])
    module_presentation_length: float = Field(examples=[268])
    num_of_prev_attempts: int = Field(ge=0, examples=[0])
    studied_credits: int = Field(gt=0, examples=[60])
    total_click: float = Field(ge=0, examples=[500])


@app.get("/")
def home():
    return {
        "message": "Smart E-Learning Assistant API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(student: StudentData):
    try:
        # Preserve the feature names and order expected by the model
        features = pd.DataFrame([student.model_dump()])
        prediction = str(model.predict(features)[0])

        return {
            "prediction": prediction,
            "message": (
                "The model predicts that the student may need additional support."
                if prediction == "Fail"
                else "The model predicts a passing result."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction failed. Check that categorical values match "
                "the OULAD training data. "
                f"Details: {str(exc)}"
            ),
        ) from exc