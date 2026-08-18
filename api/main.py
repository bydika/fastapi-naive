from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_COLUMNS = [
    "batuk_2_minggu",
    "batuk_berdarah",
    "demam_1_bulan",
    "sesak_nafas",
    "penurunan_nafsu",
    "penurunan_berat",
    "berkeringat",
]
POSITIVE_THRESHOLD = 85.0


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file tidak ditemukan di {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()

app = FastAPI(
    title="TBC Prediction API",
    version="1.0.0",
    description="API sederhana untuk prediksi indikasi TBC berbasis gejala pasien.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PasienInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batuk_2_minggu: int = Field(ge=0, le=1)
    batuk_berdarah: int = Field(ge=0, le=1)
    demam_1_bulan: int = Field(ge=0, le=1)
    sesak_nafas: int = Field(ge=0, le=1)
    penurunan_nafsu: int = Field(ge=0, le=1)
    penurunan_berat: int = Field(ge=0, le=1)
    berkeringat: int = Field(ge=0, le=1)


@app.get("/")
async def index() -> dict:
    return {
        "message": "TBC Prediction API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_tbc(data: PasienInput) -> dict:
    df = pd.DataFrame([data.model_dump()])[FEATURE_COLUMNS]
    probabilities = predict_proba_manual(model, df)

    if 1 not in probabilities:
        raise HTTPException(status_code=500, detail="Model tidak memiliki kelas positif.")

    positive_probability = round(probabilities[1] * 100, 2)
    result = "terduga" if positive_probability >= POSITIVE_THRESHOLD else "negatif"

    return {
        "persentase_positif": positive_probability,
        "hasil": result,
        "threshold": POSITIVE_THRESHOLD,
    }


def predict_proba_manual(loaded_model: Dict, df: pd.DataFrame) -> Dict[int, float]:
    row = df.iloc[0]
    class_scores: Dict[int, float] = {}
    total_score = 0.0

    for target_class, class_data in loaded_model.items():
        log_probability = np.log(class_data["prior"])
        for feature in FEATURE_COLUMNS:
            feature_probability = class_data["probs"][feature]
            log_probability += np.log(feature_probability if row[feature] == 1 else 1 - feature_probability)

        score = float(np.exp(log_probability))
        class_scores[target_class] = score
        total_score += score

    if total_score == 0:
        raise HTTPException(status_code=500, detail="Gagal menghitung probabilitas model.")

    for target_class in class_scores:
        class_scores[target_class] /= total_score

    return class_scores


handler = app
