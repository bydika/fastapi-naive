from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import requests
import io

# Load model
# model = joblib.load('model_naive_bayes.pkl')

# Load model dari URL
def load_model():
    url = "https://ybbzadgwnxmwoalkkecn.supabase.co/storage/v1/object/public/news//model_naive_bayes.pkl"
    response = requests.get(url)
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model()

app = FastAPI()

# 🛠 PASTIKAN NAMA-NAMA FIELD INI SAMA DENGAN NAMA FITUR DI MODEL
class PasienInput(BaseModel):
    batuk_2_minggu: int
    batuk_berdarah: int
    demam_1_bulan: int
    sesak_napas_nyeri_dada: int
    nafsu_makan_turun: int
    berat_badan_turun: int
    keringat_malam: int

@app.get("/")
async def index():
    return "Home"

@app.post("/predict")
def predict_tbc(data: PasienInput):
    df = pd.DataFrame([data.dict()])

    # Opsional: pastikan urutan kolom sesuai
    df = df[[
        "batuk_2_minggu",
        "batuk_berdarah",
        "demam_1_bulan",
        "sesak_napas_nyeri_dada",
        "nafsu_makan_turun",
        "berat_badan_turun",
        "keringat_malam"
    ]]

    # Hitung probabilitas positif
    prob = model.predict_proba(df)[0][1] * 100
    hasil = "terduga" if prob >= 75 else "negatif"

    return {
        "persentase_positif": round(prob, 2),
        "hasil": hasil
    }

handler = app