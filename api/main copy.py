from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import requests
import io
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load('model_manual_naive_bayes.pkl')

# Load model dari URL
# def load_model():
#     url = "https://ybbzadgwnxmwoalkkecn.supabase.co/storage/v1/object/public/news//model_naive_bayes.pkl"
#     response = requests.get(url)
#     model = joblib.load(io.BytesIO(response.content))
#     return model

# model = load_model()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti "*" dengan asal frontend kamu di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
# def predict_tbc(data: PasienInput):
#     df = pd.DataFrame([data.dict()])

#     # Opsional: pastikan urutan kolom sesuai
#     df = df[[
#         "batuk_2_minggu",
#         "batuk_berdarah",
#         "demam_1_bulan",
#         "sesak_napas_nyeri_dada",
#         "nafsu_makan_turun",
#         "berat_badan_turun",
#         "keringat_malam"
#     ]]

#     # Hitung probabilitas positif
#     prob = model.predict_proba(df)[0][1] * 100
#     hasil = "terduga" if prob >= 85 else "negatif"

#     return {
#         "persentase_positif": round(prob, 2),
#         "hasil": hasil
#     }
def predict_tbc(data: PasienInput):
    df = pd.DataFrame([data.dict()])

    # Pastikan urutan kolom sesuai
    df = df[[  
        "batuk_2_minggu",
        "batuk_berdarah",
        "demam_1_bulan",
        "sesak_napas_nyeri_dada",
        "nafsu_makan_turun",
        "berat_badan_turun",
        "keringat_malam"
    ]]

    # Gunakan model manual untuk prediksi probabilitas
    hasil_proba = predict_proba_manual(model, df)
    prob_positif = hasil_proba[1] * 100
    hasil = "terduga" if prob_positif >= 85 else "negatif"

    return {
        "persentase_positif": round(prob_positif, 2),
        "hasil": hasil
    }

def predict_proba_manual(model, df):
    row = df.iloc[0]
    class_scores = {}
    total = 0

    for c in model:
        log_prob = np.log(model[c]['prior'])  # log prior
        for feature in df.columns:
            prob = model[c]['probs'][feature]
            if row[feature] == 1:
                log_prob += np.log(prob)
            else:
                log_prob += np.log(1 - prob)
        class_scores[c] = np.exp(log_prob)
        total += class_scores[c]

    # Normalisasi
    for c in class_scores:
        class_scores[c] /= total

    return class_scores

handler = app