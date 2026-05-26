import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

# Memuat environment variables (.env jika ada)
load_dotenv()

# Wajib mendefinisikan kembali Custom Layer agar Keras bisa memuat model dengan lancar
@tf.keras.utils.register_keras_serializable(package='Custom')
class FinancialFeatureGate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FinancialFeatureGate, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gate = self.add_weight(
            name='feature_gate',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.gate

    def get_config(self):
        config = super(FinancialFeatureGate, self).get_config()
        return config

# Konfigurasi Path Artefak
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gabungkan dengan folder artifacts secara benar
MODEL_PATH = os.path.join(CURRENT_DIR, 'artifacts', 'final_financial_health_classifier.keras')
SCALER_PATH = os.path.join(CURRENT_DIR, 'artifacts', 'financial_health_scaler.pkl')
LABEL_MAPPING_PATH = os.path.join(CURRENT_DIR, 'artifacts', 'cluster_to_label.pkl')

# Memuat Artefak Model dan Scaler
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'FinancialFeatureGate': FinancialFeatureGate}
    )
    scaler = joblib.load(SCALER_PATH)
    cluster_to_label = joblib.load(LABEL_MAPPING_PATH)
except Exception as error:
    raise RuntimeError(f'Gagal memuat artifact model atau scaler: {error}')

# Inisialisasi Gemini Client dengan SDK google-genai terbaru
gemini_api_key = os.getenv('GEMINI_API_KEY')
client = None
if gemini_api_key and gemini_api_key != 'API KEY GEMINI LAH POKOKNYA':
    client = genai.Client(api_key=gemini_api_key)

# Inisialisasi FastAPI
app = FastAPI(
    title='Financial Health Prediction API',
    description='REST API untuk memprediksi kesehatan finansial bulanan pengguna.',
    version='1.0.0'
)

# Mengaktifkan CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Pydantic Schema untuk Request Data
class FinancialHealthRequest(BaseModel):
    user_id: str = Field(..., example='user_001')
    month: int = Field(..., ge=1, le=12, example=6)
    year: int = Field(..., ge=2000, example=2026)
    income: float = Field(..., gt=0, example=5000000)
    rent: float = Field(..., ge=0, example=800000)
    loan_repayment: float = Field(..., ge=0, example=500000)
    insurance: float = Field(..., ge=0, example=200000)
    groceries: float = Field(..., ge=0, example=1200000)
    transport: float = Field(..., ge=0, example=400000)
    eating_out: float = Field(..., ge=0, example=300000)
    entertainment: float = Field(..., ge=0, example=250000)
    utilities: float = Field(..., ge=0, example=300000)
    healthcare: float = Field(..., ge=0, example=150000)
    education: float = Field(..., ge=0, example=0)
    miscellaneous: float = Field(..., ge=0, example=200000)

def calculate_financial_features(data: FinancialHealthRequest):
    total_expense = (
        data.rent + data.loan_repayment + data.insurance + data.groceries +
        data.transport + data.eating_out + data.entertainment + data.utilities +
        data.healthcare + data.education + data.miscellaneous
    )

    essential_expense = (
        data.rent + data.loan_repayment + data.insurance + data.groceries +
        data.utilities + data.healthcare
    )

    discretionary_spending = (
        data.eating_out + data.entertainment + data.miscellaneous
    )

    expense_to_income_ratio = total_expense / data.income
    essential_ratio = essential_expense / data.income
    disposable_income_ratio = (data.income - total_expense) / data.income
    loan_to_income_ratio = data.loan_repayment / data.income
    discretionary_ratio = discretionary_spending / data.income

    feature_data = pd.DataFrame([{
        'Expense_to_Income_Ratio': expense_to_income_ratio,
        'Essential_Ratio': essential_ratio,
        'Disposable_Income_Ratio': disposable_income_ratio,
        'Loan_to_Income_Ratio': loan_to_income_ratio,
        'Discretionary_Ratio': discretionary_ratio
    }])

    ratios = {
        'expense_to_income_ratio': expense_to_income_ratio,
        'essential_ratio': essential_ratio,
        'disposable_income_ratio': disposable_income_ratio,
        'loan_to_income_ratio': loan_to_income_ratio,
        'discretionary_ratio': discretionary_ratio
    }

    summary = {
        'income': data.income,
        'total_expense': total_expense,
        'disposable_income': data.income - total_expense,
        'essential_expense': essential_expense,
        'discretionary_spending': discretionary_spending
    }

    return feature_data, ratios, summary

def predict_with_model(feature_data: pd.DataFrame):
    input_scaled = scaler.transform(feature_data).astype(np.float32)
    pred_probabilities = model.predict(input_scaled, verbose=0)

    pred_class = int(np.argmax(pred_probabilities, axis=1)[0])
    pred_label = cluster_to_label[pred_class]
    confidence = float(np.max(pred_probabilities))

    probabilities = {
        cluster_to_label[i]: float(pred_probabilities[0][i])
        for i in range(len(cluster_to_label))
    }

    return pred_class, pred_label, confidence, probabilities

def generate_fallback_recommendation(prediction_label: str):
    if prediction_label == 'Financially Healthy':
        return (
            'Kondisi finansial Anda tergolong sehat. Pengeluaran masih terkendali '
            'dan sisa pendapatan cukup baik. Pertahankan pola ini dan pertimbangkan '
            'untuk meningkatkan alokasi tabungan.'
        )
    if prediction_label == 'Moderate':
        return (
            'Kondisi finansial Anda berada pada kategori sedang. Keuangan masih cukup aman, '
            'tetapi pengeluaran perlu mulai dikontrol agar tidak mendekati total pendapatan. '
            'Perhatikan kembali pengeluaran non-esensial.'
        )
    if prediction_label == 'At Risk':
        return (
            'Kondisi finansial Anda berada pada kategori berisiko. Sebagian besar pendapatan '
            'telah digunakan untuk pengeluaran. Prioritaskan kebutuhan utama, evaluasi cicilan, '
            'dan kurangi pengeluaran non-esensial.'
        )
    return 'Rekomendasi belum tersedia untuk kategori ini.'

def generate_ai_recommendation(prediction_label: str, confidence: float, ratios: dict, summary: dict):
    if client is None:
        return generate_fallback_recommendation(prediction_label), 'fallback_no_gemini_api_key'

    prompt = f"""
Kamu adalah asisten finansial pribadi yang ramah dan suportif untuk aplikasi pencatatan keuangan.

Tugasmu adalah membuat rekomendasi finansial singkat, hangat, dan profesional berdasarkan data pengguna berikut:

Hasil prediksi kesehatan finansial:
- Status: {prediction_label}

Ringkasan nominal bulanan:
- Income: Rp{summary['income']:,.0f}
- Total pengeluaran: Rp{summary['total_expense']:,.0f}
- Sisa income: Rp{summary['disposable_income']:,.0f} (Mewakili {sisa_pendapatan_persen:.0f}% dari total pendapatan)

Instruksi Gaya Bahasa & Format (PENTING):
1. Gunakan sudut pandang orang pertama ("Anda").
2. Buat teks maksimal 3-4 kalimat dalam 1 paragraf mengalir.
3. Bahasa harus sopan, memotivasi, ringan, dan mudah dipahami (seperti seorang konsultan keuangan pribadi yang ramah).
4. Sebutkan angka persentase sisa pendapatan yang berhasil disisihkan (yaitu {sisa_pendapatan_persen:.0f}%) sebagai bentuk apresiasi atau evaluasi. Jangan sebutkan angka nominal Rp lainnya agar teks tetap bersih.
5. Berikan saran logis (seperti tabungan darurat atau menekan pengeluaran) sesuai status kesehatannya.

Contoh gaya bahasa yang diinginkan jika statusnya bagus:
"Kondisi keuangan Anda bulan ini terlihat sangat baik, dengan pengeluaran yang terkontrol membuat Anda memiliki ruang finansial yang luas. Ini menunjukkan manajemen keuangan yang sangat efisien, di mana sebagian besar pendapatan Anda, yaitu {sisa_pendapatan_persen:.0f}%, berhasil Anda sisihkan. Pertahankan kebiasaan baik ini, dan manfaatkan sisa dana Anda untuk mempercepat pencapaian tujuan finansial seperti tabungan darurat atau rencana masa depan yang lebih besar."
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        recommendation = response.text.strip()
        if not recommendation:
            return generate_fallback_recommendation(prediction_label), 'fallback_empty_gemini_response'
        return recommendation, 'generative_ai_gemini'
    except Exception as error:
        return generate_fallback_recommendation(prediction_label), f'fallback_gemini_error: {str(error)}'

@app.get('/')
def root():
    return {
        'status': 'success',
        'message': 'Financial Health Prediction API is running.'
    }

@app.get('/health')
def health_check():
    return {
        'status': 'success',
        'message': 'API and model are ready.'
    }

@app.post('/predict-financial-health')
def predict_financial_health(data: FinancialHealthRequest):
    try:
        feature_data, ratios, summary = calculate_financial_features(data)
        pred_class, pred_label, confidence, probabilities = predict_with_model(feature_data)
        recommendation, recommendation_source = generate_ai_recommendation(
            prediction_label=pred_label,
            confidence=confidence,
            ratios=ratios,
            summary=summary
        )
        return {
            'status': 'success',
            'user_id': data.user_id,
            'month': data.month,
            'year': data.year,
            'prediction': {
                'class_id': pred_class,
                'label': pred_label,
                'confidence': confidence
            },
            'probabilities': probabilities,
            'ratios': ratios,
            'summary': summary,
            'recommendation': recommendation,
            'recommendation_source': recommendation_source
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f'Terjadi kesalahan saat melakukan prediksi: {str(error)}'
        )
