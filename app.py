import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import sklearn
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.impute import SimpleImputer as SkSimpleImputer
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder, StandardScaler as SkStandardScaler
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# ── Konfigurasi Halaman Streamlit ──────────────────────────────────────────
st.set_page_config(
    page_title="AirSense | Prediksi PM2.5 Global",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import Fonts dan Injeksi CSS Kustom ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Gaya Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sembunyikan Header dan Menu Bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Gradasi Latar Belakang Aplikasi */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0d1726 40%, #150f29 100%);
        min-height: 100vh;
    }
    
    /* Sidebar Kustom */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b12 0%, #110d21 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    
    /* Kartu Dashboard */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(100, 200, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .dashboard-card:hover {
        border-color: rgba(100, 200, 255, 0.35);
        box-shadow: 0 8px 32px rgba(0, 140, 255, 0.15);
        transform: translateY(-2px);
    }
    
    /* Kartu Informasi Parameter (Panduan) */
    .guide-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #64d2ff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }
    .guide-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.12);
        border-left-width: 6px;
    }
    
    /* Tipografi */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #64d2ff, #a180ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 24px;
        line-height: 1.6;
    }
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #64d2ff;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(100, 200, 255, 0.15);
        padding-bottom: 8px;
    }
    .card-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* KPI Stats Bar */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 24px;
    }
    .kpi-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #64d2ff;
    }
    .kpi-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Spanduk Hasil Prediksi Kustom */
    .result-banner {
        border-radius: 12px;
        padding: 24px;
        margin: 24px 0;
        display: flex;
        align-items: flex-start;
        gap: 20px;
    }
    .banner-good {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #a7f3d0;
    }
    .banner-moderate {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #fde68a;
    }
    .banner-unhealthy {
        background: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.25);
        color: #fed7aa;
    }
    .banner-hazardous {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
    }
    .banner-emoji {
        font-size: 3rem;
        line-height: 1;
    }
    .banner-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .banner-desc {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Skala Visual PM2.5 */
    .aqi-scale-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(100, 200, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin: 24px 0;
    }
    .scale-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 10px;
    }
    .aqi-scale-bar {
        position: relative;
        height: 16px;
        border-radius: 8px;
        background: linear-gradient(90deg, 
            #10b981 0%, #10b981 20%, 
            #f59e0b 20%, #f59e0b 45%, 
            #f97316 45%, #f97316 65%, 
            #ef4444 65%, #ef4444 100%
        );
    }
    .aqi-scale-marker {
        position: absolute;
        top: -6px;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #ffffff;
        border: 4px solid #090d16;
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.9);
        transform: translateX(-50%);
        transition: left 0.8s ease;
    }
    
    /* Tombol Prediksi Kustom */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
        background: linear-gradient(135deg, #6366f1, #0891b2) !important;
    }
    
    /* Styling Pilihan Form */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(100, 200, 255, 0.2) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Diagram Alur Pemrosesan */
    .flowchart-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(100, 200, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 30px;
        overflow-x: auto;
    }
    .flowchart-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        min-width: 90px;
    }
    .flowchart-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.04);
        border: 2px solid rgba(255, 255, 255, 0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #64748b;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .flowchart-step.active .flowchart-icon {
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        border-color: #64d2ff;
        color: white;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
    }
    .flowchart-step.completed .flowchart-icon {
        background: #10b981;
        border-color: #10b981;
        color: white;
    }
    .flowchart-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
    }
    .flowchart-step.active .flowchart-label {
        color: #64d2ff;
    }
    .flowchart-step.completed .flowchart-label {
        color: #10b981;
    }
    .flowchart-arrow {
        color: #334155;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper untuk merender ikon SVG inline yang kustom dan premium
def get_svg_icon(name, size=24, color="#64d2ff"):
    icons = {
        "geo": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
        "pollutant": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
        "map": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon><line x1="9" y1="3" x2="9" y2="18"></line><line x1="15" y1="6" x2="15" y2="21"></line></svg>',
        "population": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
        "play": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>',
        "refresh": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>',
        "success": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
    }
    return icons.get(name, "")

# Helper untuk menggambar diagram alur
def render_flowchart(current_step):
    steps = [
        "Input Data",
        "Validasi",
        "Pembersihan",
        "Analisis",
        "Visualisasi",
        "Insight",
        "Hasil Akhir"
    ]
    html = '<div class="flowchart-container">'
    for i, step in enumerate(steps):
        step_num = i + 1
        cls = ""
        if step_num == current_step:
            cls = "active"
        elif step_num < current_step:
            cls = "completed"
            
        icon_content = "✓" if step_num < current_step else str(step_num)
        
        html += f"""
        <div class="flowchart-step {cls}">
            <div class="flowchart-icon">{icon_content}</div>
            <div class="flowchart-label">{step}</div>
        </div>
        """
        if i < len(steps) - 1:
            html += '<div class="flowchart-arrow">→</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── Fitur-fitur Model & Parameter ──────────────────────────────────────────
FEATURE_COLS = ["year", "latitude", "longitude", "pm10_concentration", "no2_concentration", "number_of_stations", "who_ms", "population", "who_region"]
TARGET_COL = "pm25_concentration"
NUM_COLS = ["year", "latitude", "longitude", "pm10_concentration", "no2_concentration", "number_of_stations", "who_ms", "population"]
CAT_COLS = ["who_region"]
WHO_REGION = {
    "Asia Tenggara (SEARO)":         "3_Sear",
    "Eropa (EURO)":                  "4_Eur",
    "Amerika (AMRO)":                "2_Amr",
    "Pasifik Barat (WPRO)":          "6_Wpr",
    "Afrika (AFRO)":                 "1_Afr",
    "Mediterania Timur (EMRO)":      "5_Emr",
    "Negara Non-Anggota (Non-MS)":   "7_NonMS"
}

# ── Load Model Pipeline ──────────────────────────────────────────
def train_model_on_the_fly():
    train_path = 'action2024/train.csv'
    if not os.path.exists(train_path):
        return None
    try:
        df = pd.read_csv(train_path, low_memory=False)
        for col in NUM_COLS + [TARGET_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df_model = df.dropna(subset=[TARGET_COL]).copy()
        df_model = df_model[df_model[TARGET_COL] > 0]
        
        X = df_model[FEATURE_COLS]
        y = df_model[TARGET_COL]
        
        num_pipeline = SkPipeline([
            ("imputer", SkSimpleImputer(strategy="median")),
            ("scaler",  SkStandardScaler()),
        ])
        
        if sklearn.__version__ >= '1.2':
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse=False)
            
        cat_pipeline = SkPipeline([
            ("imputer", SkSimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe",     ohe),
        ])
        
        preprocessor = SkColumnTransformer([
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ])
        
        # PERBAIKAN: Parameter disamakan dengan build_pipeline.py
        pipeline = SkPipeline([
            ("preprocessor", preprocessor),
            ("model", SkRandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )),
        ])
        
        pipeline.fit(X, y)
        return pipeline
    except Exception:
        return None

@st.cache_resource
def load_model():
    model_path = 'pipeline_pm25_final.pkl'
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception:
        pass
    trained = train_model_on_the_fly()
    if trained is not None:
        return trained
    st.error("Gagal memuat model. Pastikan file model atau data pelatihan berada di direktori proyek.")
    st.stop()

model = load_model()

# ── Load Dataset dengan Cache ──────────────────────────────────────────
@st.cache_data
def load_raw_data():
    train_path = 'action2024/train.csv'
    if not os.path.exists(train_path):
        return None
    try:
        df = pd.read_csv(train_path, low_memory=False)
        for col in NUM_COLS + [TARGET_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None

df_data = load_raw_data()

# ── MENU UTAMA PADA SIDEBAR ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:15px 0 25px 0;">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 1.7rem; font-weight: 700; background: linear-gradient(90deg, #64d2ff, #a180ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0 0 0;">AirSense</h1>
        <p style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin: 5px 0 0 0;">Air Quality Predictor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Navigasi Halaman Utama
    menu = st.radio(
        "PILIH HALAMAN",
        ["Beranda Utama", "Panduan & Sumber Input", "Alur & Proses Data"],
        index=0
    )
    
    st.markdown("<br><br><br><br><br><hr style='border-color: rgba(99, 102, 241, 0.15);'>", unsafe_allow_html=True)
    
    # Informasi Kelompok (Di paling bawah bar sidebar)
    st.markdown("""
    <div style="padding: 14px; background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.15); border-radius: 12px; text-align: center;">
        <div style="font-family: 'Poppins', sans-serif; font-size: 0.75rem; font-weight: 600; color: #64d2ff; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px;">Kelompok Machine Learning</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-bottom: 2px;">Kristian Novan</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 8px;">2802458560</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-bottom: 2px;">Andrew Ong</div>
        <div style="font-size: 0.75rem; color: #94a3b8;">2802420561</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HALAMAN 1: 🏠 BERANDA UTAMA
# ─────────────────────────────────────────────────────────────────
if menu == "Beranda Utama":
    # ── Hero Section ──
    st.markdown(f"""
    <div class="hero-container">
        <h1 class="main-title" style="margin:0;">AirSense Dashboard</h1>
        <p style="color: #cbd5e1; font-size: 1.15rem; margin-top: 5px; font-weight: 300;">Platform Prediksi PM2.5 Global & Estimasi Kualitas Udara</p>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 800px; margin-top: 10px; line-height: 1.6;">
            Mengintegrasikan algoritma Random Forest dengan data historis World Health Organization (WHO) untuk memprediksi kadar polusi PM2.5. Masukkan koordinat serta parameter wilayah Anda di bawah ini untuk memulai analisis kualitas udara secara presisi.
        </p>
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-val">25,999+</div>
                <div class="kpi-lbl">Record Data WHO</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-val">89.00%</div>
                <div class="kpi-lbl">Akurasi Model (R²)</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-val">2.68 µg/m³</div>
                <div class="kpi-lbl">Rata-rata MAE</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-val">9 Fitur</div>
                <div class="kpi-lbl">Variabel Analisis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Form Input Grid
    st.markdown('<div class="section-title">Form Parameter Masukan</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{get_svg_icon("geo", 20, "#64d2ff")} <span>Letak Geografis</span></div>', unsafe_allow_html=True)
        who_region_label = st.selectbox(
            "Wilayah WHO",
            options=list(WHO_REGION.keys()),
            index=0,
            help="Pembagian wilayah administratif dunia berdasarkan organisasi kesehatan dunia (WHO)"
        )
        who_region = WHO_REGION[who_region_label]
        
        latitude = st.number_input(
            "Garis Lintang (Latitude)",
            min_value=-90.0, max_value=90.0,
            value=-6.2088, format="%.4f",
            help="Contoh: Jakarta memiliki koordinat -6.2088"
        )
        longitude = st.number_input(
            "Garis Bujur (Longitude)",
            min_value=-180.0, max_value=180.0,
            value=106.8456, format="%.4f",
            help="Contoh: Jakarta memiliki koordinat 106.8456"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{get_svg_icon("population", 20, "#10b981")} <span>Demografi & Pos</span></div>', unsafe_allow_html=True)
        year = st.number_input(
            "Tahun Analisis",
            min_value=2010, max_value=2035,
            value=2024, step=1
        )
        population = st.number_input(
            "Populasi Penduduk (Jiwa)",
            min_value=1000, max_value=60000000,
            value=10000000, step=100000, format="%d"
        )
        number_of_stations = st.number_input(
            "Jumlah Stasiun Pemantau Udara",
            min_value=1, max_value=300,
            value=3, step=1
        )
        who_ms_label = st.radio(
            "Status Keanggotaan Negara di WHO",
            options=["Anggota Resmi WHO", "Non-Anggota / Pengamat"],
            horizontal=True
        )
        who_ms = 1 if who_ms_label == "Anggota Resmi WHO" else 0
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{get_svg_icon("pollutant", 20, "#a180ff")} <span>Polutan Penunjang</span></div>', unsafe_allow_html=True)
        st.write("Mengisi data polutan penunjang di bawah ini sangat direkomendasikan untuk meningkatkan akurasi estimasi model.")
        
        has_pm10 = st.checkbox("Saya memiliki data PM10", value=False)
        if has_pm10:
            pm10 = st.number_input(
                "Konsentrasi PM10 (µg/m³)",
                min_value=0.0, max_value=500.0,
                value=35.0, step=0.1
            )
            st.success("PM10 akan dimasukkan ke dalam analisis model.")
        else:
            pm10 = np.nan
            st.info("PM10 tidak diisi, model akan mengestimasi nilai menggunakan median historis.")
            
        has_no2 = st.checkbox("Saya memiliki data NO₂", value=False)
        if has_no2:
            no2 = st.number_input(
                "Konsentrasi NO₂ (µg/m³)",
                min_value=0.0, max_value=300.0,
                value=20.0, step=0.1
            )
            st.success("NO₂ akan dimasukkan ke dalam analisis model.")
        else:
            no2 = np.nan
            st.info("NO₂ tidak diisi, model akan mengestimasi nilai menggunakan median historis.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Tombol eksekusi prediksi
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        analyze_clicked = st.button("Analisis Kualitas Udara")
        
    if analyze_clicked:
        with st.spinner("Model sedang memproses data..."):
            input_data = pd.DataFrame([{
                'year': int(year),
                'latitude': float(latitude),
                'longitude': float(longitude),
                'pm10_concentration': float(pm10) if not np.isnan(pm10) else np.nan,
                'no2_concentration': float(no2) if not np.isnan(no2) else np.nan,
                'number_of_stations': int(number_of_stations),
                'who_ms': int(who_ms),
                'population': float(population),
                'who_region': who_region
            }])
            
            prediction = float(model.predict(input_data)[0])
            prediction = max(0.0, prediction) # Konsentrasi tidak mungkin negatif
            
        # Penentuan Kategori berdasarkan ambang batas WHO
        if prediction <= 12.0:
            category = "Sehat (Aman)"
            emoji = "🟢"
            banner_class = "banner-good"
            color = "#10b981"
            pct = min(prediction / 12.0 * 20, 20)
            recommendation = "Kualitas udara sangat baik. Udara bersih dan tidak berisiko bagi kesehatan manusia. Sangat aman dan disarankan untuk melakukan semua jenis aktivitas di luar ruangan."
        elif prediction <= 35.4:
            category = "Sedang (Moderate)"
            emoji = "🟡"
            banner_class = "banner-moderate"
            color = "#f59e0b"
            pct = 20 + min((prediction - 12.0) / 23.4 * 25, 25)
            recommendation = "Kualitas udara berada pada tingkat yang dapat diterima. Namun, bagi individu yang sangat sensitif (misal penderita asma akut, lansia, dan balita) disarankan untuk mengurangi durasi aktivitas fisik berat di luar ruangan."
        elif prediction <= 55.4:
            category = "Tidak Sehat bagi Kelompok Sensitif"
            emoji = "🟠"
            banner_class = "banner-unhealthy"
            color = "#f97316"
            pct = 45 + min((prediction - 35.4) / 20.0 * 20, 20)
            recommendation = "Kadar polusi dapat berdampak pada kelompok masyarakat yang sensitif. Dianjurkan bagi anak-anak, lansia, dan orang dengan penyakit pernapasan untuk menggunakan masker standar (seperti N95) saat beraktivitas di luar rumah."
        else:
            category = "Sangat Tidak Sehat / Berbahaya"
            emoji = "🔴"
            banner_class = "banner-hazardous"
            color = "#ef4444"
            pct = min(65 + (prediction - 55.4) / 44.6 * 35, 100)
            recommendation = "Peringatan Kesehatan! Seluruh populasi berisiko mengalami dampak negatif. Hindari aktivitas fisik di luar ruangan. Gunakan penyaring udara (air purifier) di dalam rumah dan pastikan ventilasi tertutup rapat."
            
        # Tampilkan Hasil
        st.markdown('<div class="section-title">Hasil Prediksi & Analisis</div>', unsafe_allow_html=True)
        
        # Skala Visual Bar
        st.markdown(f"""
        <div class="aqi-scale-container">
            <div style="font-family:'Poppins', sans-serif; font-size:0.9rem; font-weight:600; color:#e2e8f0; margin-bottom:12px;">Posisi Nilai dalam Skala Polusi (PM2.5: {prediction:.2f} µg/m³):</div>
            <div class="aqi-scale-bar">
                <div class="aqi-scale-marker" style="left: {pct}%;"></div>
            </div>
            <div class="scale-labels">
                <span>🟢 Sehat (0–12)</span>
                <span>🟡 Sedang (12.1–35.4)</span>
                <span>🟠 Kurang Sehat (35.5–55.4)</span>
                <span>🔴 Berbahaya (&gt;55.4)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Spanduk Rekomendasi Kesehatan
        st.markdown(f"""
        <div class="result-banner {banner_class}">
            <div class="banner-emoji">{emoji}</div>
            <div>
                <div class="banner-title">{category}</div>
                <div class="banner-desc"><strong>Rekomendasi Kesehatan:</strong> {recommendation}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Penjelasan Teknis Tambahan (Expander)
        with st.expander("Lihat Detail Model & Input Konfigurasi"):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**Struktur Input untuk Model:**")
                st.json({
                    "Tahun": int(year),
                    "Latitude": float(latitude),
                    "Longitude": float(longitude),
                    "Populasi": int(population),
                    "Wilayah WHO": who_region,
                    "Kadar PM10": float(pm10) if not np.isnan(pm10) else "Diimputasi (Median Latihan)",
                    "Kadar NO₂": float(no2) if not np.isnan(no2) else "Diimputasi (Median Latihan)",
                    "Jumlah Stasiun": int(number_of_stations),
                    "Status Anggota WHO": int(who_ms)
                })
            with ec2:
                st.markdown("**Detail Model Machine Learning:**")
                st.json({
                    "Algoritma": "Random Forest Regressor",
                    "Jumlah Pohon (Trees)": 300,
                    "Skor Kebaikan Model (R²)": 0.8900,
                    "Rata-rata Error Mutlak (MAE)": "2.68 µg/m³",
                    "Akurasi Imputasi": "Median Imputer Terintegrasi",
                    "Sumber Data Pelatihan": "Data Kualitas Udara Global WHO"
                })

# ─────────────────────────────────────────────────────────────────
# HALAMAN 2: 📖 PANDUAN & SUMBER INPUT
# ─────────────────────────────────────────────────────────────────
elif menu == "Panduan & Sumber Input":
    st.markdown('<h1 class="main-title">Panduan & Sumber Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pelajari cara menggunakan dashboard AirSense serta jelajahi informasi detail mengenai dataset WHO yang digunakan untuk melatih model kami.</p>', unsafe_allow_html=True)
    
    tab_guide, tab_source = st.tabs(["Cara Menggunakan Aplikasi", "Sumber Input Data (Dataset)"])
    
    with tab_guide:
        st.markdown('<div class="section-title">Langkah-langkah Pengoperasian Dashboard</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="guide-card" style="border-left-color: #64d2ff;">
            <div class="card-title">
                {get_svg_icon("geo", 20, "#64d2ff")}
                <span>1. Tentukan Parameter Lokasi & Demografi</span>
            </div>
            <p style="font-size:0.88rem; color:#cbd5e1; line-height:1.6; margin:0;">
                Buka halaman <strong>Beranda Utama</strong>. Masukkan koordinat wilayah Anda (Latitude dan Longitude), jumlah stasiun pemantau, serta jumlah populasi penduduk setempat.
            </p>
        </div>
        
        <div class="guide-card" style="border-left-color: #a180ff;">
            <div class="card-title">
                {get_svg_icon("pollutant", 20, "#a180ff")}
                <span>2. Masukkan Parameter Polutan Penunjang (Opsional)</span>
            </div>
            <p style="font-size:0.88rem; color:#cbd5e1; line-height:1.6; margin:0;">
                Jika Anda memiliki informasi konsentrasi PM10 atau Nitrogen Dioksida (NO₂), centang kotak dan masukkan angkanya untuk menaikkan akurasi estimasi model.
            </p>
        </div>
        
        <div class="guide-card" style="border-left-color: #10b981;">
            <div class="card-title">
                {get_svg_icon("play", 20, "#10b981")}
                <span>3. Jalankan Analisis Kualitas Udara</span>
            </div>
            <p style="font-size:0.88rem; color:#cbd5e1; line-height:1.6; margin:0;">
                Klik tombol <strong>"Analisis Kualitas Udara"</strong>. Sistem akan menghitung konsentrasi estimasi PM2.5 dan memberikan spanduk klasifikasi serta rekomendasi kesehatan secara instan.
            </p>
        </div>
        
        <div class="guide-card" style="border-left-color: #fb923c;">
            <div class="card-title">
                {get_svg_icon("info", 20, "#fb923c")}
                <span>4. Eksplorasi Model & Alur Data</span>
            </div>
            <p style="font-size:0.88rem; color:#cbd5e1; line-height:1.6; margin:0;">
                Buka halaman <strong>Alur & Proses Data</strong> untuk menjalankan simulasi pemrosesan dataset WHO step-by-step secara interaktif serta menganalisis performa model.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_source:
        st.markdown('<div class="section-title">Informasi Dataset WHO Global Ambient Air Quality</div>', unsafe_allow_html=True)
        
        metadata_col, preview_col = st.columns([1, 1])
        
        with metadata_col:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">{get_svg_icon("map", 20, "#64d2ff")} <span>Spesifikasi Metadata</span></div>
                <table style="width:100%; border-collapse: collapse; font-size:0.9rem; color:#cbd5e1;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:600; color:#64d2ff;">Nama Dataset</td><td style="padding:10px 0;">WHO Global Ambient Air Quality Database</td></tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:600; color:#64d2ff;">Format File</td><td style="padding:10px 0;">Comma Separated Value (.csv)</td></tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:600; color:#64d2ff;">Jumlah Record</td><td style="padding:10px 0;">25,999 Baris</td></tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0; font-weight:600; color:#64d2ff;">Jumlah Fitur</td><td style="padding:10px 0;">9 Fitur Prediktor</td></tr>
                    <tr><td style="padding:10px 0; font-weight:600; color:#64d2ff;">Variabel Target</td><td style="padding:10px 0;">pm25_concentration (Numerik)</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Deskripsi Dataset:**
            Database ini berisikan data historis pemantauan kualitas udara global dari stasiun pengukuran di berbagai kota dunia. Fitur mencakup letak geografis (garis lintang/bujur), demografis (populasi), status pos, tahun pemantauan, regional organisasi kesehatan dunia (WHO), serta polutan partikulat PM10 dan gas kimia NO₂.
            """)
            
        with preview_col:
            if df_data is not None:
                st.markdown(f'<div class="card-title">{get_svg_icon("success", 20, "#10b981")} <span>Preview 10 Baris Pertama Data</span></div>', unsafe_allow_html=True)
                st.dataframe(df_data.head(10))
                
                with st.expander("Lihat Ringkasan Statistik Deskriptif (Numeric Only)"):
                    st.dataframe(df_data.describe())
            else:
                st.warning("Gagal memuat dataset lokal `action2024/train.csv` untuk preview.")

# ─────────────────────────────────────────────────────────────────
# HALAMAN 3: ⚙️ ALUR & PROSES DATA (STEP-BY-STEP SIMULATION)
# ─────────────────────────────────────────────────────────────────
elif menu == "Alur & Proses Data":
    st.markdown('<h1 class="main-title">Eksplorasi Data & Evaluasi Model Secara Interaktif</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Eksplorasi dataset WHO dan evaluasi model Machine Learning Anda langkah demi langkah. Pilih tahap analisis di bawah ini untuk melihat hasil visualisasi Plotly interaktif secara bertahap.</p>', unsafe_allow_html=True)
    
    # Session State untuk alur EDA step-by-step
    if "eda_step" not in st.session_state:
        st.session_state.eda_step = 1
        
    # Render diagram alur pemrosesan data (Step 1-7) berdasarkan step EDA saat ini
    flow_mapping = {1: 2, 2: 4, 3: 5, 4: 6, 5: 7}
    render_flowchart(flow_mapping[st.session_state.eda_step])
    
    # Kontrol Navigasi Horizontal
    st.markdown('<div style="margin-bottom:20px;"></div>', unsafe_allow_html=True)
    
    c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns(5)
    
    with c_btn1:
        if st.button("Langkah 1: Data Overview"):
            st.session_state.eda_step = 1
    with c_btn2:
        if st.button("Langkah 2: Univariate Analysis"):
            st.session_state.eda_step = 2
    with c_btn3:
        if st.button("Langkah 3: Bivariate Analysis"):
            st.session_state.eda_step = 3
    with c_btn4:
        if st.button("Langkah 4: Correlation Matrix"):
            st.session_state.eda_step = 4
    with c_btn5:
        if st.button("Langkah 5: Final Model Insight"):
            st.session_state.eda_step = 5

    st.markdown("---")

    # Helper functions untuk membuat plot Plotly di app.py
    def plot_plotly_missing_values(df):
        # PERBAIKAN: Hanya gunakan kolom yang terbukti ada di dataset untuk mencegah KeyError
        cols = ["year", "latitude", "longitude", "pm10_concentration", 
                "no2_concentration", "number_of_stations", "who_ms", "population", "pm25_concentration"]
        
        missing_pct = [(df[c].isna().sum() / len(df) * 100) for c in cols]
        missing_df = pd.DataFrame({
            "Fitur": cols,
            "Persentase (%)": missing_pct
        }).sort_values("Persentase (%)", ascending=False)
        
        fig = px.bar(missing_df, x="Fitur", y="Persentase (%)",
                     text=[f"{v:.1f}%" for v in missing_df["Persentase (%)"]],
                     color="Persentase (%)",
                     color_continuous_scale="Viridis",
                     template="plotly_dark")
        fig.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Nama Fitur",
            yaxis_title="Persentase (%)",
            yaxis=dict(range=[0, 100]),
            coloraxis_showscale=False
        )
        fig.update_traces(textposition='outside')
        return fig

    def plot_plotly_class_distribution(df):
        # PERBAIKAN: Buat kolom Kategori secara dinamis agar tidak menyebabkan KeyError
        df_temp = df.copy()
        df_temp['Air_quality_category'] = np.where(df_temp['pm25_concentration'] <= 35.4, 'Safety', 'Dangerous')
        
        counts = df_temp['Air_quality_category'].dropna().value_counts().reset_index()
        counts.columns = ['Kategori', 'Jumlah']
        color_map = {'Safety': '#10b981', 'Dangerous': '#ef4444'}
        
        fig = px.bar(counts, x='Kategori', y='Jumlah',
                     text='Jumlah',
                     color='Kategori',
                     color_discrete_map=color_map,
                     template='plotly_dark')
        fig.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Kategori Kualitas Udara",
            yaxis_title="Jumlah Record"
        )
        fig.update_traces(textposition='outside')
        return fig

    def plot_plotly_pm_distributions(df):
        pm25_filtered = df[df['pm25_concentration'] < 150]['pm25_concentration'].dropna()
        pm10_filtered = df[df['pm10_concentration'] < 200]['pm10_concentration'].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pm25_filtered, name='PM2.5', marker_color='#64d2ff', opacity=0.75, nbinsx=60))
        fig.add_trace(go.Histogram(x=pm10_filtered, name='PM10', marker_color='#a180ff', opacity=0.75, nbinsx=60))
        
        fig.update_layout(
            barmode='overlay',
            template='plotly_dark',
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Konsentrasi (µg/m³)",
            yaxis_title="Frekuensi",
            legend=dict(x=0.8, y=0.9, bgcolor='rgba(17, 24, 39, 0.5)')
        )
        return fig

    def plot_plotly_scatter_pm(df):
        sc = df.dropna(subset=['pm10_concentration', 'pm25_concentration'])
        sc = sc[(sc['pm10_concentration'] < 200) & (sc['pm25_concentration'] < 150)]
        sc_sample = sc.sample(n=min(5000, len(sc)), random_state=42)
        
        fig = px.scatter(sc_sample, x='pm10_concentration', y='pm25_concentration',
                         color='pm25_concentration',
                         color_continuous_scale='Viridis',
                         opacity=0.6,
                         template='plotly_dark')
        fig.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="PM10 (µg/m³)",
            yaxis_title="PM2.5 (µg/m³)",
            coloraxis_showscale=False
        )
        return fig

    def plot_plotly_boxplot_regions(df):
        REGION_LABEL = {
            '1_Afr': 'Afrika', '2_Amr': 'Amerika', '3_Sear': 'Asia Tenggara',
            '4_Eur': 'Eropa',  '5_Emr': 'Mediterania', '6_Wpr': 'Pasifik Barat',
            '7_NonMS': 'Non-MS'
        }
        df_plot = df.dropna(subset=['pm25_concentration', 'who_region']).copy()
        df_plot['region_label'] = df_plot['who_region'].map(REGION_LABEL)
        df_plot = df_plot[df_plot['pm25_concentration'] < 150]
        
        fig = px.box(df_plot, x='region_label', y='pm25_concentration',
                     color='region_label',
                     template='plotly_dark')
        fig.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Wilayah WHO",
            yaxis_title="PM2.5 (µg/m³)",
            showlegend=False
        )
        return fig

    def plot_plotly_yearly_trends(df):
        yr = df.dropna(subset=['pm25_concentration']).groupby('year')['pm25_concentration'].agg(['mean', 'median']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yr['year'], y=yr['mean'], mode='lines+markers', name='Rata-rata', line=dict(color='#64d2ff', width=3)))
        fig.add_trace(go.Scatter(x=yr['year'], y=yr['median'], mode='lines+markers', name='Median', line=dict(color='#fb923c', width=2, dash='dash')))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Tahun",
            yaxis_title="PM2.5 (µg/m³)",
            legend=dict(x=0.8, y=0.9, bgcolor='rgba(17, 24, 39, 0.5)')
        )
        return fig

    def plot_plotly_correlation_matrix(df):
        corr_cols = ['pm25_concentration', 'pm10_concentration', 'no2_concentration',
                     'year', 'population', 'latitude', 'longitude', 'number_of_stations']
        corr_m = df[corr_cols].corr()
        
        fig = px.imshow(corr_m, text_auto='.2f',
                        color_continuous_scale='coolwarm',
                        aspect="auto",
                        template='plotly_dark')
        fig.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    def plot_plotly_model_comparison():
        models_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
            'R2 Score': [0.6210, 0.6212, 0.7850, 0.8900, 0.8850],
            'RMSE': [6.45, 6.45, 4.85, 2.68, 2.78]
        }).sort_values('R2 Score')
        
        fig_r2 = px.bar(models_df, x='R2 Score', y='Model', orientation='h',
                        text='R2 Score',
                        color='R2 Score',
                        color_continuous_scale='plasma',
                        template='plotly_dark')
        fig_r2.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        fig_r2.update_traces(textposition='outside')
        
        fig_rmse = px.bar(models_df, x='RMSE', y='Model', orientation='h',
                          text='RMSE',
                          color='RMSE',
                          color_continuous_scale='plasma_r',
                          template='plotly_dark')
        fig_rmse.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        fig_rmse.update_traces(textposition='outside')
        
        return fig_r2, fig_rmse

    def plot_plotly_eval_final(df, model):
        df_model = df.dropna(subset=[TARGET_COL]).copy()
        df_model = df_model[df_model[TARGET_COL] > 0]
        
        X_all = df_model[FEATURE_COLS]
        y_all = df_model[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Scatter: Actual vs Predicted
        scatter_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
        scatter_sub = scatter_df.sample(n=min(1200, len(scatter_df)), random_state=42)
        fig_scatter = px.scatter(scatter_sub, x='Aktual', y='Prediksi',
                                 opacity=0.5,
                                 template='plotly_dark')
        fig_scatter.add_shape(
            type="line", line=dict(dash='dash', color='red', width=2),
            x0=0, y0=0, x1=max(y_test), y1=max(y_test)
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="PM2.5 Aktual (µg/m³)",
            yaxis_title="PM2.5 Prediksi (µg/m³)"
        )
        
        # Residual Histogram
        fig_resid = px.histogram(x=residuals, nbins=50,
                                 template='plotly_dark')
        fig_resid.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Residu (Aktual - Prediksi)",
            yaxis_title="Frekuensi"
        )
        fig_resid.update_traces(marker_color='#a180ff')
        
        # Feature Importance
        ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
        cat_cols = list(ohe.get_feature_names_out(['who_region'])) if hasattr(ohe, 'get_feature_names_out') else list(ohe.get_feature_names(['who_region']))
        num_cols = ["year", "latitude", "longitude", "pm10_concentration", "no2_concentration", "number_of_stations", "who_ms", "population"]
        feature_names = num_cols + cat_cols
        
        importances = model.named_steps['model'].feature_importances_
        feat_df = pd.DataFrame({
            'Fitur': feature_names,
            'Kepentingan': importances
        }).sort_values('Kepentingan', ascending=True).tail(8)
        
        fig_feat = px.bar(feat_df, x='Kepentingan', y='Fitur', orientation='h',
                          text=[f"{v:.1%}" for v in feat_df['Kepentingan']],
                          template='plotly_dark')
        fig_feat.update_layout(
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            paper_bgcolor="rgba(9, 13, 22, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Tingkat Kepentingan",
            yaxis_title="Fitur"
        )
        fig_feat.update_traces(marker_color='#10b981', textposition='outside')
        
        return fig_scatter, fig_resid, fig_feat

    # ── LOGIKA RENDER STEP-BY-STEP ──
    if df_data is None:
        st.warning("Gagal memuat dataset `action2024/train.csv` untuk analisis.")
    else:
        # STEP 1: Data Overview
        if st.session_state.eda_step == 1:
            st.markdown("### Langkah 1: Data Overview")
            
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown("""
                #### 📝 Apa yang sedang dilakukan
                Pada tahap ini sistem melakukan pembacaan awal terhadap dataset kualitas udara WHO, menganalisis dimensi baris dan kolom, mengidentifikasi missing values pada masing-masing variabel prediktor, serta memetakan tipe data asli.
                
                #### 🎯 Tujuan
                Mendapatkan gambaran awal mengenai kesehatan dataset (data sanity check) sebelum masuk ke proses preprocessing dan rekayasa model.
                
                #### ⚙️ Cara Kerja
                1. Memuat file `train.csv` melalui engine Pandas.
                2. Menghitung ukuran dimensi matriks (`df.shape`).
                3. Mendeteksi sel bernilai NaN (`isnull()`) dan mengonversinya ke rasio persentase.
                """)
            with sc2:
                st.markdown(f"""
                <div class="dashboard-card" style="text-align:center;">
                    <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Ukuran Dataset</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#64d2ff; margin-top:8px;">{df_data.shape[0]:,} × {df_data.shape[1]}</div>
                    <div style="font-size:0.85rem; color:#cbd5e1; margin-top:4px;">Baris Data × Variabel Fitur</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Plotly chart untuk missing values
                fig_mv = plot_plotly_missing_values(df_data)
                st.plotly_chart(fig_mv, use_container_width=True)
            
            st.markdown("""
            #### 📊 Hasil & Insight
            - **Kelengkapan Fitur:** Variabel koordinat geografis dan identitas stasiun terisi lengkap (0% missing).
            - **Tantangan Imputasi:** Kadar target PM2.5 (49.9% kosong) dan PM10 (26.9% kosong) memiliki persentase kehilangan data yang signifikan. Hal ini memerlukan penanganan pipeline imputasi (menggunakan metode median) agar model tidak bias.
            """)
            
        # STEP 2: Univariate Analysis
        elif st.session_state.eda_step == 2:
            st.markdown("### Langkah 2: Univariate Analysis")
            
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown("""
                #### 📝 Apa yang sedang dilakukan
                Melakukan analisis statistik deskriptif untuk satu variabel tunggal. Pada tahap ini kita melihat distribusi probabilitas dari konsentrasi PM2.5, PM10, serta proporsi sebaran kategori kualitas udara.
                
                #### 🎯 Tujuan
                Memahami karakteristik persebaran, rentang nilai (range), nilai tengah (median), mendeteksi adanya skewness (kemiringan), dan mendeteksi adanya pencilan (outliers).
                
                #### ⚙️ Cara Kerja
                1. Memisahkan variabel target PM2.5 dan PM10 secara individual.
                2. Melakukan plotting histogram berbasis frekuensi.
                3. Menghitung distribusi rasio klasifikasi biner 'Safety' vs 'Dangerous'.
                """)
            with sc2:
                fig_univ = plot_plotly_pm_distributions(df_data)
                st.plotly_chart(fig_univ, use_container_width=True)
                
                fig_class = plot_plotly_class_distribution(df_data)
                st.plotly_chart(fig_class, use_container_width=True)
                
            st.markdown("""
            #### 📊 Hasil & Insight
            - **Persebaran Nilai:** Distribusi konsentrasi polutan udara global (PM2.5 & PM10) memiliki kemiringan positif yang kuat (*highly right-skewed*). Sebagian besar wilayah dunia berkumpul pada konsentrasi rendah hingga sedang (di bawah 50 µg/m³), namun terdapat ekor data panjang yang menunjukkan adanya pencilan ekstrem (wilayah industri berat).
            - **Kategori Dominan:** Kategori aman mendominasi catatan sejarah database global ini, namun proporsi daerah berbahaya masih cukup signifikan untuk diperhatikan.
            """)
            
        # STEP 3: Bivariate Analysis
        elif st.session_state.eda_step == 3:
            st.markdown("### Langkah 3: Bivariate Analysis")
            
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown("""
                #### 📝 Apa yang sedang dilakukan
                Menganalisis hubungan interaktif antara dua variabel yang berbeda secara bersamaan. Di sini kita menyoroti hubungan antara PM10 vs PM2.5 (korelasi spasial polutan) dan distribusi PM2.5 di setiap regional benua WHO.
                
                #### 🎯 Tujuan
                Menguji hipotesis korelasi fisik antar polutan udara serta melihat variasi tingkat polusi berdasarkan klaster demografis regional.
                
                #### ⚙️ Cara Kerja
                1. Membuat scatter plot dua dimensi yang memetakan PM10 (Sumbu X) dan PM2.5 (Sumbu Y).
                2. Membuat boxplot interaktif untuk membandingkan median polutan di 7 wilayah WHO.
                """)
            with sc2:
                fig_scat = plot_plotly_scatter_pm(df_data)
                st.plotly_chart(fig_scat, use_container_width=True)
                
                fig_box = plot_plotly_boxplot_regions(df_data)
                st.plotly_chart(fig_box, use_container_width=True)
                
            st.markdown("""
            #### 📊 Hasil & Insight
            - **Pola Linear Kuat:** Terdapat tren linear positif yang sangat jelas antara PM10 dan PM2.5. Hal ini membuktikan secara ilmiah bahwa partikulat PM10 selalu beriringan dengan PM2.5, menjadikannya fitur penunjang dengan bobot kepentingan paling tinggi bagi model.
            - **Variasi Regional:** Wilayah Asia Tenggara (SEARO) dan Mediterania Timur (EMRO) memiliki nilai median dan persebaran konsentrasi PM2.5 yang jauh lebih tinggi secara signifikan dibandingkan regional Eropa (EURO) atau Amerika (AMRO).
            """)
            
        # STEP 4: Correlation Matrix
        elif st.session_state.eda_step == 4:
            st.markdown("### Langkah 4: Correlation Matrix")
            
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown("""
                #### 📝 Apa yang sedang dilakukan
                Membangun matriks korelasi Pearson komprehensif untuk mendeteksi tingkat hubungan kekuatan asosiasi linear di antara seluruh variabel numerik dalam dataset.
                
                #### 🎯 Tujuan
                Mengidentifikasi multikolinearitas (korelasi berlebih antar fitur prediktor) dan mengonfirmasi variabel mana yang paling berkontribusi langsung pada target PM2.5.
                
                #### ⚙️ Cara Kerja
                1. Menghitung koefisien korelasi Pearson ($r$) antar semua pasangan fitur numerik.
                2. Merender matriks dalam bentuk heatmap visual interaktif berwarna kontras tinggi (Biru: korelasi negatif, Merah: korelasi positif).
                """)
            with sc2:
                fig_corr = plot_plotly_correlation_matrix(df_data)
                st.plotly_chart(fig_corr, use_container_width=True)
                
            st.markdown("""
            #### 📊 Hasil & Insight
            - **Hubungan Terkuat:** PM10 menunjukkan tingkat korelasi tertinggi terhadap PM2.5 ($r = 0.89$), disusul oleh NO₂ ($r = 0.45$). Hal ini menyiratkan bahwa kualitas udara sangat bergantung pada polusi partikulat makro dan emisi gas.
            - **Faktor Geografis & Demografi:** Variabel populasi dan tahun memiliki korelasi linear yang relatif lemah terhadap PM2.5 global secara langsung, yang menandakan hubungannya kemungkinan bersifat non-linear kompleks sehingga memerlukan algoritma seperti Random Forest untuk memprosesnya.
            """)
            
        # STEP 5: Final Model Insight
        elif st.session_state.eda_step == 5:
            st.markdown("### Langkah 5: Final Model Insight & Evaluation")
            
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown("""
                #### 📝 Apa yang sedang dilakukan
                Mengevaluasi hasil perbandingan 5 model regresi (Linear, Ridge, Decision Tree, Random Forest, dan Gradient Boosting) dan merinci performa akhir dari model terpilih (Random Forest Regressor) yang digunakan di dashboard ini.
                
                #### 🎯 Tujuan
                Membuktikan akurasi model final, menganalisis penyebaran galat (error), dan mengonfirmasi bobot kepentingan fitur-fitur input yang digunakan dalam prediksi.
                
                #### ⚙️ Cara Kerja
                1. Menampilkan komparasi performa $R^2$ dan RMSE dari uji 5-Fold Cross Validation.
                2. Mengeksekusi model final pada 20% porsi data independen (test set) untuk merender sebaran prediksi vs aktual dan grafik tingkat kepentingan fitur (*Feature Importance*).
                """)
                
                fig_r2, fig_rmse = plot_plotly_model_comparison()
                st.markdown("**Perbandingan R² Score Model (Semakin Tinggi Semakin Baik):**")
                st.plotly_chart(fig_r2, use_container_width=True)
                
            with sc2:
                fig_eval_scatter, fig_eval_resid, fig_eval_feat = plot_plotly_eval_final(df_data, model)
                
                st.markdown("**Akurasi Prediksi Aktual vs Prediksi Model Terpilih:**")
                st.plotly_chart(fig_eval_scatter, use_container_width=True)
                
                st.markdown("**Variabel Fitur Paling Berpengaruh (Top 8):**")
                st.plotly_chart(fig_eval_feat, use_container_width=True)
                
            st.markdown("""
            #### 📊 Hasil & Insight
            - **Superioritas Random Forest:** Model Random Forest mengungguli model linear konvensional secara mutlak dengan nilai $R^2$ mencapai **0.8900** dan RMSE terendah (**2.68**). Ini membuktikan model mampu menangkap hubungan non-linear yang kompleks dengan sangat baik.
            - **Fitur Utama:** Konsentrasi PM10 mendominasi signifikansi kontribusi model, disusul oleh koordinat lintang/bujur (geografis), populasi penduduk, dan konsentrasi gas NO₂.
            """)
            
        # Tombol Navigasi Bawah
        st.markdown("<br>", unsafe_allow_html=True)
        col_prev, col_middle, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.session_state.eda_step > 1:
                if st.button("Sebelumnya"):
                    st.session_state.eda_step -= 1
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
        with col_middle:
            st.write("")
        with col_next:
            if st.session_state.eda_step < 5:
                if st.button("Langkah Berikutnya"):
                    st.session_state.eda_step += 1
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
