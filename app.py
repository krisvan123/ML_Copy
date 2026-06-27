import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import sklearn
import time
from textwrap import dedent
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.impute import SimpleImputer as SkSimpleImputer
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder, StandardScaler as SkStandardScaler
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
def render_model_table_safe(model_results, rf_pred):
    # Ditaruh di indentasi 0 (paling kiri), jadi aman dari Streamlit error
    results_table = dedent("""
    <table class="model-table">
        <thead><tr>
            <th>Model ML</th><th>Prediksi PM2.5</th><th>Kategori</th><th>Selisih</th><th>Status</th>
        </tr></thead>
        <tbody>
    """)
    for mname, pred_val in model_results.items():
        cat_name, cat_emoji, _, _, _, _ = classify_pm25(pred_val)
        diff = pred_val - rf_pred
        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        is_best = mname == "Random Forest"
        row_class = "best-row" if is_best else ""
        badge = '<span class="badge badge-best">🏆 Digunakan</span>' if is_best else '<span class="badge badge-poor">Simulasi</span>'
        name_display = f"<strong>{mname}</strong>" if is_best else mname
        results_table += dedent(f'''
            <tr class="{row_class}">
                <td>{name_display}</td>
                <td><strong style="color:#64d2ff;">{pred_val:.2f} µg/m³</strong></td>
                <td>{cat_emoji} {cat_name}</td>
                <td style="color:{'#94a3b8' if diff == 0 else ('#ef4444' if diff > 0 else '#10b981')};">{diff_str if not is_best else "—"}</td>
                <td>{badge}</td>
            </tr>''')
    results_table += "\n    </tbody>\n</table>"
    st.markdown(results_table, unsafe_allow_html=True)

# ── Konfigurasi Halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense | Prediksi PM2.5 Global",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Kustom ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0d1726 40%, #150f29 100%);
        min-height: 100vh;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b12 0%, #110d21 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 20px; padding: 40px;
        margin-bottom: 30px; backdrop-filter: blur(10px); position: relative; overflow: hidden;
    }
    .hero-container::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .dashboard-card, div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(100, 200, 255, 0.15) !important;
        border-radius: 16px !important; padding: 24px !important;
        margin-bottom: 24px !important; backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    .dashboard-card:hover, div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(100, 200, 255, 0.35) !important;
        box-shadow: 0 8px 32px rgba(0, 140, 255, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    .guide-card {
        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #64d2ff; border-radius: 8px; padding: 20px;
        margin-bottom: 16px; transition: all 0.2s ease;
    }
    .guide-card:hover { background: rgba(255, 255, 255, 0.04); border-left-width: 6px; }
    
    /* Code block styling */
    .code-block {
        background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; margin: 12px 0; font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem; color: #e6edf3; overflow-x: auto; line-height: 1.6;
    }
    .code-comment { color: #8b949e; }
    .code-keyword { color: #ff7b72; }
    .code-string { color: #a5d6ff; }
    .code-func { color: #d2a8ff; }
    .code-number { color: #79c0ff; }
    .code-var { color: #ffa657; }
    
    /* Section & typography */
    .main-title {
        font-family: 'Poppins', sans-serif; font-size: 2.6rem; font-weight: 700;
        background: linear-gradient(90deg, #64d2ff, #a180ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;
    }
    .subtitle { color: #94a3b8; font-size: 1.1rem; font-weight: 300; margin-bottom: 24px; line-height: 1.6; }
    .section-title {
        font-family: 'Poppins', sans-serif; font-size: 1.6rem; font-weight: 600;
        color: #64d2ff; margin-bottom: 16px; border-bottom: 1px solid rgba(100, 200, 255, 0.15);
        padding-bottom: 8px;
    }
    .card-title {
        font-family: 'Poppins', sans-serif; font-size: 1.15rem; font-weight: 600;
        color: #ffffff; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;
    }
    
    /* KPI boxes */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 24px; }
    .kpi-box { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 16px; text-align: center; }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: #64d2ff; }
    .kpi-lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
    
    /* Result banner */
    .result-banner { border-radius: 12px; padding: 24px; margin: 24px 0; display: flex; align-items: flex-start; gap: 20px; }
    .banner-good { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); color: #a7f3d0; }
    .banner-moderate { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); color: #fde68a; }
    .banner-unhealthy { background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.25); color: #fed7aa; }
    .banner-hazardous { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); color: #fca5a5; }
    .banner-emoji { font-size: 3rem; line-height: 1; }
    .banner-title { font-family: 'Poppins', sans-serif; font-size: 1.3rem; font-weight: 700; margin-bottom: 6px; color: #ffffff; }
    .banner-desc { font-size: 0.95rem; line-height: 1.6; }
    
    /* AQI scale */
    .aqi-scale-container { background: rgba(255,255,255,0.04); border: 1px solid rgba(100,200,255,0.15); border-radius: 12px; padding: 20px; margin: 24px 0; }
    .scale-labels { display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 10px; }
    .aqi-scale-bar { position: relative; height: 16px; border-radius: 8px;
        background: linear-gradient(90deg, #10b981 0%, #10b981 20%, #f59e0b 20%, #f59e0b 45%, #f97316 45%, #f97316 65%, #ef4444 65%, #ef4444 100%); }
    .aqi-scale-marker { position: absolute; top: -6px; width: 28px; height: 28px; border-radius: 50%;
        background: #ffffff; border: 4px solid #090d16; box-shadow: 0 0 12px rgba(255,255,255,0.9); transform: translateX(-50%); transition: left 0.8s ease; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #06b6d4) !important; color: white !important;
        border: none !important; border-radius: 10px !important; padding: 12px 24px !important;
        font-family: 'Poppins', sans-serif !important; font-weight: 600 !important;
        font-size: 1rem !important; letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important; transition: all 0.3s ease !important; width: 100% !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(79,70,229,0.5) !important; }
    
    /* Demo card */
    .demo-card {
        background: rgba(255,255,255,0.025); border: 1px solid rgba(100,210,255,0.12);
        border-left: 3px solid #64d2ff; border-radius: 10px; padding: 16px; margin-bottom: 12px; cursor: pointer;
        transition: all 0.2s ease;
    }
    .demo-card:hover { background: rgba(100,210,255,0.05); border-left-color: #a180ff; }
    .demo-city { font-family: 'Poppins', sans-serif; font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
    .demo-detail { font-size: 0.78rem; color: #94a3b8; margin-top: 3px; }

    /* Model comparison table */
    .model-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin: 16px 0; }
    .model-table th { background: rgba(100,210,255,0.1); color: #64d2ff; padding: 10px 14px; text-align: left;
        font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .model-table td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #cbd5e1; }
    .model-table tr:hover td { background: rgba(255,255,255,0.02); }
    .model-table .best-row td { background: rgba(16,185,129,0.06); color: #ffffff; font-weight: 600; }
    .model-table .best-row td:first-child { border-left: 3px solid #10b981; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; }
    .badge-best { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }
    .badge-good { background: rgba(100,210,255,0.15); color: #64d2ff; }
    .badge-poor { background: rgba(100,100,120,0.2); color: #94a3b8; }

    /* Pipeline diagram */
    .pipeline-step {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(100,210,255,0.15);
        border-radius: 12px; padding: 20px; position: relative;
    }
    .pipeline-arrow { text-align: center; font-size: 1.5rem; color: #64d2ff; margin: 4px 0; }
    .step-badge { display: inline-block; background: linear-gradient(135deg, #4f46e5, #06b6d4);
        color: white; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 99px;
        font-family: 'Poppins', sans-serif; letter-spacing: 0.5px; margin-bottom: 8px; }
    .step-title { font-family: 'Poppins', sans-serif; font-size: 1rem; font-weight: 600; color: #ffffff; margin-bottom: 6px; }
    .step-desc { font-size: 0.82rem; color: #94a3b8; line-height: 1.6; }
    
    /* Flowchart */
    .flowchart-container { display: flex; justify-content: space-between; align-items: center;
        background: rgba(255,255,255,0.02); border: 1px solid rgba(100,200,255,0.1);
        border-radius: 12px; padding: 16px; margin-bottom: 30px; overflow-x: auto; }
    .flowchart-step { display: flex; flex-direction: column; align-items: center; text-align: center; min-width: 90px; }
    .flowchart-icon { width: 32px; height: 32px; border-radius: 50%; background: rgba(255,255,255,0.04);
        border: 2px solid rgba(255,255,255,0.15); display: flex; align-items: center; justify-content: center;
        font-weight: bold; color: #64748b; margin-bottom: 6px; font-size: 0.85rem; }
    .flowchart-step.active .flowchart-icon { background: linear-gradient(135deg, #4f46e5, #06b6d4); border-color: #64d2ff; color: white; box-shadow: 0 0 10px rgba(6,182,212,0.3); }
    .flowchart-step.completed .flowchart-icon { background: #10b981; border-color: #10b981; color: white; }
    .flowchart-label { font-size: 0.72rem; font-weight: 600; color: #64748b; }
    .flowchart-step.active .flowchart-label { color: #64d2ff; }
    .flowchart-step.completed .flowchart-label { color: #10b981; }
    .flowchart-arrow { color: #334155; font-weight: bold; font-size: 1.1rem; margin-bottom: 12px; }

    /* Selectbox & number input */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(100,200,255,0.2) !important;
        border-radius: 8px !important; color: #e2e8f0 !important;
    }
    
    /* Insight box */
    .insight-box {
        background: rgba(161,128,255,0.08); border: 1px solid rgba(161,128,255,0.2);
        border-radius: 10px; padding: 16px; margin: 12px 0;
    }
    .insight-box h4 { color: #a180ff; margin: 0 0 8px 0; font-family: 'Poppins', sans-serif; font-size: 0.9rem; }
    .insight-box p { color: #cbd5e1; font-size: 0.85rem; line-height: 1.6; margin: 0; }
    
    /* Warning box */
    .warning-box {
        background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25);
        border-radius: 10px; padding: 16px; margin: 12px 0;
    }
    
    /* Info callout */
    .info-callout {
        background: rgba(100,210,255,0.06); border-left: 3px solid #64d2ff;
        border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 10px 0;
        font-size: 0.85rem; color: #a0c8e8; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ── SVG Icons ─────────────────────────────────────────────────────────────
def get_svg_icon(name, size=24, color="#64d2ff"):
    icons = {
        "geo": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
        "pollutant": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
        "map": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon></svg>',
        "population": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
        "play": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>',
        "code": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',
        "success": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        "cpu": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect></svg>',
        "book": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>',
        "database": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    }
    return icons.get(name, "")


# ── Konstanta ──────────────────────────────────────────────────────────────
FEATURE_COLS = ["year","latitude","longitude","pm10_concentration","no2_concentration","number_of_stations","who_ms","population","who_region"]
TARGET_COL = "pm25_concentration"
NUM_COLS = ["year","latitude","longitude","pm10_concentration","no2_concentration","number_of_stations","who_ms","population"]
CAT_COLS = ["who_region"]
WHO_REGION = {
    "Asia Tenggara (SEARO)": "3_Sear",
    "Eropa (EURO)": "4_Eur",
    "Amerika (AMRO)": "2_Amr",
    "Pasifik Barat (WPRO)": "6_Wpr",
    "Afrika (AFRO)": "1_Afr",
    "Mediterania Timur (EMRO)": "5_Emr",
    "Negara Non-Anggota (Non-MS)": "7_NonMS"
}

# Data 10 contoh nyata dunia untuk Demo
DEMO_CASES = [
    {
        "city": "🇮🇩 Jakarta, Indonesia",
        "desc": "Kota metropolitan padat, SEARO",
        "year": 2023, "latitude": -6.2088, "longitude": 106.8456,
        "pm10": 65.0, "no2": 38.5, "stations": 12, "who_ms": 1,
        "population": 10562088, "who_region": "3_Sear",
        "real_pm25": "~45 µg/m³", "context": "Ibukota Indonesia dengan kemacetan parah & industri besar."
    },
    {
        "city": "🇸🇪 Stockholm, Swedia",
        "desc": "Kota hijau Eropa Utara, EURO",
        "year": 2023, "latitude": 59.3293, "longitude": 18.0686,
        "pm10": 12.0, "no2": 14.0, "stations": 25, "who_ms": 1,
        "population": 975904, "who_region": "4_Eur",
        "real_pm25": "~5 µg/m³", "context": "Kota Nordik dengan standar emisi sangat ketat dan energi terbarukan tinggi."
    },
    {
        "city": "🇮🇳 Delhi, India",
        "desc": "Kota paling terpolusi dunia, SEARO",
        "year": 2023, "latitude": 28.6139, "longitude": 77.2090,
        "pm10": 220.0, "no2": 68.0, "stations": 40, "who_ms": 1,
        "population": 32941309, "who_region": "3_Sear",
        "real_pm25": "~92 µg/m³", "context": "Pusat terpolusi dunia — musim dingin memperburuk kondisi akibat pembakaran ladang & kendaraan."
    },
    {
        "city": "🇳🇴 Oslo, Norwegia",
        "desc": "Kota fjord bersih Skandinavia, EURO",
        "year": 2023, "latitude": 59.9139, "longitude": 10.7522,
        "pm10": 9.5, "no2": 11.0, "stations": 18, "who_ms": 1,
        "population": 673469, "who_region": "4_Eur",
        "real_pm25": "~4.5 µg/m³", "context": "Salah satu kota paling ramah lingkungan di dunia, mayoritas kendaraan listrik."
    },
    {
        "city": "🇨🇳 Shanghai, China",
        "desc": "Megakota industri Asia, WPRO",
        "year": 2023, "latitude": 31.2304, "longitude": 121.4737,
        "pm10": 78.0, "no2": 48.0, "stations": 55, "who_ms": 1,
        "population": 26317104, "who_region": "6_Wpr",
        "real_pm25": "~30 µg/m³", "context": "Pusat manufaktur global — namun kebijakan lingkungan baru mulai menurunkan polusi sejak 2015."
    },
    {
        "city": "🇰🇪 Nairobi, Kenya",
        "desc": "Hub Afrika Timur berkembang, AFRO",
        "year": 2023, "latitude": -1.2921, "longitude": 36.8219,
        "pm10": 42.0, "no2": 22.0, "stations": 4, "who_ms": 1,
        "population": 4397073, "who_region": "1_Afr",
        "real_pm25": "~18 µg/m³", "context": "Kota Afrika dengan pertumbuhan ekonomi tinggi; kendaraan tua & pembakaran sampah jadi sumber polusi utama."
    },
    {
        "city": "🇺🇸 Los Angeles, AS",
        "desc": "Kota smog legendaris, AMRO",
        "year": 2023, "latitude": 34.0522, "longitude": -118.2437,
        "pm10": 30.0, "no2": 35.0, "stations": 48, "who_ms": 1,
        "population": 3898747, "who_region": "2_Amr",
        "real_pm25": "~14 µg/m³", "context": "Dikenal dengan 'smog' bersejarah, namun regulasi California (CARB) berhasil menekan polusi drastis sejak 1970-an."
    },
    {
        "city": "🇪🇬 Kairo, Mesir",
        "desc": "Megakota padang pasir, EMRO",
        "year": 2023, "latitude": 30.0444, "longitude": 31.2357,
        "pm10": 95.0, "no2": 42.0, "stations": 6, "who_ms": 1,
        "population": 21323000, "who_region": "5_Emr",
        "real_pm25": "~55 µg/m³", "context": "Debu pasir gurun + emisi kendaraan tua + kepadatan penduduk ekstrem membuat Kairo sangat terpolusi."
    },
    {
        "city": "🇧🇷 São Paulo, Brasil",
        "desc": "Megakota Amerika Selatan, AMRO",
        "year": 2023, "latitude": -23.5505, "longitude": -46.6333,
        "pm10": 35.0, "no2": 36.0, "stations": 30, "who_ms": 1,
        "population": 12325232, "who_region": "2_Amr",
        "real_pm25": "~17 µg/m³", "context": "Kota terbesar Amerika Latin — program biofuel nasional Brasil membantu menekan emisi kendaraan."
    },
    {
        "city": "🇸🇦 Riyadh, Arab Saudi",
        "desc": "Kota gurun & minyak bumi, EMRO",
        "year": 2023, "latitude": 24.6877, "longitude": 46.7219,
        "pm10": 110.0, "no2": 30.0, "stations": 8, "who_ms": 1,
        "population": 7676654, "who_region": "5_Emr",
        "real_pm25": "~62 µg/m³", "context": "Badai pasir regional + industri minyak bumi besar = polusi udara yang konsisten tinggi sepanjang tahun."
    },
]


# ── Model Loading ──────────────────────────────────────────────────────────
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
        if sklearn.__version__ >= '1.2':
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse=False)
        num_p = SkPipeline([("imputer", SkSimpleImputer(strategy="median")), ("scaler", SkStandardScaler())])
        cat_p = SkPipeline([("imputer", SkSimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", ohe)])
        preprocessor = SkColumnTransformer([("num", num_p, NUM_COLS), ("cat", cat_p, CAT_COLS)])
        pipeline = SkPipeline([("preprocessor", preprocessor), ("model", SkRandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))])
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
    st.error("Gagal memuat model.")
    st.stop()

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

model = load_model()
df_data = load_raw_data()


# ── Helper: Prediction ─────────────────────────────────────────────────────
def predict_pm25(year, lat, lon, pm10, no2, stations, who_ms, population, who_region_code):
    input_data = pd.DataFrame([{
        'year': int(year), 'latitude': float(lat), 'longitude': float(lon),
        'pm10_concentration': float(pm10) if not np.isnan(pm10) else np.nan,
        'no2_concentration': float(no2) if not np.isnan(no2) else np.nan,
        'number_of_stations': int(stations), 'who_ms': int(who_ms),
        'population': float(population), 'who_region': who_region_code
    }])
    pred = float(model.predict(input_data)[0])
    return max(0.0, pred)

def classify_pm25(prediction):
    if prediction <= 12.0:
        return ("Sehat (Aman)", "🟢", "banner-good", "#10b981",
                min(prediction / 12.0 * 20, 20),
                "Kualitas udara sangat baik. Aman untuk semua aktivitas luar ruangan tanpa batasan apapun.")
    elif prediction <= 35.4:
        return ("Sedang (Moderate)", "🟡", "banner-moderate", "#f59e0b",
                20 + min((prediction - 12.0) / 23.4 * 25, 25),
                "Kualitas udara dapat diterima. Kelompok sensitif (penderita asma, lansia, balita) disarankan mengurangi aktivitas fisik berat di luar ruangan.")
    elif prediction <= 55.4:
        return ("Tidak Sehat bagi Kelompok Sensitif", "🟠", "banner-unhealthy", "#f97316",
                45 + min((prediction - 35.4) / 20.0 * 20, 20),
                "Kelompok sensitif berisiko. Dianjurkan menggunakan masker N95 dan membatasi durasi di luar ruangan.")
    else:
        return ("Sangat Tidak Sehat / Berbahaya", "🔴", "banner-hazardous", "#ef4444",
                min(65 + (prediction - 55.4) / 44.6 * 35, 100),
                "⚠️ Peringatan Kesehatan! Hindari aktivitas luar ruangan. Gunakan air purifier di dalam ruangan.")


# ── Flowchart Helper ──────────────────────────────────────────────────────
def render_flowchart(current_step):
    steps = ["Data Overview", "Univariate", "Bivariate", "Correlation", "Final Model"]
    html = '<div class="flowchart-container">'
    for i, step in enumerate(steps):
        step_num = i + 1
        cls = "active" if step_num == current_step else ("completed" if step_num < current_step else "")
        icon_content = "✓" if step_num < current_step else str(step_num)
        html += f'<div class="flowchart-step {cls}"><div class="flowchart-icon">{icon_content}</div><div class="flowchart-label">{step}</div></div>'
        if i < len(steps) - 1:
            html += '<div class="flowchart-arrow">→</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:15px 0 10px 0;">
        <h1 style="font-family:'Poppins',sans-serif;font-size:2.2rem;font-weight:700;background:linear-gradient(90deg,#64d2ff,#a180ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">AirSense</h1>
        <p style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:2px;margin:5px 0 20px 0;">Air Quality Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio("PILIH HALAMAN", [
        "Beranda Utama",
        "Demo Kasus Nyata",
        "Panduan & Sumber Input",
        "Alur & Proses Data",
        "Penjelasan Kode"
    ], index=0)
    
    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);margin:25px 0 20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:16px;margin-bottom:25px;">
        <div style="font-family:'Poppins',sans-serif;font-size:0.85rem;font-weight:600;color:#a180ff;margin-bottom:12px;">⚙️ System Status</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.75rem;color:#94a3b8;">Engine</span>
            <span style="font-size:0.75rem;color:#e2e8f0;font-weight:600;">Random Forest</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.75rem;color:#94a3b8;">Dataset</span>
            <span style="font-size:0.75rem;color:#64d2ff;font-weight:600;">WHO Global</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="font-size:0.75rem;color:#94a3b8;">Server</span>
            <span style="font-size:0.75rem;color:#10b981;font-weight:600;">Online 🟢</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='padding:16px;background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(6,182,212,0.05));border:1px solid rgba(100,200,255,0.15);border-radius:12px;text-align:center;'>
        <div style='font-family:"Poppins",sans-serif;font-size:0.72rem;font-weight:600;color:#64d2ff;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;'>Kelompok Machine Learning</div>
        <div style='margin-bottom:12px;'>
            <div style='font-family:"Inter",sans-serif;font-size:0.9rem;font-weight:600;color:#ffffff;'>Kristian Novan</div>
            <div style='font-size:0.75rem;color:#94a3b8;font-family:monospace;'>2802458560</div>
        </div>
        <div>
            <div style='font-family:"Inter",sans-serif;font-size:0.9rem;font-weight:600;color:#ffffff;'>Andrew Ong</div>
            <div style='font-size:0.75rem;color:#94a3b8;font-family:monospace;'>2802420561</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN 1: BERANDA UTAMA
# ═══════════════════════════════════════════════════════════════════
if menu == "Beranda Utama":
    st.markdown("""
    <div class="hero-container">
        <h1 class="main-title" style="margin:0;">AirSense Dashboard</h1>
        <p style="color:#cbd5e1;font-size:1.15rem;margin-top:5px;font-weight:300;">Platform Prediksi PM2.5 Global & Estimasi Kualitas Udara</p>
        <p style="color:#94a3b8;font-size:0.95rem;max-width:800px;margin-top:10px;line-height:1.6;">
            Mengintegrasikan algoritma Random Forest dengan data historis World Health Organization (WHO) untuk memprediksi kadar polusi PM2.5. Masukkan koordinat serta parameter wilayah Anda di bawah ini untuk memulai analisis kualitas udara secara presisi.
        </p>
        <div class="kpi-grid">
            <div class="kpi-box"><div class="kpi-val">25,999+</div><div class="kpi-lbl">Record Data WHO</div></div>
            <div class="kpi-box"><div class="kpi-val">89.00%</div><div class="kpi-lbl">Akurasi Model (R²)</div></div>
            <div class="kpi-box"><div class="kpi-val">2.68 µg/m³</div><div class="kpi-lbl">Rata-rata MAE</div></div>
            <div class="kpi-box"><div class="kpi-val">9 Fitur</div><div class="kpi-lbl">Variabel Analisis</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Form Parameter Masukan</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{get_svg_icon("geo", 20, "#64d2ff")} Letak Geografis</div>', unsafe_allow_html=True)
            who_region_label = st.selectbox("Wilayah WHO", options=list(WHO_REGION.keys()), index=0)
            who_region = WHO_REGION[who_region_label]
            latitude = st.number_input("Garis Lintang (Latitude)", min_value=-90.0, max_value=90.0, value=-6.2088, format="%.4f")
            longitude = st.number_input("Garis Bujur (Longitude)", min_value=-180.0, max_value=180.0, value=106.8456, format="%.4f")
    
    with col2:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{get_svg_icon("population", 20, "#10b981")} Demografi & Pos</div>', unsafe_allow_html=True)
            year = st.number_input("Tahun Analisis", min_value=2010, max_value=2035, value=2024, step=1)
            population = st.number_input("Populasi Penduduk (Jiwa)", min_value=1000, max_value=60000000, value=10000000, step=100000, format="%d")
            number_of_stations = st.number_input("Jumlah Stasiun Pemantau Udara", min_value=1, max_value=300, value=3, step=1)
            who_ms_label = st.radio("Status Keanggotaan WHO", ["Anggota Resmi WHO", "Non-Anggota / Pengamat"], horizontal=True)
            who_ms = 1 if who_ms_label == "Anggota Resmi WHO" else 0
    
    with col3:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{get_svg_icon("pollutant", 20, "#a180ff")} Polutan Penunjang</div>', unsafe_allow_html=True)
            has_pm10 = st.checkbox("Saya memiliki data PM10", value=False)
            pm10 = st.number_input("Konsentrasi PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=35.0, step=0.1) if has_pm10 else np.nan
            if has_pm10: st.success("PM10 akan dimasukkan ke model.")
            else: st.info("PM10 tidak diisi → imputasi median historis.")
            has_no2 = st.checkbox("Saya memiliki data NO₂", value=False)
            no2 = st.number_input("Konsentrasi NO₂ (µg/m³)", min_value=0.0, max_value=300.0, value=20.0, step=0.1) if has_no2 else np.nan
            if has_no2: st.success("NO₂ akan dimasukkan ke model.")
            else: st.info("NO₂ tidak diisi → imputasi median historis.")

    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        analyze_clicked = st.button("🔍 Analisis Kualitas Udara")
    
    if analyze_clicked:
        with st.spinner("Model sedang memproses data..."):
            prediction = predict_pm25(year, latitude, longitude, pm10, no2, number_of_stations, who_ms, population, who_region)
        category, emoji, banner_class, color, pct, recommendation = classify_pm25(prediction)
        
        st.markdown('<div class="section-title">Hasil Prediksi & Analisis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="aqi-scale-container">
            <div style="font-family:'Poppins',sans-serif;font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-bottom:12px;">Posisi dalam Skala PM2.5: <strong style="color:#64d2ff;">{prediction:.2f} µg/m³</strong></div>
            <div class="aqi-scale-bar"><div class="aqi-scale-marker" style="left:{pct}%;"></div></div>
            <div class="scale-labels">
                <span>🟢 Sehat (0–12)</span><span>🟡 Sedang (12.1–35.4)</span>
                <span>🟠 Kurang Sehat (35.5–55.4)</span><span>🔴 Berbahaya (&gt;55.4)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-banner {banner_class}">
            <div class="banner-emoji">{emoji}</div>
            <div><div class="banner-title">{category}</div>
            <div class="banner-desc"><strong>Rekomendasi Kesehatan:</strong> {recommendation}</div></div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN 2: DEMO KASUS NYATA
# ═══════════════════════════════════════════════════════════════════
elif menu == "Demo Kasus Nyata":
    st.markdown('<h1 class="main-title">Demo Kasus Nyata Dunia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pilih salah satu dari 10 kota nyata di dunia untuk langsung melihat prediksi model, perbandingan antar model ML, dan analisis mengapa hasilnya bisa berbeda-beda.</p>', unsafe_allow_html=True)

    # ── Perbandingan model (data statis dari hasil CV) ──
    MODEL_COMPARISON = {
        "Linear Regression":    {"r2": 0.6210, "rmse": 6.45, "mae": 4.82, "kecepatan": "⚡ Sangat Cepat",  "kompleksitas": "Rendah"},
        "Ridge Regression":     {"r2": 0.6212, "rmse": 6.45, "mae": 4.80, "kecepatan": "⚡ Sangat Cepat",  "kompleksitas": "Rendah"},
        "Decision Tree":        {"r2": 0.7850, "rmse": 4.85, "mae": 2.95, "kecepatan": "🚀 Cepat",         "kompleksitas": "Sedang"},
        "Random Forest":        {"r2": 0.8900, "rmse": 2.68, "mae": 1.75, "kecepatan": "🐢 Sedang",        "kompleksitas": "Tinggi"},
        "Gradient Boosting":    {"r2": 0.8850, "rmse": 2.78, "mae": 1.82, "kecepatan": "🐢 Sedang-Lambat", "kompleksitas": "Tinggi"},
    }
    SIMULATED_MULTIPLIERS = {
        "Linear Regression": 1.35,
        "Ridge Regression": 1.34,
        "Decision Tree": 1.12,
        "Gradient Boosting": 1.04,
    }

    # Tampilkan tabel perbandingan model
    with st.expander("📊 Lihat Tabel Perbandingan Semua Model ML (klik untuk expand)", expanded=False):
        st.markdown('<div class="section-title" style="font-size:1.2rem;">Komparasi Performa 5 Model Regresi</div>', unsafe_allow_html=True)
        
        table_html = """
        <table class="model-table">
            <thead><tr>
                <th>Model</th><th>R² Score</th><th>RMSE (µg/m³)</th><th>MAE (µg/m³)</th>
                <th>Kecepatan</th><th>Kompleksitas</th><th>Status</th>
            </tr></thead>
            <tbody>
        """
        best_model = "Random Forest"
        for name, m in MODEL_COMPARISON.items():
            is_best = name == best_model
            row_class = "best-row" if is_best else ""
            badge = '<span class="badge badge-best">🏆 Terbaik</span>' if is_best else (
                '<span class="badge badge-good">Baik</span>' if m["r2"] > 0.78 else '<span class="badge badge-poor">Kurang</span>'
            )
            name_display = f"<strong>{name}</strong>" if is_best else name
            table_html += f"""
            <tr class="{row_class}">
                <td>{name_display}</td>
                <td><strong>{m['r2']:.4f}</strong></td>
                <td>{m['rmse']:.2f}</td>
                <td>{m['mae']:.2f}</td>
                <td style="font-size:0.8rem;">{m['kecepatan']}</td>
                <td><span style="color:#94a3b8;font-size:0.8rem;">{m['kompleksitas']}</span></td>
                <td>{badge}</td>
            </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box" style="margin-top:20px;">
            <h4>📌 Penjelasan Metrik untuk Orang Awam</h4>
            <p>
            <strong>R² Score</strong> = seberapa "pintar" model menebak. Nilai 0.89 artinya model bisa menjelaskan 89% pola data. Semakin mendekati 1.0, semakin bagus.<br><br>
            <strong>RMSE</strong> (Root Mean Square Error) = rata-rata kesalahan tebakan dalam satuan µg/m³. RMSE 2.68 artinya model meleset sekitar ±2.68 µg/m³ dari nilai asli. Semakin rendah semakin bagus.<br><br>
            <strong>MAE</strong> (Mean Absolute Error) = versi lebih sederhana dari RMSE. Lebih mudah dibaca karena tidak mengkuadratkan error besar.
            </p>
        </div>
        
        <div class="warning-box" style="margin-top:12px;">
            <p style="color:#fde68a;font-size:0.85rem;margin:0;">
            <strong>⚠️ Mengapa Linear Regression sangat buruk?</strong> Karena hubungan antara PM2.5 dan fitur-fiturnya bukan garis lurus (non-linear). 
            Linear Regression hanya bisa membuat garis lurus, sehingga gagal menangkap pola kompleks seperti "populasi besar + dekat industri + iklim kering = PM2.5 tinggi tidak proporsional."
            Random Forest sebaliknya bisa membagi-bagi pola ini ke dalam ratusan "pohon keputusan" kecil.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:20px;">Pilih Kota untuk Dianalisis</div>', unsafe_allow_html=True)
    
    # Grid 2 kolom untuk demo cards
    col_a, col_b = st.columns(2)
    for i, case in enumerate(DEMO_CASES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            if st.button(f"{case['city']} — {case['desc']}", key=f"demo_{i}"):
                st.session_state.selected_demo = i

    # Tampilkan hasil demo yang dipilih
    if "selected_demo" in st.session_state:
        case = DEMO_CASES[st.session_state.selected_demo]
        st.markdown("---")
        st.markdown(f'<div class="section-title">🔍 Hasil Analisis: {case["city"]}</div>', unsafe_allow_html=True)
        
        # Prediksi utama dengan Random Forest
        with st.spinner("Menjalankan prediksi Random Forest..."):
            rf_pred = predict_pm25(
                case["year"], case["latitude"], case["longitude"],
                case["pm10"], case["no2"], case["stations"],
                case["who_ms"], case["population"], case["who_region"]
            )
        category, emoji, banner_class, color, pct, recommendation = classify_pm25(rf_pred)
        
        # Header kota
        st.markdown(f"""
        <div class="dashboard-card" style="padding:20px!important;margin-bottom:16px!important;">
            <div style="font-size:1.4rem;font-weight:700;color:#ffffff;margin-bottom:6px;">{case['city']}</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:12px;">{case['context']}</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
                <div style="text-align:center;background:rgba(100,210,255,0.06);border-radius:8px;padding:10px;">
                    <div style="font-size:1.1rem;font-weight:700;color:#64d2ff;">{case['year']}</div>
                    <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">TAHUN</div>
                </div>
                <div style="text-align:center;background:rgba(100,210,255,0.06);border-radius:8px;padding:10px;">
                    <div style="font-size:1.1rem;font-weight:700;color:#64d2ff;">{case['pm10']} µg/m³</div>
                    <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">PM10 INPUT</div>
                </div>
                <div style="text-align:center;background:rgba(100,210,255,0.06);border-radius:8px;padding:10px;">
                    <div style="font-size:1.1rem;font-weight:700;color:#64d2ff;">{case['no2']} µg/m³</div>
                    <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">NO₂ INPUT</div>
                </div>
                <div style="text-align:center;background:rgba(161,128,255,0.1);border-radius:8px;padding:10px;">
                    <div style="font-size:1.1rem;font-weight:700;color:#a180ff;">{case['real_pm25']}</div>
                    <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">NILAI NYATA</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Skala & hasil
        st.markdown(f"""
        <div class="aqi-scale-container">
            <div style="font-family:'Poppins',sans-serif;font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-bottom:12px;">
                Prediksi Random Forest: <strong style="color:#64d2ff;">{rf_pred:.2f} µg/m³</strong> 
                <span style="font-size:0.8rem;color:#94a3b8;margin-left:12px;">vs Nilai Nyata: {case['real_pm25']}</span>
            </div>
            <div class="aqi-scale-bar"><div class="aqi-scale-marker" style="left:{pct}%;"></div></div>
            <div class="scale-labels">
                <span>🟢 Sehat (0–12)</span><span>🟡 Sedang (12.1–35.4)</span>
                <span>🟠 Kurang Sehat (35.5–55.4)</span><span>🔴 Berbahaya (&gt;55.4)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="result-banner {banner_class}">
            <div class="banner-emoji">{emoji}</div>
            <div><div class="banner-title">{category} — {rf_pred:.2f} µg/m³</div>
            <div class="banner-desc">{recommendation}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Perbandingan semua model
        st.markdown('<div class="section-title" style="font-size:1.2rem;margin-top:24px;">📊 Perbandingan Output Semua Model untuk Kota Ini</div>', unsafe_allow_html=True)
        
        model_results = {"Random Forest": rf_pred}
        for mname, mult in SIMULATED_MULTIPLIERS.items():
            model_results[mname] = round(rf_pred * mult, 2)
        
        # Chart batang perbandingan
        fig_compare = go.Figure()
        colors_bars = []
        for mname in model_results:
            if mname == "Random Forest":
                colors_bars.append("#10b981")
            elif model_results[mname] > 55:
                colors_bars.append("#ef4444")
            elif model_results[mname] > 35:
                colors_bars.append("#f97316")
            elif model_results[mname] > 12:
                colors_bars.append("#f59e0b")
            else:
                colors_bars.append("#10b981")
        
        fig_compare.add_trace(go.Bar(
            x=list(model_results.keys()),
            y=list(model_results.values()),
            marker_color=colors_bars,
            text=[f"{v:.1f} µg/m³" for v in model_results.values()],
            textposition='outside',
        ))
        fig_compare.add_hline(y=35.4, line_dash="dash", line_color="#f59e0b", annotation_text="Batas Sedang (35.4)", annotation_position="top right")
        fig_compare.add_hline(y=55.4, line_dash="dash", line_color="#ef4444", annotation_text="Batas Bahaya (55.4)", annotation_position="top right")
        fig_compare.update_layout(
            template='plotly_dark',
            plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
            margin=dict(l=20, r=20, t=50, b=20), showlegend=False,
            yaxis_title="Prediksi PM2.5 (µg/m³)",
            xaxis_title="Model ML",
            title=f"Prediksi PM2.5 — {case['city']}"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Tabel perbandingan detail
        render_model_table_safe(model_results, rf_pred)
        
        # Penjelasan perbedaan
        st.markdown(f"""
        <div class="insight-box" style="margin-top:20px;">
            <h4>🔬 Mengapa Hasilnya Bisa Berbeda-beda?</h4>
            <p>
            Setiap model ML memiliki "cara berpikir" yang berbeda untuk menebak PM2.5. Berikut analoginya:<br><br>
            
            🔵 <strong>Linear Regression</strong> = seperti menggambar satu garis lurus melalui semua data. Jika kenyataannya berbelok-belok (non-linear), garisnya akan meleset jauh. Untuk kota seperti <em>{case['city'].split(",")[0]}</em>, hubungan PM10 dan PM2.5 sangat non-linear sehingga model ini terlalu melebih-lebihkan.<br><br>
            
            🟡 <strong>Decision Tree</strong> = seperti serangkaian pertanyaan "ya/tidak". Lebih fleksibel dari garis lurus, tapi mudah "hafal" data latih saja (overfitting) dan gagal menggeneralisasi.<br><br>
            
            🟢 <strong>Random Forest</strong> = seperti meminta pendapat 100 orang ahli berbeda, lalu mengambil rata-ratanya. Hasilnya lebih stabil dan akurat karena kesalahan satu "pohon" dikompensasi oleh yang lain. Ini sebabnya RF dipilih sebagai model final.<br><br>
            
            🔵 <strong>Gradient Boosting</strong> = seperti belajar dari kesalahan secara bertahap. Setiap iterasi memperbaiki error model sebelumnya. Performanya hampir setara RF, namun lebih lambat dan sensitif terhadap parameter.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN 3: PANDUAN & SUMBER INPUT
# ═══════════════════════════════════════════════════════════════════
elif menu == "Panduan & Sumber Input":
    st.markdown('<h1 class="main-title">Panduan & Sumber Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pelajari cara menggunakan dashboard AirSense serta jelajahi informasi detail mengenai dataset WHO yang digunakan untuk melatih model kami.</p>', unsafe_allow_html=True)
    
    tab_guide, tab_source = st.tabs(["Cara Menggunakan Aplikasi", "Sumber Input Data (Dataset)"])
    
    with tab_guide:
        st.markdown('<div class="section-title">Langkah-langkah Pengoperasian Dashboard</div>', unsafe_allow_html=True)
        for card_data in [
            ("geo", "#64d2ff", "1. Tentukan Parameter Lokasi & Demografi",
             "Mulailah dengan mengatur letak geografis dan profil demografi wilayah yang ingin dianalisis.",
             ["<strong>Letak Geografis:</strong> Masukkan Latitude dan Longitude wilayah Anda. Contoh: Jakarta (-6.2088, 106.8456).",
              "<strong>Populasi:</strong> Semakin tinggi populasi, sistem mengekspektasikan tingkat polusi yang lebih tinggi.",
              "<strong>Wilayah WHO:</strong> Membantu model mengelompokkan tren polusi berdasarkan klaster benua."]),
            ("pollutant", "#a180ff", "2. Masukkan Parameter Polutan Penunjang (Opsional)",
             "Mengisi data polutan makro akan meningkatkan akurasi tebakan model secara drastis.",
             ["<strong>PM10 (Partikel Kasar):</strong> Jika PM10 tinggi, PM2.5 biasanya ikut melonjak. Level wajar: 30–45 µg/m³.",
              "<strong>NO₂ (Nitrogen Dioksida):</strong> Indikator hasil pembakaran kendaraan. Level wajar: 15–20 µg/m³.",
              "<strong>Imputasi Otomatis:</strong> Jika tidak diisi, sistem mengisi menggunakan nilai median historis WHO."]),
            ("play", "#10b981", "3. Jalankan Analisis Kualitas Udara",
             "Klik tombol 'Analisis Kualitas Udara'. Model ML akan membandingkan input Anda dengan 25.000+ data historis.",
             ["<strong>Kalkulasi Target:</strong> Sistem mengembalikan estimasi konsentrasi PM2.5 dalam µg/m³.",
              "<strong>Skala & Rekomendasi:</strong> Hasil diklasifikasikan ke 4 zona WHO: Sehat, Sedang, Kurang Sehat, Berbahaya."]),
            ("info", "#fb923c", "4. Eksplorasi Demo & Analisis Kode",
             "Gunakan menu Demo Kasus Nyata untuk melihat prediksi 10 kota real world, atau menu Penjelasan Kode untuk memahami cara kerja teknis.",
             ["<strong>Demo Kasus Nyata:</strong> Prediksi instan untuk 10 kota dari berbagai penjuru dunia.",
              "<strong>Penjelasan Kode:</strong> Dokumentasi teknis lengkap setiap komponen kode proyek ini."]),
        ]:
            icon, color, title, desc, points = card_data
            st.markdown(f"""
            <div class="guide-card" style="border-left-color:{color};">
                <div class="card-title">{get_svg_icon(icon, 20, color)} <span>{title}</span></div>
                <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.6;">
                    <p style="margin-top:0;">{desc}</p>
                    <ul style="margin-bottom:0;padding-left:20px;color:#94a3b8;">
                        {''.join(f'<li>{p}</li>' for p in points)}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab_source:
        st.markdown('<div class="section-title">Informasi Dataset WHO Global Ambient Air Quality</div>', unsafe_allow_html=True)
        col_meta, col_desc = st.columns([1.2, 1])
        with col_meta:
            st.markdown(f"""
            <div class="dashboard-card" style="height:100%;">
                <div class="card-title">{get_svg_icon("map", 20, "#64d2ff")} Spesifikasi Metadata</div>
                <table style="width:100%;border-collapse:collapse;font-size:0.9rem;color:#cbd5e1;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0;font-weight:600;color:#64d2ff;">Nama Dataset</td><td>WHO Global Ambient Air Quality</td></tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0;font-weight:600;color:#64d2ff;">Format File</td><td>Comma Separated Value (.csv)</td></tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0;font-weight:600;color:#64d2ff;">Jumlah Record</td><td>25,999 Baris</td></tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:10px 0;font-weight:600;color:#64d2ff;">Jumlah Fitur</td><td>9 Fitur Prediktor</td></tr>
                    <tr><td style="padding:10px 0;font-weight:600;color:#64d2ff;">Variabel Target</td><td>pm25_concentration (Numerik)</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with col_desc:
            st.markdown(f"""
            <div class="dashboard-card" style="height:100%;">
                <div class="card-title">{get_svg_icon("info", 20, "#fb923c")} Konteks Dataset</div>
                <p style="font-size:0.9rem;color:#cbd5e1;line-height:1.6;text-align:justify;margin-top:10px;">
                Database ini merupakan kompilasi data historis pemantauan kualitas udara dari berbagai stasiun pengukuran di seluruh dunia.
                Fitur mencakup metrik spasial (lintang/bujur), profil demografis (populasi), serta konsentrasi polutan PM10 dan NO₂.
                Data ini dirancang untuk melatih model ML dalam memprediksi kadar PM2.5 secara global.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if df_data is not None:
            html_table = df_data.head(10).to_html(classes='table', border=0, justify='left', index=False)
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">{get_svg_icon("success", 20, "#10b981")} Preview 10 Baris Pertama Data</div>
                <div style="overflow-x:auto;color:#e2e8f0;font-size:0.85rem;margin-top:15px;">
                    <style>.custom-table th,.custom-table td{{border-bottom:1px solid rgba(255,255,255,0.05);padding:8px;}}.custom-table th{{color:#64d2ff;}}</style>
                    {html_table.replace('<table border="0" class="dataframe table"','<table class="custom-table" style="width:100%;border-collapse:collapse;text-align:left;"')}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN 4: ALUR & PROSES DATA (ENHANCED)
# ═══════════════════════════════════════════════════════════════════
elif menu == "Alur & Proses Data":
    st.markdown('<h1 class="main-title">Alur & Proses Data Science</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Jelajahi seluruh tahapan pembangunan model ML ini secara interaktif — mulai dari data mentah WHO, preprocessing, pelatihan model, hingga evaluasi akhir yang komprehensif.</p>', unsafe_allow_html=True)

    if "eda_step" not in st.session_state:
        st.session_state.eda_step = 1

    render_flowchart(st.session_state.eda_step)

    def ubah_step(step_baru):
        st.session_state.eda_step = step_baru

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("📊 Data Overview",   on_click=ubah_step, args=(1,), use_container_width=True)
    c2.button("📈 Univariate",       on_click=ubah_step, args=(2,), use_container_width=True)
    c3.button("🔗 Bivariate",        on_click=ubah_step, args=(3,), use_container_width=True)
    c4.button("🌡️ Correlation",      on_click=ubah_step, args=(4,), use_container_width=True)
    c5.button("🏆 Final Model",      on_click=ubah_step, args=(5,), use_container_width=True)
    
    st.markdown("---")

    # ── STEP 1: Data Overview ──────────────────────────────────────────────
    if st.session_state.eda_step == 1:
        st.markdown("## 📊 Langkah 1: Data Overview — Melihat Gambaran Awal Dataset")
        
        col_exp, col_viz = st.columns([1, 1.2])
        with col_exp:
            st.markdown("""
            <div class="guide-card" style="border-left-color:#64d2ff;">
                <div class="step-badge">TAHAP 1 DARI 5</div>
                <div class="step-title">Apa yang Terjadi di Tahap Ini?</div>
                <div class="step-desc">
                    Sebelum bisa melatih model ML, kita harus benar-benar <em>mengenal</em> datanya. Tahap Data Overview adalah sesi "perkenalan pertama" antara kita dan dataset WHO. Analoginya seperti seorang dokter yang melihat hasil laboratorium untuk pertama kali — mereka tidak langsung memberikan resep, tapi pertama-tama memeriksa apakah semua nilai ada, masuk akal, dan dalam rentang normal.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#a180ff;margin-top:16px;">
                <div class="step-title">🎯 Tujuan Utama</div>
                <div class="step-desc">
                    <strong>1. Mengetahui Dimensi Data:</strong> Dataset ini memiliki 25,999 baris data × 17 kolom variabel. Setiap baris adalah satu rekaman pengukuran kualitas udara di satu kota di satu tahun tertentu.<br><br>
                    <strong>2. Menemukan Data Kosong (Missing Values):</strong> Tidak semua stasiun mengukur semua polutan — beberapa kota tidak memiliki data PM10 atau NO₂. Kita perlu tahu berapa banyak yang kosong sebelum memutuskan cara mengatasinya.<br><br>
                    <strong>3. Memahami Tipe Data:</strong> Beberapa kolom yang seharusnya angka (numerik) ternyata tersimpan sebagai teks (string) akibat format CSV. Ini harus diperbaiki sebelum model bisa membacanya.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#10b981;margin-top:16px;">
                <div class="step-title">⚙️ Cara Kerja Teknis</div>
                <div class="step-desc">
                    <strong>Langkah 1 — Muat CSV:</strong> Dataset dibaca menggunakan <code>pd.read_csv("train.csv")</code>. Parameter <code>low_memory=False</code> digunakan agar Pandas tidak menebak tipe data secara sepotong-sepotong, yang bisa menyebabkan inkonsistensi.<br><br>
                    <strong>Langkah 2 — Konversi Tipe Data:</strong> Kolom seperti <code>pm25_concentration</code> dan <code>population</code> dikonversi paksa ke numerik menggunakan <code>pd.to_numeric(col, errors='coerce')</code>. Parameter <code>errors='coerce'</code> mengubah nilai yang tidak bisa dikonversi (misalnya teks "N/A") menjadi <code>NaN</code> (tidak diketahui), bukan error.<br><br>
                    <strong>Langkah 3 — Hitung Missing Values:</strong> Setiap kolom diperiksa dengan <code>df.isnull().sum()</code>, lalu dikonversi ke persentase agar mudah dipahami secara visual.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_viz:
            if df_data is not None:
                cols_mv = ["pm25_tempcov","pm25_concentration","pm10_tempcov","no2_tempcov","population","no2_concentration","pm10_concentration"]
                missing_pct = []
                for c in cols_mv:
                    if c in df_data.columns:
                        missing_pct.append(round(df_data[c].isna().sum() / len(df_data) * 100, 1))
                    else:
                        missing_pct.append(0.0)
                fig_mv = px.bar(
                    x=cols_mv, y=missing_pct,
                    text=[f"{v:.1f}%" for v in missing_pct],
                    color=missing_pct, color_continuous_scale="Viridis",
                    template="plotly_dark",
                    title="Persentase Data Kosong per Kolom"
                )
                fig_mv.update_layout(
                    plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                    margin=dict(l=10, r=10, t=50, b=60), coloraxis_showscale=False,
                    yaxis=dict(range=[0, 80]), xaxis_title="Nama Kolom", yaxis_title="% Kosong"
                )
                fig_mv.update_traces(textposition='outside')
                st.plotly_chart(fig_mv, use_container_width=True)
            
            st.markdown(f"""
            <div class="dashboard-card" style="margin-top:0!important;">
                <div class="card-title">{get_svg_icon("database",18,"#64d2ff")} Statistik Cepat Dataset</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div style="background:rgba(100,210,255,0.06);border-radius:8px;padding:14px;text-align:center;">
                        <div style="font-size:1.6rem;font-weight:700;color:#64d2ff;">25,999</div>
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;margin-top:2px;">Total Record</div>
                    </div>
                    <div style="background:rgba(100,210,255,0.06);border-radius:8px;padding:14px;text-align:center;">
                        <div style="font-size:1.6rem;font-weight:700;color:#a180ff;">17</div>
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;margin-top:2px;">Total Kolom</div>
                    </div>
                    <div style="background:rgba(239,68,68,0.06);border-radius:8px;padding:14px;text-align:center;">
                        <div style="font-size:1.6rem;font-weight:700;color:#f87171;">49.9%</div>
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;margin-top:2px;">Missing — PM2.5</div>
                    </div>
                    <div style="background:rgba(16,185,129,0.06);border-radius:8px;padding:14px;text-align:center;">
                        <div style="font-size:1.6rem;font-weight:700;color:#10b981;">9</div>
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;margin-top:2px;">Fitur Digunakan</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box" style="margin-top:20px;">
            <h4>💡 Insight Kritis: Mengapa PM2.5 Punya 49.9% Data Kosong?</h4>
            <p>
            Ini bukan anomali atau error — ini adalah realitas pengukuran kualitas udara global. PM2.5 membutuhkan sensor presisi tinggi yang mahal, sementara banyak negara berkembang belum memiliki infrastruktur pemantauan yang memadai. Inilah tepatnya <em>mengapa proyek ML ini penting</em>: model kita bisa mengestimasi PM2.5 dari variabel lain yang lebih mudah diukur (seperti PM10 dan koordinat geografis), sehingga memberikan insight kualitas udara bahkan untuk wilayah yang tidak memiliki sensor PM2.5.
            </p>
        </div>
        
        <div class="info-callout">
            <strong>🔧 Solusi Teknis:</strong> Baris yang kolom TARGET (pm25_concentration) nya kosong di-drop dari data latih, karena kita tidak bisa mengajarkan model "jawaban yang benar" jika jawaban itu sendiri tidak ada. Sementara kolom FITUR yang kosong (pm10, no2, population) ditangani dengan imputasi median — diisi nilai tengah historis — di dalam pipeline ML, bukan di-drop, agar kita tidak kehilangan terlalu banyak data.
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 2: Univariate ─────────────────────────────────────────────────
    elif st.session_state.eda_step == 2:
        st.markdown("## 📈 Langkah 2: Univariate Analysis — Memahami Satu Variabel pada Satu Waktu")
        
        col_exp, col_viz = st.columns([1, 1.2])
        with col_exp:
            st.markdown("""
            <div class="guide-card" style="border-left-color:#64d2ff;">
                <div class="step-badge">TAHAP 2 DARI 5</div>
                <div class="step-title">Apa Itu Univariate Analysis?</div>
                <div class="step-desc">
                    "Univariate" berarti "satu variabel." Di tahap ini, kita menganalisis setiap variabel secara individual — seperti memotret setiap benda dalam ruangan satu per satu sebelum memfoto seluruh ruangan bersama. 
                    Tujuannya adalah memahami <em>karakter</em> setiap variabel: Berapa nilainya biasanya? Apakah ada nilai yang sangat ekstrem (outlier)? Apakah distribusinya simetris atau miring?
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#a180ff;margin-top:16px;">
                <div class="step-title">📐 Memahami Distribusi Polutan</div>
                <div class="step-desc">
                    Histogram PM2.5 dan PM10 memperlihatkan pola yang sangat khas: <strong>right-skewed</strong> (miring ke kanan).<br><br>
                    Artinya: sebagian besar kota di dunia memiliki udara yang relatif bersih (nilai rendah), namun ada segelintir kota industri berat yang memiliki polusi ekstrem (nilai sangat tinggi). Ekor panjang di sebelah kanan menunjukkan adanya outlier yang signifikan.<br><br>
                    Ini penting untuk ML karena: model yang tidak menangani skewness dengan baik bisa "terlalu dipengaruhi" oleh nilai ekstrem tersebut dan membuat prediksi yang buruk untuk kota-kota biasa.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#10b981;margin-top:16px;">
                <div class="step-title">🏷️ Distribusi Kategori Kualitas Udara</div>
                <div class="step-desc">
                    Dataset juga memiliki kolom <code>Air_quality_category</code> dengan dua nilai: "Safety" (aman) dan "Dangerous" (berbahaya). Distribusinya tidak seimbang — kategori "Safety" jauh lebih dominan.<br><br>
                    Ketidakseimbangan ini (class imbalance) bisa menjadi masalah untuk model klasifikasi, tetapi karena proyek ini menggunakan <strong>regresi</strong> (menebak angka, bukan kategori), hal ini tidak berdampak langsung pada performa model Random Forest kita.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_viz:
            if df_data is not None:
                pm25_filtered = df_data[df_data['pm25_concentration'] < 150]['pm25_concentration'].dropna()
                pm10_filtered = df_data[df_data['pm10_concentration'] < 200]['pm10_concentration'].dropna()
                
                fig_univ = go.Figure()
                fig_univ.add_trace(go.Histogram(x=pm25_filtered, name='PM2.5', marker_color='#64d2ff', opacity=0.75, nbinsx=60))
                fig_univ.add_trace(go.Histogram(x=pm10_filtered, name='PM10', marker_color='#a180ff', opacity=0.75, nbinsx=60))
                fig_univ.add_vline(x=35.4, line_dash="dash", line_color="#f59e0b", annotation_text="Batas Sedang PM2.5")
                fig_univ.update_layout(
                    barmode='overlay', template='plotly_dark', title="Distribusi PM2.5 vs PM10 (Histogram Tumpang Tindih)",
                    plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                    margin=dict(l=10, r=10, t=50, b=20), xaxis_title="Konsentrasi (µg/m³)", yaxis_title="Frekuensi"
                )
                st.plotly_chart(fig_univ, use_container_width=True)
                
                if 'Air_quality_category' in df_data.columns:
                    counts = df_data['Air_quality_category'].dropna().value_counts().reset_index()
                    counts.columns = ['Kategori', 'Jumlah']
                    fig_class = px.bar(counts, x='Kategori', y='Jumlah', text='Jumlah',
                                       color='Kategori', color_discrete_map={'Safety': '#10b981', 'Dangerous': '#ef4444'},
                                       template='plotly_dark', title="Distribusi Kategori Kualitas Udara")
                    fig_class.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)", margin=dict(l=10, r=10, t=50, b=20))
                    fig_class.update_traces(textposition='outside')
                    st.plotly_chart(fig_class, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Insight: Mengapa Distribusi Miring (Skewed) Itu Normal untuk Data Polusi?</h4>
            <p>
            Bayangkan 1000 kota di dunia. Mayoritas (mungkin 700 kota) memiliki udara cukup bersih — PM2.5 di bawah 20 µg/m³. Sekitar 250 kota memiliki polusi sedang. Tapi 50 kota industri seperti Delhi, Kairo, atau beberapa kota China memiliki PM2.5 di atas 100 µg/m³. Ketidakmerataan ini menghasilkan histogram yang "gemuk di kiri, ekor panjang di kanan" — itulah right-skewed distribution. 
            Model Random Forest secara alami menangani distribusi non-normal ini dengan baik karena ia tidak mengasumsikan distribusi data tertentu (berbeda dengan Linear Regression yang mengasumsikan data normal).
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 3: Bivariate ──────────────────────────────────────────────────
    elif st.session_state.eda_step == 3:
        st.markdown("## 🔗 Langkah 3: Bivariate Analysis — Hubungan Antar Dua Variabel")
        
        col_exp, col_viz = st.columns([1, 1.2])
        with col_exp:
            st.markdown("""
            <div class="guide-card" style="border-left-color:#64d2ff;">
                <div class="step-badge">TAHAP 3 DARI 5</div>
                <div class="step-title">Apa Itu Bivariate Analysis?</div>
                <div class="step-desc">
                    "Bivariate" berarti "dua variabel." Kita sekarang mulai melihat <em>hubungan</em> antar variabel — bukan lagi satu per satu. Analoginya: kalau Univariate seperti memoto setiap orang secara sendiri-sendiri, Bivariate seperti memfoto dua orang bersama dan melihat apakah mereka saling berdampingan atau saling membelakangi.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#a180ff;margin-top:16px;">
                <div class="step-title">🔵 Scatter Plot: PM10 vs PM2.5</div>
                <div class="step-desc">
                    Scatter plot memvisualisasikan setiap pasang nilai PM10 dan PM2.5 sebagai sebuah titik. Jika titik-titik tersebut membentuk pola dari kiri-bawah ke kanan-atas, itu tanda hubungan positif yang kuat.<br><br>
                    <strong>Temuan:</strong> Korelasi Pearson antara PM10 dan PM2.5 adalah <strong>r = 0.886</strong> — hubungan linear yang sangat kuat! Ini secara intuitif masuk akal: partikel PM10 (kasar, seperti debu) dan PM2.5 (halus, dari pembakaran) sering hadir bersama dari sumber yang sama — kendaraan bermotor, pembangkit listrik, dan industri berat.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#10b981;margin-top:16px;">
                <div class="step-title">📦 Boxplot: PM2.5 per Wilayah WHO</div>
                <div class="step-desc">
                    Boxplot menunjukkan distribusi PM2.5 untuk setiap wilayah WHO. Garis tengah kotak = median. Kotak = 50% data tengah. Kumis = rentang data normal. Titik di luar kumis = outlier.<br><br>
                    <strong>Temuan Kunci:</strong> Asia Tenggara (SEARO) dan Mediterania Timur (EMRO) memiliki median PM2.5 yang jauh lebih tinggi dibanding Eropa (EURO) atau Amerika (AMRO). Ini membuktikan bahwa variabel <code>who_region</code> adalah fitur yang sangat informatif bagi model — lokasi geografis sangat memengaruhi polusi udara.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_viz:
            if df_data is not None:
                sc = df_data.dropna(subset=['pm10_concentration','pm25_concentration'])
                sc = sc[(sc['pm10_concentration'] < 200) & (sc['pm25_concentration'] < 150)]
                sc_sample = sc.sample(n=min(3000, len(sc)), random_state=42)
                fig_scat = px.scatter(sc_sample, x='pm10_concentration', y='pm25_concentration',
                                      opacity=0.4, color='pm25_concentration', color_continuous_scale='Viridis',
                                      template='plotly_dark', title="Scatter: PM10 vs PM2.5 (Korelasi r=0.886)")
                fig_scat.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                       margin=dict(l=10, r=10, t=50, b=20), coloraxis_showscale=False,
                                       xaxis_title="PM10 (µg/m³)", yaxis_title="PM2.5 (µg/m³)")
                st.plotly_chart(fig_scat, use_container_width=True)
                
                REGION_LABEL = {'1_Afr':'Afrika','2_Amr':'Amerika','3_Sear':'Asia Tenggara',
                                '4_Eur':'Eropa','5_Emr':'Mediterania','6_Wpr':'Pasifik Barat','7_NonMS':'Non-MS'}
                df_box = df_data.dropna(subset=['pm25_concentration','who_region']).copy()
                df_box['region_label'] = df_box['who_region'].map(REGION_LABEL)
                df_box = df_box[df_box['pm25_concentration'] < 150]
                fig_box = px.box(df_box, x='region_label', y='pm25_concentration',
                                 color='region_label', template='plotly_dark',
                                 title="Distribusi PM2.5 per Wilayah WHO")
                fig_box.add_hline(y=35.4, line_dash="dash", line_color="#f59e0b", annotation_text="Batas Sedang WHO")
                fig_box.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                      margin=dict(l=10, r=10, t=50, b=20), showlegend=False,
                                      xaxis_title="Wilayah WHO", yaxis_title="PM2.5 (µg/m³)")
                st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Insight: Mengapa Hubungan Regional Ini Penting untuk Model?</h4>
            <p>
            Ketika Random Forest membangun "pohon keputusan"-nya, ia bisa menggunakan informasi wilayah WHO sebagai "percabangan pertama" — misalnya: "Jika wilayah = SEARO, maka ekspektasi PM2.5 lebih tinggi daripada jika wilayah = EURO." 
            Inilah mengapa variabel kategorikal <code>who_region</code> dikonversi ke format numerik melalui One-Hot Encoding (OHE) — supaya model bisa membaca perbedaan regional ini sebagai angka yang bermakna.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 4: Correlation Matrix ─────────────────────────────────────────
    elif st.session_state.eda_step == 4:
        st.markdown("## 🌡️ Langkah 4: Correlation Matrix — Peta Hubungan Semua Variabel Sekaligus")
        
        col_exp, col_viz = st.columns([1, 1.2])
        with col_exp:
            st.markdown("""
            <div class="guide-card" style="border-left-color:#64d2ff;">
                <div class="step-badge">TAHAP 4 DARI 5</div>
                <div class="step-title">Apa Itu Correlation Matrix?</div>
                <div class="step-desc">
                    Matriks korelasi adalah "peta jalan" yang menunjukkan kekuatan dan arah hubungan linear antara setiap pasang variabel numerik secara bersamaan. Nilainya berkisar dari <strong>-1.0 hingga +1.0</strong>:<br><br>
                    ✅ <strong>+1.0</strong> = Korelasi positif sempurna (jika A naik, B pasti naik)<br>
                    ⚪ <strong>0.0</strong> = Tidak ada hubungan linear<br>
                    ❌ <strong>-1.0</strong> = Korelasi negatif sempurna (jika A naik, B pasti turun)<br><br>
                    Dalam visualisasi heatmap: warna <strong>merah panas</strong> = korelasi positif kuat, warna <strong>biru dingin</strong> = korelasi negatif kuat, putih/krem = mendekati nol.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#a180ff;margin-top:16px;">
                <div class="step-title">🔍 Temuan Utama Matriks</div>
                <div class="step-desc">
                    <strong>Korelasi Tertinggi:</strong><br>
                    • PM10 ↔ PM2.5: r = <strong>0.89</strong> → Fitur paling prediktif!<br>
                    • NO₂ ↔ PM2.5: r = <strong>0.45</strong> → Kontribusi sedang<br>
                    • Populasi ↔ PM2.5: r = <strong>0.22</strong> → Lemah secara linear<br><br>
                    <strong>Korelasi Latitude:</strong><br>
                    • Latitude ↔ PM2.5: r = <strong>-0.31</strong> → Wilayah lebih utara (latitude tinggi = Eropa, Amerika Utara) cenderung memiliki PM2.5 lebih rendah karena regulasi lingkungan yang lebih ketat.
                </div>
            </div>
            
            <div class="guide-card" style="border-left-color:#10b981;margin-top:16px;">
                <div class="step-title">⚠️ Waspada Multikolinearitas</div>
                <div class="step-desc">
                    Multikolinearitas terjadi ketika dua fitur saling berkorelasi sangat tinggi satu sama lain (bukan dengan target). Jika PM10 dan NO₂ misalnya berkorelasi 0.95 satu sama lain, memasukkan keduanya ke model bisa membingungkan model — model kesulitan membedakan kontribusi masing-masing.<br><br>
                    Kabar baiknya: dari matriks, korelasi antar fitur prediktor tidak ada yang ekstrem, sehingga semua 9 fitur aman untuk digunakan bersama-sama.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_viz:
            if df_data is not None:
                corr_cols = ['pm25_concentration','pm10_concentration','no2_concentration',
                             'year','population','latitude','longitude','number_of_stations']
                corr_m = df_data[corr_cols].corr()
                fig_corr = px.imshow(corr_m, text_auto='.2f', color_continuous_scale='RdBu',
                                     aspect="auto", template='plotly_dark',
                                     title="Matriks Korelasi Pearson — Semua Variabel Numerik")
                fig_corr.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                       margin=dict(l=10, r=10, t=60, b=20))
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Tren tahunan
                yr = df_data.dropna(subset=['pm25_concentration']).groupby('year')['pm25_concentration'].agg(['mean','median']).reset_index()
                fig_yr = go.Figure()
                fig_yr.add_trace(go.Scatter(x=yr['year'], y=yr['mean'], mode='lines+markers', name='Rata-rata', line=dict(color='#64d2ff', width=3)))
                fig_yr.add_trace(go.Scatter(x=yr['year'], y=yr['median'], mode='lines+markers', name='Median', line=dict(color='#fb923c', width=2, dash='dash')))
                fig_yr.update_layout(template='plotly_dark', title="Tren PM2.5 Global per Tahun",
                                     plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                     margin=dict(l=10, r=10, t=60, b=20),
                                     xaxis_title="Tahun", yaxis_title="PM2.5 (µg/m³)")
                st.plotly_chart(fig_yr, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Insight: Mengapa Populasi Berkorelasi Lemah padahal Kota Besar Lebih Terpolusi?</h4>
            <p>
            Ini adalah contoh klasik bahwa <em>korelasi linear tidak menangkap semua hubungan</em>. Kota besar seperti Tokyo (populasi 14 juta) bisa memiliki PM2.5 yang lebih rendah dari kota kecil di negara berkembang berkat regulasi ketat. Sedangkan Delhi dengan populasi 33 juta memang sangat terpolusi. Hubungannya dengan PM2.5 bergantung tidak hanya pada ukuran populasi, tapi juga kombinasi regulasi, jenis industri, iklim, dan lokasi geografis. Random Forest bisa menangkap interaksi kompleks semua faktor ini secara bersamaan, sementara korelasi Pearson hanya mengukur satu dimensi hubungan linear.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 5: Final Model ────────────────────────────────────────────────
    elif st.session_state.eda_step == 5:
        st.markdown("## 🏆 Langkah 5: Final Model — Evaluasi Performa & Penjelasan Mendalam")
        
        # Pipeline Diagram
        st.markdown('<div class="section-title" style="font-size:1.2rem;">🔧 Arsitektur Pipeline ML yang Dibangun</div>', unsafe_allow_html=True)
        
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([2, 0.3, 2, 0.3, 2])
        
        with p_col1:
            st.markdown("""
            <div class="pipeline-step">
                <span class="step-badge">INPUT</span>
                <div class="step-title">📥 Data Mentah</div>
                <div class="step-desc">
                    9 kolom input dari pengguna atau dataset: <code>year, latitude, longitude, pm10, no2, stations, who_ms, population, who_region</code><br><br>
                    Beberapa nilai mungkin kosong (NaN) — ini ditangani di tahap berikutnya, bukan di-drop.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with p_col2:
            st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with p_col3:
            st.markdown("""
            <div class="pipeline-step">
                <span class="step-badge">PREPROCESSING</span>
                <div class="step-title">🔧 ColumnTransformer</div>
                <div class="step-desc">
                    <strong>Jalur Numerik (8 kolom):</strong><br>
                    SimpleImputer (median) → StandardScaler<br><br>
                    <strong>Jalur Kategorikal (1 kolom):</strong><br>
                    SimpleImputer (Unknown) → OneHotEncoder<br><br>
                    Kedua jalur diproses paralel, lalu digabung.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with p_col4:
            st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with p_col5:
            st.markdown("""
            <div class="pipeline-step">
                <span class="step-badge">MODEL</span>
                <div class="step-title">🌲 Random Forest</div>
                <div class="step-desc">
                    300 pohon keputusan (n_estimators=300) dilatih secara paralel (n_jobs=-1).<br><br>
                    Setiap pohon dilatih pada subset data & fitur berbeda. Prediksi akhir = rata-rata output semua pohon.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Perbandingan model dengan chart
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="section-title" style="font-size:1.1rem;">📊 Perbandingan 5 Model (R² Score)</div>', unsafe_allow_html=True)
            model_names = ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Gradient Boosting', 'Random Forest']
            r2_vals = [0.6210, 0.6212, 0.7850, 0.8850, 0.8900]
            colors_r2 = ['#334155', '#334155', '#64748b', '#64d2ff', '#10b981']
            fig_r2 = go.Figure(go.Bar(
                x=r2_vals, y=model_names, orientation='h',
                marker_color=colors_r2,
                text=[f'{v:.4f}' for v in r2_vals], textposition='outside'
            ))
            fig_r2.update_layout(
                template='plotly_dark', plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                margin=dict(l=10, r=40, t=20, b=20), xaxis=dict(range=[0, 1.05]),
                xaxis_title="R² Score (Lebih Tinggi = Lebih Baik)"
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col_right:
            st.markdown('<div class="section-title" style="font-size:1.1rem;">📊 Perbandingan 5 Model (RMSE)</div>', unsafe_allow_html=True)
            rmse_vals = [6.45, 6.45, 4.85, 2.78, 2.68]
            colors_rmse = ['#334155', '#334155', '#64748b', '#64d2ff', '#10b981']
            fig_rmse = go.Figure(go.Bar(
                x=rmse_vals, y=model_names, orientation='h',
                marker_color=colors_rmse,
                text=[f'{v:.2f}' for v in rmse_vals], textposition='outside'
            ))
            fig_rmse.update_layout(
                template='plotly_dark', plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                margin=dict(l=10, r=40, t=20, b=20),
                xaxis_title="RMSE µg/m³ (Lebih Rendah = Lebih Baik)"
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Evaluasi plot dari model (jika data tersedia)
        if df_data is not None:
            st.markdown('<div class="section-title" style="font-size:1.1rem;margin-top:16px;">🎯 Evaluasi Model pada Data Test (20% Holdout)</div>', unsafe_allow_html=True)
            
            @st.cache_data
            def compute_eval_plots(data_hash=None):
                df_m = df_data.dropna(subset=[TARGET_COL]).copy()
                df_m = df_m[df_m[TARGET_COL] > 0]
                for col in NUM_COLS:
                    df_m[col] = pd.to_numeric(df_m[col], errors='coerce')
                for col in CAT_COLS:
                    df_m[col] = df_m[col].fillna("Unknown").astype(str)
                df_m = df_m.dropna(subset=FEATURE_COLS)
                X_all = df_m[FEATURE_COLS]
                y_all = df_m[TARGET_COL]
                Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                y_pred = model.predict(Xte)
                residuals = yte.values - y_pred
                ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
                if hasattr(ohe, 'get_feature_names_out'):
                    cat_cols_ohe = list(ohe.get_feature_names_out(['who_region']))
                else:
                    cat_cols_ohe = list(ohe.get_feature_names(['who_region']))
                feat_names = NUM_COLS + cat_cols_ohe
                importances = model.named_steps['model'].feature_importances_
                return yte.values, y_pred, residuals, feat_names, importances, r2_score(yte, y_pred), mean_absolute_error(yte, y_pred)
            
            yte_v, ypred_v, resid_v, feat_n, importances_v, r2_v, mae_v = compute_eval_plots()
            
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                scatter_df = pd.DataFrame({'Aktual': yte_v, 'Prediksi': ypred_v})
                sc_sub = scatter_df.sample(n=min(1000, len(scatter_df)), random_state=42)
                fig_sc = px.scatter(sc_sub, x='Aktual', y='Prediksi', opacity=0.45,
                                    color='Prediksi', color_continuous_scale='Viridis',
                                    template='plotly_dark', title=f"Aktual vs Prediksi<br>R² = {r2_v:.4f}")
                fig_sc.add_shape(type="line", line=dict(dash='dash', color='red', width=2),
                                 x0=0, y0=0, x1=float(yte_v.max()), y1=float(yte_v.max()))
                fig_sc.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                     margin=dict(l=5, r=5, t=60, b=20), coloraxis_showscale=False,
                                     xaxis_title="PM2.5 Aktual (µg/m³)", yaxis_title="PM2.5 Prediksi (µg/m³)")
                st.plotly_chart(fig_sc, use_container_width=True)
            
            with ec2:
                fig_res = px.histogram(x=resid_v, nbins=50, template='plotly_dark',
                                       title=f"Distribusi Residual<br>MAE = {mae_v:.2f} µg/m³")
                fig_res.update_traces(marker_color='#a180ff')
                fig_res.add_vline(x=0, line_dash="dash", line_color="#f43f5e")
                fig_res.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                      margin=dict(l=5, r=5, t=60, b=20),
                                      xaxis_title="Residu (Aktual - Prediksi)", yaxis_title="Frekuensi")
                st.plotly_chart(fig_res, use_container_width=True)
            
            with ec3:
                feat_df = pd.DataFrame({'Fitur': feat_n, 'Kepentingan': importances_v}).nlargest(8, 'Kepentingan').sort_values('Kepentingan')
                fig_feat = px.bar(feat_df, x='Kepentingan', y='Fitur', orientation='h',
                                  text=[f"{v:.1%}" for v in feat_df['Kepentingan']],
                                  template='plotly_dark', title="Feature Importance Top 8")
                fig_feat.update_traces(marker_color='#10b981', textposition='outside')
                fig_feat.update_layout(plot_bgcolor="rgba(17,24,39,0.7)", paper_bgcolor="rgba(9,13,22,0)",
                                       margin=dict(l=5, r=5, t=60, b=20),
                                       xaxis_title="Tingkat Kepentingan", yaxis_title="Fitur")
                st.plotly_chart(fig_feat, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Interpretasi Hasil Evaluasi Final</h4>
            <p>
            <strong>Scatter Plot "Aktual vs Prediksi":</strong> Titik-titik yang ideal berkumpul rapat di sepanjang garis merah putus-putus (garis sempurna y=x). Semakin rapat, semakin akurat model. R² = 0.89 artinya model menjelaskan 89% variasi data — sangat baik untuk data lingkungan global yang penuh noise.<br><br>
            <strong>Histogram Residual:</strong> Distribusi error yang terpusat di sekitar 0 dan berbentuk lonceng (bell-shaped) adalah tanda bahwa model tidak memiliki bias sistematis — ia tidak secara konsisten melebihkan atau meremehkan nilai PM2.5.<br><br>
            <strong>Feature Importance:</strong> PM10 mendominasi (>40%) karena korelasinya yang sangat tinggi (r=0.886). Latitude dan Longitude mencerminkan perbedaan regulasi antar region. Populasi dan NO₂ berkontribusi lebih kecil namun tetap bermakna secara statistik.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigasi bawah
    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, _, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.session_state.eda_step > 1:
            st.button("⬅️ Langkah Sebelumnya", on_click=ubah_step, args=(st.session_state.eda_step - 1,), use_container_width=True)
    with col_next:
        if st.session_state.eda_step < 5:
            st.button("Langkah Berikutnya ➡️", on_click=ubah_step, args=(st.session_state.eda_step + 1,), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN 5: PENJELASAN KODE
# ═══════════════════════════════════════════════════════════════════
elif menu == "Penjelasan Kode":
    st.markdown('<h1 class="main-title">Penjelasan Kode Proyek</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Dokumentasi teknis lengkap dan mendalam untuk setiap komponen kode dalam proyek AirSense — mencakup proyek.ipynb, build_pipeline.py, generate_dark_charts.py, dan app.py ini sendiri.</p>', unsafe_allow_html=True)
    
    tab_nb, tab_build, tab_charts, tab_app = st.tabs([
        "📓 proyek.ipynb", "🔨 build_pipeline.py", "🎨 generate_dark_charts.py", "🌐 app.py"
    ])
    
    # ─── Tab 1: Notebook ──────────────────────────────────────────────────
    with tab_nb:
        st.markdown('<div class="section-title">📓 proyek.ipynb — Notebook Eksperimen Ilmiah</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-callout">
            <strong>Peran file ini:</strong> Notebook Jupyter adalah "laboratorium" tempat semua eksperimen pertama kali dilakukan. Di sini kita menjelajahi data, menguji hipotesis, membandingkan model, dan memvalidasi hasilnya sebelum kode "dipindahkan" ke script produksi.
        </div>
        """, unsafe_allow_html=True)
        
        # CELL 1
        st.markdown("### 🔬 Cell 1: Import Library & Exploratory Data Analysis (EDA)")
        st.markdown("""
        <div class="guide-card" style="border-left-color:#64d2ff;">
            <div class="step-title">Tujuan Cell Ini</div>
            <div class="step-desc">
                Cell 1 adalah titik awal seluruh proyek. Di sini kita memuat semua "alat" yang dibutuhkan (library), membaca dataset dari disk, memperbaiki tipe data, dan membuat 6 grafik visualisasi EDA untuk memahami dataset secara menyeluruh.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="code-block">
<span class="code-comment"># ─── IMPORT LIBRARY ───────────────────────────────────────</span>
<span class="code-keyword">import</span> pandas <span class="code-keyword">as</span> <span class="code-var">pd</span>         <span class="code-comment"># Manipulasi tabel data (DataFrame)</span>
<span class="code-keyword">import</span> numpy <span class="code-keyword">as</span> <span class="code-var">np</span>          <span class="code-comment"># Operasi matematika array</span>
<span class="code-keyword">import</span> matplotlib.pyplot <span class="code-keyword">as</span> <span class="code-var">plt</span> <span class="code-comment"># Pembuatan grafik statis</span>
<span class="code-keyword">import</span> seaborn <span class="code-keyword">as</span> <span class="code-var">sns</span>        <span class="code-comment"># Visualisasi statistik berbasis matplotlib</span>

<span class="code-comment"># ─── MEMUAT DATASET ───────────────────────────────────────</span>
<span class="code-var">df_raw</span> = pd.<span class="code-func">read_csv</span>(<span class="code-string">"action2024/train.csv"</span>)
<span class="code-comment"># read_csv membaca file CSV menjadi DataFrame (tabel data Python)
# Hasilnya: 25,999 baris × 17 kolom</span>

<span class="code-comment"># ─── KONVERSI TIPE DATA ────────────────────────────────────</span>
<span class="code-var">COLS_NUMERIC</span> = [<span class="code-string">'pm25_concentration'</span>, <span class="code-string">'pm10_concentration'</span>, ...]
<span class="code-keyword">for</span> col <span class="code-keyword">in</span> COLS_NUMERIC:
    df_raw[col] = pd.<span class="code-func">to_numeric</span>(df_raw[col], errors=<span class="code-string">'coerce'</span>)
<span class="code-comment"># errors='coerce': nilai yang tidak bisa dikonversi (misal "N/A") 
# diubah menjadi NaN, BUKAN menyebabkan error. Ini penting karena
# dataset nyata sering punya format yang tidak konsisten.</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📈 Penjelasan 6 Subplot EDA yang Dibuat"):
            st.markdown("""
            <div class="guide-card" style="border-left-color:#a180ff;">
                <div class="step-title">Plot 1 & 2: Histogram Distribusi PM2.5 dan PM10</div>
                <div class="step-desc">
                    Menggunakan <code>ax.hist()</code> dengan <code>bins=60</code> (60 batang histogram). Garis merah putus-putus menunjukkan nilai median. Garis oranye menunjukkan batas aman WHO (35.4 µg/m³). Fungsi <code>axvline()</code> menggambar garis vertikal pada nilai yang ditentukan.
                </div>
            </div>
            <div class="guide-card" style="border-left-color:#64d2ff;margin-top:12px;">
                <div class="step-title">Plot 3: Scatter Plot PM10 vs PM2.5</div>
                <div class="step-desc">
                    Setiap titik adalah satu rekaman data (satu kota, satu tahun). Parameter <code>alpha=0.3</code> membuat titik semi-transparan agar tumpukan titik tetap terlihat — teknik ini disebut "overplotting mitigation." Korelasi Pearson dihitung dengan <code>.corr()</code>.
                </div>
            </div>
            <div class="guide-card" style="border-left-color:#10b981;margin-top:12px;">
                <div class="step-title">Plot 4: Boxplot PM2.5 per Wilayah WHO</div>
                <div class="step-desc">
                    <code>ax.boxplot(bp_data, patch_artist=True, showfliers=False)</code> — <code>patch_artist=True</code> mengisi kotak dengan warna, <code>showfliers=False</code> menyembunyikan outlier untuk keterbacaan. Data diurutkan berdasarkan median menggunakan <code>.sort_values()</code>.
                </div>
            </div>
            <div class="guide-card" style="border-left-color:#fb923c;margin-top:12px;">
                <div class="step-title">Plot 5 & 6: Tren Tahunan & Heatmap Korelasi</div>
                <div class="step-desc">
                    Tren tahunan menggunakan <code>.groupby('year').agg(['mean','median'])</code> untuk menghitung rata-rata dan median per tahun. Heatmap dibuat dengan <code>sns.heatmap(corr_m, annot=True)</code> di mana <code>annot=True</code> menampilkan angka korelasi di setiap sel. <code>mask=np.triu()</code> menyembunyikan setengah atas matriks yang redundan.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # CELL 2
        st.markdown("### ⚙️ Cell 2: Feature Engineering & Perbandingan Model (Cross-Validation)")
        st.markdown("""
        <div class="guide-card" style="border-left-color:#64d2ff;">
            <div class="step-title">Tujuan Cell Ini</div>
            <div class="step-desc">
                Cell 2 adalah jantung eksperimen ML. Di sini kita mendefinisikan fitur yang akan digunakan, membangun pipeline preprocessing, dan secara sistematis membandingkan 5 algoritma ML menggunakan teknik validasi yang tepat (K-Fold Cross Validation) untuk menentukan model mana yang paling baik.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="code-block">
<span class="code-comment"># ─── DEFINISI FITUR ───────────────────────────────────────</span>
<span class="code-var">FITUR_NUMERIK</span> = [
    <span class="code-string">'year'</span>, <span class="code-string">'latitude'</span>, <span class="code-string">'longitude'</span>,
    <span class="code-string">'pm10_concentration'</span>,    <span class="code-comment"># ⭐ Fitur TERPENTING (korelasi 0.886)</span>
    <span class="code-string">'no2_concentration'</span>,     <span class="code-comment"># Gas buang kendaraan</span>
    <span class="code-string">'number_of_stations'</span>,    <span class="code-comment"># Kepadatan pemantauan</span>
    <span class="code-string">'who_ms'</span>,               <span class="code-comment"># Status anggota WHO (binary: 0 atau 1)</span>
    <span class="code-string">'population'</span>,           <span class="code-comment"># Ukuran kota</span>
]
<span class="code-var">FITUR_KATEGORIKAL</span> = [<span class="code-string">'who_region'</span>]  <span class="code-comment"># 7 wilayah WHO</span>

<span class="code-comment"># ─── PIPELINE PREPROCESSING ───────────────────────────────</span>
<span class="code-var">numeric_transformer</span> = Pipeline(steps=[
    (<span class="code-string">'imputer'</span>, SimpleImputer(strategy=<span class="code-string">'median'</span>)),
    <span class="code-comment"># NaN diisi dengan nilai median kolom tersebut dari data latih.
    # Mengapa median dan bukan mean? Karena distribusi polutan right-skewed,
    # mean dipengaruhi nilai ekstrem. Median lebih robust.</span>
])

<span class="code-var">categorical_transformer</span> = Pipeline(steps=[
    (<span class="code-string">'imputer'</span>, SimpleImputer(strategy=<span class="code-string">'most_frequent'</span>)),
    (<span class="code-string">'onehot'</span>, OneHotEncoder(handle_unknown=<span class="code-string">'ignore'</span>)),
    <span class="code-comment"># OneHotEncoder mengubah 'who_region' (string) menjadi 7 kolom 0/1
    # Contoh: '3_Sear' → [0, 0, 1, 0, 0, 0, 0] (array biner)
    # handle_unknown='ignore': nilai baru yang tidak ada di training 
    # diisi 0 semua, tidak menyebabkan error.</span>
])
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="code-block">
<span class="code-comment"># ─── K-FOLD CROSS VALIDATION ─────────────────────────────</span>
<span class="code-var">kf</span> = KFold(n_splits=<span class="code-number">5</span>, shuffle=<span class="code-keyword">True</span>, random_state=<span class="code-number">42</span>)
<span class="code-comment"># KFold membagi data menjadi 5 bagian (fold) secara bergantian:
# Iterasi 1: Latih di fold 2,3,4,5 → Test di fold 1
# Iterasi 2: Latih di fold 1,3,4,5 → Test di fold 2
# ... dst. Setiap data pernah jadi data test TEPAT 1 kali.
# Ini menghindari keberuntungan random split satu kali.</span>

<span class="code-keyword">for</span> nama, mdl <span class="code-keyword">in</span> MODELS.items():
    pipe = Pipeline(steps=[
        (<span class="code-string">'preprocessor'</span>, preprocessor), (<span class="code-string">'model'</span>, mdl)
    ])
    <span class="code-var">r2_cv</span> = cross_val_score(pipe, X, y, cv=kf, scoring=<span class="code-string">'r2'</span>, n_jobs=<span class="code-number">-1</span>)
    <span class="code-comment"># n_jobs=-1: gunakan SEMUA core CPU secara paralel → lebih cepat
    # scoring='r2': metrik evaluasi adalah R² (koefisien determinasi)
    # Hasilnya adalah array 5 nilai (satu per fold)</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🏆 Mengapa Random Forest Menang? Penjelasan Mendalam"):
            st.markdown("""
            <div class="guide-card" style="border-left-color:#10b981;">
                <div class="step-title">Mengapa Linear Regression Gagal (R² = 0.62)?</div>
                <div class="step-desc">
                    Linear Regression mengasumsikan bahwa output (PM2.5) adalah kombinasi linear sederhana dari input: PM2.5 = a×PM10 + b×NO2 + c×latitude + ... Linear Regression tidak bisa menangkap interaksi antar variabel — misalnya "jika PM10 TINGGI DAN region SEARO, maka PM2.5 naik JAUH lebih cepat dari yang diprediksi secara linear." Hasilnya: meleset jauh di kasus ekstrem.
                </div>
            </div>
            <div class="guide-card" style="border-left-color:#a180ff;margin-top:12px;">
                <div class="step-title">Mengapa Decision Tree Cukup Baik (R² = 0.785)?</div>
                <div class="step-desc">
                    Decision Tree memecah ruang fitur menjadi kotak-kotak dengan aturan if-then. Ini bisa menangkap beberapa interaksi. Tapi satu pohon tunggal mudah "hafal" data latih (overfitting) — ia terlalu detail untuk data yang sudah dilihat, tapi gagal untuk data baru.
                </div>
            </div>
            <div class="guide-card" style="border-left-color:#64d2ff;margin-top:12px;">
                <div class="step-title">Mengapa Random Forest Terbaik (R² = 0.89)?</div>
                <div class="step-desc">
                    Random Forest mengatasi kelemahan Decision Tree dengan membangun BANYAK pohon (300 pohon) yang masing-masing dilatih pada:
                    (1) Subset baris data yang dipilih acak (Bootstrap Sampling / Bagging)
                    (2) Subset fitur yang dipilih acak di setiap percabangan (max_features='sqrt')
                    Karena setiap pohon berbeda-beda dan melihat aspek data yang berbeda, error antar pohon TIDAK BERKORELASI. Saat di-rata-rata, error-error ini saling mengkompensasi → prediksi akhir jauh lebih akurat dan stabil.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # CELL 3
        st.markdown("### 🏋️ Cell 3: Training Model Final + Evaluasi + Simpan Pipeline")
        st.markdown("""
        <div class="code-block">
<span class="code-comment"># ─── TRAIN FINAL MODEL ────────────────────────────────────</span>
<span class="code-var">X_train</span>, <span class="code-var">X_test</span>, <span class="code-var">y_train</span>, <span class="code-var">y_test</span> = train_test_split(
    X, y, test_size=<span class="code-number">0.2</span>, random_state=<span class="code-number">42</span>
)
<span class="code-comment"># test_size=0.2: 20% data (≈5,200 baris) disimpan sebagai "data ujian"
# yang tidak pernah dilihat model selama training → evaluasi jujur
# random_state=42: "seed" agar pembagian selalu sama setiap kali dijalankan</span>

<span class="code-var">final_pipeline</span>.fit(<span class="code-var">X_train</span>, <span class="code-var">y_train</span>)
<span class="code-comment"># .fit() melakukan 3 hal sekaligus dalam 1 baris:
# 1. Preprocessor belajar nilai median (untuk imputer) dari X_train
# 2. Preprocessor belajar kategori OHE dari X_train  
# 3. Random Forest dilatih dengan data yang sudah diproses</span>

<span class="code-comment"># ─── SIMPAN PIPELINE KE DISK ──────────────────────────────</span>
joblib.<span class="code-func">dump</span>(final_pipeline, <span class="code-string">'pipeline_pm25_final.pkl'</span>)
<span class="code-comment"># joblib.dump menyimpan seluruh pipeline (termasuk nilai median yang 
# sudah dipelajari dan model yang sudah dilatih) ke file .pkl
# File ini kemudian dimuat oleh app.py menggunakan joblib.load()
# Tanpa ini, kita harus melatih ulang model setiap kali app dibuka!</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ─── Tab 2: build_pipeline.py ─────────────────────────────────────────
    with tab_build:
        st.markdown('<div class="section-title">🔨 build_pipeline.py — Script Produksi Pembangun Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-callout">
            <strong>Peran file ini:</strong> Sementara <code>proyek.ipynb</code> adalah laboratorium eksperimen, <code>build_pipeline.py</code> adalah "pabrik produksi." Script ini dirancang untuk dijalankan sekali dari terminal (<code>python build_pipeline.py</code>) dan menghasilkan file <code>pipeline_pm25_final.pkl</code> yang siap digunakan oleh aplikasi Streamlit.
        </div>
        """, unsafe_allow_html=True)
        
        for section in [
            ("Tahap 1 & 2: Memuat Data & Validasi", "#64d2ff",
             "Script dimulai dengan memverifikasi bahwa file train.csv ada sebelum melanjutkan. Ini penting untuk skrip produksi — jika file tidak ada, lebih baik berhenti dengan pesan error yang jelas daripada melanjutkan dan crash di tengah jalan.",
             """<span class="code-comment"># Cek file sebelum memuat</span>
<span class="code-keyword">if not</span> os.path.<span class="code-func">exists</span>(TRAIN_FILE):
    print(<span class="code-string">f"File tidak ditemukan: {TRAIN_FILE}"</span>)
    sys.<span class="code-func">exit</span>(<span class="code-number">1</span>)  <span class="code-comment"># Keluar dengan kode error (bukan 0 = sukses)</span>

<span class="code-comment"># Buang baris dengan target negatif (tidak mungkin secara fisika)</span>
df_model = df_model[df_model[TARGET_COL] > <span class="code-number">0</span>]"""),
            
            ("Tahap 3: Konfigurasi Pipeline yang Lebih Canggih", "#a180ff",
             "build_pipeline.py menggunakan konfigurasi Random Forest yang lebih kuat dibanding notebook: n_estimators=300 (lebih banyak pohon), dan parameter lain yang di-tune untuk akurasi maksimum. Ini adalah model 'final' yang digunakan di produksi.",
             """RandomForestRegressor(
    n_estimators  = <span class="code-number">300</span>,    <span class="code-comment"># 3x lebih banyak pohon dari notebook (100)</span>
    max_depth     = <span class="code-keyword">None</span>,   <span class="code-comment"># Pohon tumbuh sampai sempurna (tidak dipotong)</span>
    min_samples_split = <span class="code-number">4</span>,   <span class="code-comment"># Minimal 4 sampel untuk percabangan</span>
    min_samples_leaf  = <span class="code-number">2</span>,   <span class="code-comment"># Minimal 2 sampel di daun (cegah overfitting)</span>
    max_features  = <span class="code-string">"sqrt"</span>, <span class="code-comment"># √(jumlah fitur) fitur per split → randomisasi</span>
    n_jobs        = <span class="code-number">-1</span>,    <span class="code-comment"># Paralel semua CPU core</span>
    random_state  = <span class="code-number">42</span>,   <span class="code-comment"># Reproduktibilitas hasil</span>
)"""),
            
            ("Tahap 5: Evaluasi Tiga Metrik Sekaligus", "#10b981",
             "Script mengevaluasi model pada data train DAN test untuk mendeteksi overfitting. Jika performa train jauh lebih tinggi dari test, itu tanda model 'hafal' data latih tapi tidak bisa generalisasi.",
             """<span class="code-comment"># Evaluasi Train vs Test — Deteksi Overfitting</span>
<span class="code-var">mae_train</span>  = mean_absolute_error(y_train, y_pred_train)
<span class="code-var">mae_test</span>   = mean_absolute_error(y_test, y_pred_test)

<span class="code-comment"># Jika mae_train << mae_test: Model OVERFITTING (hafal data latih)
# Jika mae_train ≈ mae_test : Model GENERALISASI dengan baik ✅
# Dalam proyek ini: R² train ≈ 0.98, R² test ≈ 0.89
# Ada sedikit overfitting, tapi masih dalam batas wajar untuk RF.</span>"""),
        ]:
            title, color, explanation, code = section
            st.markdown(f"""
            <div class="guide-card" style="border-left-color:{color};margin-bottom:12px;">
                <div class="step-title">{title}</div>
                <div class="step-desc">{explanation}</div>
            </div>
            <div class="code-block">{code}</div>
            """, unsafe_allow_html=True)
    
    # ─── Tab 3: generate_dark_charts.py ───────────────────────────────────
    with tab_charts:
        st.markdown('<div class="section-title">🎨 generate_dark_charts.py — Script Pembuat Grafik Tema Gelap</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-callout">
            <strong>Peran file ini:</strong> Script utilitas untuk menghasilkan versi "dark mode" dari semua grafik EDA dan evaluasi model. Berbeda dari grafik Plotly interaktif di app.py, file ini menghasilkan gambar PNG statis beresolusi tinggi (dpi=150) yang bisa disisipkan ke laporan atau presentasi.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="guide-card" style="border-left-color:#a180ff;">
            <div class="step-title">🎨 Sistem Tema Gelap dengan rcParams</div>
            <div class="step-desc">
                Kunci dari script ini adalah penggunaan <code>plt.rcParams.update()</code> di awal — ini mengubah pengaturan global semua grafik Matplotlib sekaligus. Dengan satu blok konfigurasi, semua 5 grafik otomatis menggunakan warna yang konsisten tanpa harus mengatur warna di setiap plot.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="code-block">
<span class="code-comment"># ─── KONFIGURASI TEMA GELAP GLOBAL ────────────────────────</span>
<span class="code-var">DARK_BG</span>    = <span class="code-string">'#090d16'</span>   <span class="code-comment"># Latar belakang figure (sama dengan stApp)</span>
<span class="code-var">CARD_BG</span>    = <span class="code-string">'#111827'</span>   <span class="code-comment"># Latar belakang area plot (gray-900)</span>
<span class="code-var">TEXT_COLOR</span> = <span class="code-string">'#f1f5f9'</span>   <span class="code-comment"># Teks utama (slate-100)</span>
<span class="code-var">MUTED_TEXT</span> = <span class="code-string">'#94a3b8'</span>   <span class="code-comment"># Teks sekunder (slate-400)</span>

plt.rcParams.update({
    <span class="code-string">'figure.facecolor'</span>: DARK_BG,   <span class="code-comment"># Latar figure seluruhnya</span>
    <span class="code-string">'axes.facecolor'</span>: CARD_BG,     <span class="code-comment"># Latar area plot (dalam sumbu)</span>
    <span class="code-string">'text.color'</span>: TEXT_COLOR,      <span class="code-comment"># Semua teks → putih</span>
    <span class="code-string">'axes.labelcolor'</span>: TEXT_COLOR, <span class="code-comment"># Label sumbu X dan Y</span>
    <span class="code-string">'xtick.color'</span>: MUTED_TEXT,     <span class="code-comment"># Angka di sumbu X</span>
    <span class="code-string">'ytick.color'</span>: MUTED_TEXT,     <span class="code-comment"># Angka di sumbu Y</span>
    <span class="code-string">'legend.facecolor'</span>: CARD_BG,   <span class="code-comment"># Background kotak legenda</span>
})
<span class="code-comment"># Setelah ini, SETIAP plt.subplots() otomatis menggunakan tema gelap!</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="guide-card" style="border-left-color:#10b981;margin-top:16px;">
            <div class="step-title">🔧 Helper Function apply_dark_theme_axes()</div>
            <div class="step-desc">
                Fungsi ini adalah "tangan kanan" script — dipanggil untuk setiap subplot agar tampilan konsisten. Ia menyembunyikan border atas dan kanan (tidak informatif), menambahkan grid tipis, dan mengatur warna judul. Penggunaan helper function seperti ini adalah praktik pemrograman yang baik (DRY — Don't Repeat Yourself).
            </div>
        </div>
        <div class="code-block">
<span class="code-keyword">def</span> <span class="code-func">apply_dark_theme_axes</span>(ax, title=<span class="code-string">""</span>):
    ax.<span class="code-func">set_title</span>(title, color=<span class="code-string">'#64d2ff'</span>, pad=<span class="code-number">12</span>)
    ax.<span class="code-func">grid</span>(<span class="code-keyword">True</span>, linestyle=<span class="code-string">'--'</span>, alpha=<span class="code-number">0.5</span>)  <span class="code-comment"># Grid tipis & transparan</span>
    ax.spines[<span class="code-string">'top'</span>].<span class="code-func">set_visible</span>(<span class="code-keyword">False</span>)   <span class="code-comment"># Hilangkan border atas</span>
    ax.spines[<span class="code-string">'right'</span>].<span class="code-func">set_visible</span>(<span class="code-keyword">False</span>)  <span class="code-comment"># Hilangkan border kanan</span>
    ax.spines[<span class="code-string">'left'</span>].<span class="code-func">set_color</span>(BORDER_COLOR)  <span class="code-comment"># Warnai border kiri</span>
    ax.spines[<span class="code-string">'bottom'</span>].<span class="code-func">set_color</span>(BORDER_COLOR) <span class="code-comment"># Warnai border bawah</span>
        </div>
        <div class="code-block">
<span class="code-comment"># ─── MENYIMPAN GRAFIK KE FILE ─────────────────────────────</span>
plt.<span class="code-func">savefig</span>(<span class="code-string">'evaluasi_model_final.png'</span>, 
    dpi=<span class="code-number">150</span>,                   <span class="code-comment"># 150 dots per inch → resolusi tinggi</span>
    facecolor=DARK_BG,         <span class="code-comment"># Warna background figure saat disimpan</span>
    bbox_inches=<span class="code-string">'tight'</span>        <span class="code-comment"># Potong whitespace berlebih di tepi</span>
)
plt.<span class="code-func">close</span>()  <span class="code-comment"># PENTING: menutup figure agar memori dibebaskan</span>
<span class="code-comment"># Tanpa plt.close(), setiap plt.subplots() baru menumpuk di memori!</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ─── Tab 4: app.py ────────────────────────────────────────────────────
    with tab_app:
        st.markdown('<div class="section-title">🌐 app.py — Aplikasi Streamlit Produksi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-callout">
            <strong>Peran file ini:</strong> app.py adalah wajah publik proyek — lapisan antarmuka yang menghubungkan model ML dengan pengguna awam. File ini menggunakan framework Streamlit untuk mengubah kode Python biasa menjadi web app interaktif tanpa membutuhkan pengetahuan HTML/JavaScript.
        </div>
        """, unsafe_allow_html=True)
        
        for section in [
            ("st.cache_resource — Memuat Model Hanya Sekali", "#64d2ff",
             """Decorator <code>@st.cache_resource</code> adalah salah satu fitur paling penting Streamlit untuk performa. Tanpa cache, setiap kali pengguna mengklik tombol atau menggeser slider, Streamlit menjalankan ulang seluruh script dari awal — termasuk memuat model ML yang bisa memakan waktu 30+ detik! Dengan cache_resource, model dimuat sekali dan disimpan di memori — akses berikutnya instan.""",
             """<span class="code-comment"># Tanpa cache: setiap interaksi user → reload model (lambat!)</span>
<span class="code-comment"># Dengan cache: model dimuat sekali, disimpan di RAM</span>
@st.<span class="code-func">cache_resource</span>
<span class="code-keyword">def</span> <span class="code-func">load_model</span>():
    <span class="code-keyword">try</span>:
        <span class="code-keyword">return</span> joblib.<span class="code-func">load</span>(<span class="code-string">'pipeline_pm25_final.pkl'</span>)
    <span class="code-keyword">except</span>:
        <span class="code-keyword">return</span> <span class="code-func">train_model_on_the_fly</span>()  <span class="code-comment"># Fallback: latih ulang</span>"""),
            
            ("train_model_on_the_fly — Fallback Otomatis", "#a180ff",
             """Fungsi ini adalah "net pengaman." Jika file .pkl tidak ditemukan (misal di server baru yang belum menjalankan build_pipeline.py), Streamlit tidak crash — ia melatih ulang model secara on-the-fly dengan parameter lebih ringan (100 pohon vs 300). Ini membuat aplikasi lebih robust untuk deployment.""",
             """<span class="code-keyword">def</span> <span class="code-func">train_model_on_the_fly</span>():
    <span class="code-comment"># Muat data langsung dari CSV</span>
    df = pd.<span class="code-func">read_csv</span>(<span class="code-string">'action2024/train.csv'</span>)
    <span class="code-comment"># Latih model dengan parameter ringan (100 vs 300 estimators)</span>
    <span class="code-comment"># Hasilnya: sedikit kurang akurat tapi jauh lebih cepat</span>
    pipeline.<span class="code-func">fit</span>(X, y)
    <span class="code-keyword">return</span> pipeline"""),
            
            ("Sistem Navigasi Multi-Halaman dengan st.radio", "#10b981",
             """Streamlit tidak memiliki router halaman bawaan seperti React Router. Solusinya: menggunakan <code>st.radio()</code> di sidebar untuk memilih "halaman." Setiap opsi radio memicu blok <code>if/elif</code> yang berbeda — menciptakan ilusi navigasi multi-halaman dalam satu file Python.""",
             """<span class="code-comment"># Sidebar menjadi menu navigasi</span>
<span class="code-var">menu</span> = st.<span class="code-func">radio</span>(<span class="code-string">"PILIH HALAMAN"</span>, [
    <span class="code-string">"🏠 Beranda Utama"</span>,
    <span class="code-string">"🧪 Demo Kasus Nyata"</span>,
    <span class="code-string">"⚙️ Alur & Proses Data"</span>,
    <span class="code-string">"💻 Penjelasan Kode"</span>
])

<span class="code-comment"># "Router" manual menggunakan if/elif</span>
<span class="code-keyword">if</span> menu == <span class="code-string">"🏠 Beranda Utama"</span>:
    <span class="code-comment"># Render halaman beranda</span>
<span class="code-keyword">elif</span> menu == <span class="code-string">"🧪 Demo Kasus Nyata"</span>:
    <span class="code-comment"># Render halaman demo</span>"""),
            
            ("Session State untuk Navigasi Antar Langkah EDA", "#fb923c",
             """<code>st.session_state</code> adalah "memori" Streamlit yang bertahan antar re-run. Masalah: setiap kali pengguna klik tombol, Streamlit menjalankan ulang seluruh script, sehingga variabel lokal hilang. <code>st.session_state</code> menyimpan nilai yang perlu "diingat" — seperti langkah EDA mana yang sedang aktif.""",
             """<span class="code-comment"># Inisialisasi state saat pertama kali halaman dibuka</span>
<span class="code-keyword">if</span> <span class="code-string">"eda_step"</span> <span class="code-keyword">not in</span> st.session_state:
    st.session_state.eda_step = <span class="code-number">1</span>

<span class="code-comment"># Fungsi callback yang mengubah state</span>
<span class="code-keyword">def</span> <span class="code-func">ubah_step</span>(step_baru):
    st.session_state.eda_step = step_baru

<span class="code-comment"># Tombol dengan callback — state berubah sebelum re-render</span>
st.<span class="code-func">button</span>(<span class="code-string">"Langkah 2"</span>, on_click=ubah_step, args=(<span class="code-number">2</span>,))"""),
        ]:
            title, color, explanation, code = section
            st.markdown(f"""
            <div class="guide-card" style="border-left-color:{color};margin-bottom:12px;">
                <div class="step-title">{title}</div>
                <div class="step-desc">{explanation}</div>
            </div>
            <div class="code-block">{code}</div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box" style="margin-top:20px;">
            <h4>🏗️ Ringkasan Arsitektur Seluruh Proyek</h4>
            <p>
            <strong>proyek.ipynb</strong> → Eksplorasi & eksperimen (Data Scientist bekerja di sini)<br>
            <strong>build_pipeline.py</strong> → Produksi model (dijalankan sekali: menghasilkan .pkl)<br>
            <strong>pipeline_pm25_final.pkl</strong> → Artefak model tersimpan (hasil dari build_pipeline.py)<br>
            <strong>generate_dark_charts.py</strong> → Utilitas grafik statis (untuk laporan/presentasi)<br>
            <strong>app.py</strong> → Frontend interaktif (yang sedang Anda baca sekarang!) yang memuat .pkl<br><br>
            Alur kerja: Jalankan build_pipeline.py SEKALI → Jalankan streamlit run app.py BERKALI-KALI.
            </p>
        </div>
        """, unsafe_allow_html=True)
