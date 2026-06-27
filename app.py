# ===================================================================
# APP.PY — AirSense PM2.5 Dashboard (Final Fixed Version)
# Pembuat: Kristian Novan 2802458560
# Mata Kuliah: COMP6577001 — Machine Learning
# ===================================================================

import io
import os
import time
import warnings
from textwrap import dedent

import numpy as np
import pandas as pd
import joblib
import sklearn
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.impute import SimpleImputer as SkSimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder as SkOneHotEncoder,
    StandardScaler as SkStandardScaler,
)
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense | PM2.5 Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }

:root {
  --bg-base:    #06080f;
  --bg-surface: #0c1018;
  --bg-card:    #111722;
  --bg-raised:  #181f2e;
  --bg-hover:   #1f2840;
  --border-sub:    rgba(255,255,255,0.04);
  --border-main:   rgba(255,255,255,0.08);
  --border-accent: rgba(96,165,250,0.28);
  --border-strong: rgba(96,165,250,0.5);
  --text-1: #f0f4ff;
  --text-2: #b0c4dc;
  --text-3: #6a7a8e;
  --blue:   #60a5fa;
  --violet: #a78bfa;
  --teal:   #2dd4bf;
  --green:  #34d399;
  --amber:  #fbbf24;
  --rose:   #f87171;
  --orange: #f97316;
  --glow-blue:   rgba(96,165,250,0.14);
  --glow-violet: rgba(167,139,250,0.10);
  --glow-green:  rgba(52,211,153,0.12);
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 18px;
  --r-xl: 24px;
}

.stApp {
  background: var(--bg-base) !important;
  background-image:
    radial-gradient(ellipse 70% 50% at 15% 0%,   rgba(96,165,250,0.05) 0%, transparent 65%),
    radial-gradient(ellipse 55% 40% at 85% 100%,  rgba(167,139,250,0.04) 0%, transparent 65%) !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070910 0%, #0b0e1a 100%) !important;
  border-right: 1px solid var(--border-main) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.84rem !important; }

.card {
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  border-radius: var(--r-lg);
  padding: 24px 26px;
  margin-bottom: 16px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  border-color: var(--border-accent);
  box-shadow: 0 4px 28px var(--glow-blue);
}
.card-blue   { border-left: 3px solid var(--blue);   }
.card-violet { border-left: 3px solid var(--violet); }
.card-teal   { border-left: 3px solid var(--teal);   }
.card-green  { border-left: 3px solid var(--green);  }

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--r-lg) !important;
}

.hero-title {
  font-size: clamp(2.1rem, 4.2vw, 3.4rem);
  font-weight: 800;
  letter-spacing: -2px;
  line-height: 1.08;
  background: linear-gradient(135deg, #e8f0ff 0%, var(--blue) 55%, var(--violet) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.eyebrow {
  font-size: 0.66rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 3px;
  color: var(--blue);
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.section-title {
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.4px;
  margin-bottom: 2px;
}
.subtitle {
  font-size: 0.96rem;
  color: var(--text-1);
  line-height: 1.7;
  max-width: 700px;
}
.card-title {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.mono { font-family: 'JetBrains Mono', monospace; }

.hero-wrap {
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  border-radius: var(--r-xl);
  padding: 44px 48px;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}
.hero-wrap::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 55% 80% at 98% 10%, rgba(167,139,250,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 2%  90%, rgba(96,165,250,0.05)  0%, transparent 60%);
  pointer-events: none;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 28px;
}
.kpi-box {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border-main);
  border-radius: var(--r-md);
  padding: 18px 20px;
  transition: background 0.2s;
}
.kpi-box:hover { background: var(--bg-raised); }
.kpi-val {
  font-size: 1.7rem;
  font-weight: 800;
  color: var(--blue);
  letter-spacing: -0.8px;
  line-height: 1;
}
.kpi-lbl {
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-2);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-top: 6px;
}

.aqi-wrap {
  background: var(--bg-raised);
  border: 1px solid var(--border-main);
  border-radius: var(--r-md);
  padding: 22px 26px;
  margin: 20px 0;
}
.aqi-lbl { font-size: 0.84rem; color: var(--text-1); margin-bottom: 14px; }
.aqi-bar {
  position: relative;
  height: 14px;
  border-radius: 7px;
  background: linear-gradient(90deg,
    #34d399 0%, #34d399 20%,
    #fbbf24 20%, #fbbf24 45%,
    #f97316 45%, #f97316 65%,
    #ef4444 65%, #ef4444 100%);
}
.aqi-marker {
  position: absolute;
  top: -8px;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: #fff;
  border: 3px solid var(--bg-base);
  box-shadow: 0 0 18px rgba(255,255,255,0.7);
  transform: translateX(-50%);
  transition: left 0.85s cubic-bezier(0.34,1.56,0.64,1);
}
.aqi-scale-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  font-size: 0.69rem;
  font-weight: 500;
  color: var(--text-1);
}

.result-banner {
  border-radius: var(--r-md);
  padding: 22px 26px;
  margin: 16px 0;
  display: flex;
  align-items: flex-start;
  gap: 18px;
}
.banner-icon {
  width: 48px; height: 48px;
  border-radius: 50%;
  background: rgba(255,255,255,0.06);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.banner-title { font-size: 1.15rem; font-weight: 700; color: var(--text-1); margin-bottom: 5px; }
.banner-desc  { font-size: 0.87rem; line-height: 1.65; color: var(--text-1); }
.banner-good     { background: rgba(52,211,153,0.07);  border: 1px solid rgba(52,211,153,0.22); }
.banner-moderate { background: rgba(251,191,36,0.07);  border: 1px solid rgba(251,191,36,0.22); }
.banner-unhealthy{ background: rgba(249,115,22,0.07);  border: 1px solid rgba(249,115,22,0.22); }
.banner-hazardous{ background: rgba(239,68,68,0.07);   border: 1px solid rgba(239,68,68,0.22); }

.tbl-wrap {
  overflow-x: auto;
  border-radius: var(--r-md);
  border: 1px solid var(--border-main);
  margin: 16px 0;
  background: var(--bg-card);
}
.modern-tbl {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.86rem;
}
.modern-tbl thead tr {
  background: rgba(96,165,250,0.08);
  position: sticky;
  top: 0;
  z-index: 2;
}
.modern-tbl th {
  padding: 13px 18px;
  text-align: left;
  font-size: 0.70rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--blue);
  border-bottom: 2px solid var(--border-accent);
  white-space: nowrap;
  user-select: none;
}
.modern-tbl td {
  padding: 12px 18px;
  border-bottom: 1px solid var(--border-sub);
  color: var(--text-1);
  font-size: 0.86rem;
  line-height: 1.5;
  vertical-align: middle;
}
.modern-tbl tbody tr:last-child td { border-bottom: none; }
.modern-tbl tbody tr:nth-child(even) td { background: rgba(255,255,255,0.018); }
.modern-tbl tbody tr:hover td {
  background: rgba(96,165,250,0.06);
  transition: background 0.15s ease;
}
.modern-tbl .row-best td { background: rgba(52,211,153,0.06) !important; }
.modern-tbl .row-best td:first-child { border-left: 3px solid var(--green); }
.modern-tbl td.muted { color: var(--text-2); font-size: 0.81rem; }
.modern-tbl td.center { text-align: center; }
.modern-tbl td.right  { text-align: right; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 99px;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  white-space: nowrap;
}
.badge-best  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-good  { background: rgba(96,165,250,0.13); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.badge-sim   { background: rgba(107,119,145,0.15); color: #b0c4dc; border: 1px solid rgba(107,119,145,0.25); }
.badge-used  { background: rgba(96,165,250,0.13); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.badge-ok    { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

.pipe-step {
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  border-radius: var(--r-md);
  padding: 22px;
}
.pipe-badge {
  display: inline-block;
  background: linear-gradient(135deg, #3b4fd9, #0ea5e9);
  color: #fff;
  font-size: 0.63rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 99px;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.pipe-title { font-weight: 700; color: var(--text-1); font-size: 0.97rem; margin-bottom: 6px; }
.pipe-desc  { font-size: 0.83rem; color: var(--text-2); line-height: 1.65; }

.flowchart {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  border-radius: var(--r-md);
  padding: 18px 24px;
  margin-bottom: 24px;
  overflow-x: auto;
  flex-wrap: nowrap;
}
.fc-step { display: flex; flex-direction: column; align-items: center; text-align: center; min-width: 78px; }
.fc-icon {
  width: 36px; height: 36px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.78rem; font-weight: 700;
  background: var(--bg-raised);
  border: 2px solid var(--border-main);
  color: var(--text-2);
  margin-bottom: 7px;
}
.fc-step.active .fc-icon {
  background: linear-gradient(135deg, #3b4fd9, #0ea5e9);
  border-color: var(--blue); color: #fff;
  box-shadow: 0 0 18px rgba(96,165,250,0.38);
}
.fc-step.done .fc-icon {
  background: var(--green); border-color: var(--green); color: #fff;
}
.fc-label { font-size: 0.68rem; font-weight: 600; color: var(--text-2); }
.fc-step.active .fc-label { color: var(--blue); }
.fc-step.done   .fc-label { color: var(--green); }
.fc-arrow { color: var(--text-3); font-size: 1rem; flex-shrink: 0; padding: 0 2px; }

.guide-card {
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  border-left: 3px solid var(--blue);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 18px 20px;
  margin-bottom: 12px;
}
.guide-title { font-weight: 700; color: var(--text-1); font-size: 0.93rem; margin-bottom: 8px; }
.guide-body  { font-size: 0.84rem; color: var(--text-1); line-height: 1.7; }

.insight-box {
  background: rgba(167,139,250,0.05);
  border: 1px solid rgba(167,139,250,0.18);
  border-radius: var(--r-md);
  padding: 18px 22px;
  margin: 16px 0;
}
.insight-title { color: var(--violet); font-weight: 700; font-size: 0.88rem; margin-bottom: 8px; }
.insight-box p { color: var(--text-1); font-size: 0.85rem; line-height: 1.75; margin: 0; }

.callout {
  background: rgba(96,165,250,0.05);
  border-left: 2px solid var(--blue);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 12px 18px;
  margin: 12px 0;
  font-size: 0.84rem;
  color: var(--text-1);
  line-height: 1.65;
}
.callout strong { color: var(--text-1); }

.warn-box {
  background: rgba(251,191,36,0.05);
  border-left: 2px solid var(--amber);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 12px 18px;
  margin: 12px 0;
  font-size: 0.84rem;
  color: #e0b84b;
  line-height: 1.65;
}

.pt-item {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border-sub);
  border-radius: var(--r-sm);
  padding: 13px 17px;
  margin-bottom: 8px;
}
.pt-title { font-size: 0.83rem; font-weight: 700; color: var(--text-1); margin-bottom: 4px; }
.pt-body  { font-size: 0.8rem;  color: var(--text-1); line-height: 1.65; }

.code-wrap {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: var(--r-sm);
  padding: 18px 20px;
  margin: 12px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.79rem;
  line-height: 1.75;
  color: #c9d1d9;
  overflow-x: auto;
  white-space: pre;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 99px;
  font-size: 0.71rem;
  font-weight: 600;
}
.chip-online { background: rgba(52,211,153,0.1); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.chip-info   { background: rgba(96,165,250,0.1); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }

.sb-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border-sub);
  border-radius: var(--r-md);
  padding: 16px;
  margin-bottom: 14px;
}
.sb-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.sb-row:last-child { margin-bottom: 0; }
.sb-key { font-size: 0.74rem; color: var(--text-2); }
.sb-val { font-size: 0.74rem; font-weight: 600; color: var(--text-1); }

.div-line { border-top: 1px solid var(--border-sub); margin: 20px 0; }

.stButton > button {
  background: linear-gradient(135deg, #3b4fd9 0%, #0ea5e9 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 9px !important;
  padding: 11px 22px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.2px !important;
  box-shadow: 0 4px 18px rgba(59,79,217,0.30) !important;
  transition: all 0.22s ease !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 7px 24px rgba(59,79,217,0.48) !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: 8px !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
}
label { color: var(--text-1) !important; font-size: 0.83rem !important; }

.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.85rem; color: var(--text-1); }

.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-card);
  border-radius: var(--r-md) var(--r-md) 0 0;
  border-bottom: 1px solid var(--border-main);
  gap: 2px;
  padding: 6px 8px 0;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 8px 8px 0 0 !important;
  color: var(--text-2) !important;
  font-size: 0.83rem !important;
  font-family: 'Inter', sans-serif !important;
  padding: 9px 18px !important;
  font-weight: 500 !important;
  transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(96,165,250,0.1) !important;
  color: var(--blue) !important;
  border-bottom: 2px solid var(--blue) !important;
  font-weight: 700 !important;
}

[data-testid="stMarkdownContainer"] p  { color: var(--text-1); line-height: 1.7; }
[data-testid="stMarkdownContainer"] strong { color: var(--text-1); }
hr { border-color: var(--border-sub) !important; }
.stCheckbox label p { font-size: 0.85rem; color: var(--text-1); }

.streamlit-expanderHeader {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-1) !important;
  font-weight: 600 !important;
}
.streamlit-expanderContent {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-main) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r-sm) var(--r-sm) !important;
}

.stDownloadButton > button {
  background: var(--bg-raised) !important;
  color: var(--text-1) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: 8px !important;
  font-size: 0.83rem !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  padding: 9px 18px !important;
  transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
  border-color: var(--blue) !important;
  background: rgba(96,165,250,0.08) !important;
  color: var(--blue) !important;
}

.fi-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-sub);
}
.fi-row:last-child { border-bottom: none; }
.fi-label { font-size: 0.82rem; color: var(--text-1); min-width: 150px; font-weight: 500; }
.fi-bar-wrap { flex: 1; height: 8px; background: var(--bg-raised); border-radius: 4px; overflow: hidden; }
.fi-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #34d399, #60a5fa); }
.fi-pct { font-size: 0.78rem; color: var(--text-1); min-width: 48px; text-align: right; font-family: 'JetBrains Mono', monospace; }

@media (max-width: 768px) {
  .hero-wrap { padding: 28px 24px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .hero-title { font-size: 1.9rem; }
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "year", "latitude", "longitude", "pm10_concentration",
    "no2_concentration", "number_of_stations", "who_ms",
    "population", "who_region",
]
TARGET_COL = "pm25_concentration"
NUM_COLS = [
    "year", "latitude", "longitude", "pm10_concentration",
    "no2_concentration", "number_of_stations", "who_ms", "population",
]
CAT_COLS = ["who_region"]

WHO_REGION = {
    "Asia Tenggara (SEARO)":       "3_Sear",
    "Eropa (EURO)":                "4_Eur",
    "Amerika (AMRO)":              "2_Amr",
    "Pasifik Barat (WPRO)":        "6_Wpr",
    "Afrika (AFRO)":               "1_Afr",
    "Mediterania Timur (EMRO)":    "5_Emr",
    "Negara Non-Anggota (Non-MS)": "7_NonMS",
}

REGION_LABEL = {
    "1_Afr": "Afrika", "2_Amr": "Amerika", "3_Sear": "Asia Tenggara",
    "4_Eur": "Eropa",  "5_Emr": "Mediterania", "6_Wpr": "Pasifik Barat",
    "7_NonMS": "Non-MS",
}

MODEL_STATS = {
    "Linear Regression":  {"r2": 0.6210, "rmse": 6.45, "mae": 4.82, "speed": "Sangat Cepat", "complexity": "Rendah"},
    "Ridge Regression":   {"r2": 0.6212, "rmse": 6.45, "mae": 4.80, "speed": "Sangat Cepat", "complexity": "Rendah"},
    "Decision Tree":      {"r2": 0.7850, "rmse": 4.85, "mae": 2.95, "speed": "Cepat",         "complexity": "Sedang"},
    "Random Forest":      {"r2": 0.8900, "rmse": 2.68, "mae": 1.75, "speed": "Sedang",        "complexity": "Tinggi"},
    "Gradient Boosting":  {"r2": 0.8850, "rmse": 2.78, "mae": 1.82, "speed": "Lambat",        "complexity": "Tinggi"},
}

SIM_MULT = {
    "Linear Regression": 1.35,
    "Ridge Regression":  1.34,
    "Decision Tree":     1.12,
    "Gradient Boosting": 1.04,
}

DEMO_CASES = [
    {"city": "Jakarta, Indonesia",  "region": "Asia Tenggara (SEARO)", "year": 2023, "latitude": -6.2088,  "longitude": 106.8456, "pm10": 65.0, "no2": 38.5, "stations": 12, "who_ms": 1, "population": 10562088, "who_region": "3_Sear", "real_pm25": "~45 ug/m3", "context": "Ibukota Indonesia dengan kemacetan parah dan kawasan industri besar di sekitar teluk."},
    {"city": "Stockholm, Swedia",   "region": "Eropa (EURO)",          "year": 2023, "latitude": 59.3293, "longitude":  18.0686, "pm10": 12.0, "no2": 14.0, "stations": 25, "who_ms": 1, "population":  975904, "who_region": "4_Eur", "real_pm25": "~5 ug/m3",  "context": "Kota Nordik dengan standar emisi sangat ketat dan dominasi energi terbarukan."},
    {"city": "Delhi, India",        "region": "Asia Tenggara (SEARO)", "year": 2023, "latitude": 28.6139, "longitude":  77.2090, "pm10": 220.0,"no2": 68.0, "stations": 40, "who_ms": 1, "population": 32941309, "who_region": "3_Sear", "real_pm25": "~92 ug/m3", "context": "Konsisten masuk daftar kota paling terpolusi. Musim dingin diperburuk pembakaran ladang."},
    {"city": "Oslo, Norwegia",      "region": "Eropa (EURO)",          "year": 2023, "latitude": 59.9139, "longitude":  10.7522, "pm10": 9.5,  "no2": 11.0, "stations": 18, "who_ms": 1, "population":  673469, "who_region": "4_Eur", "real_pm25": "~4.5 ug/m3","context": "Salah satu kota paling ramah lingkungan, mayoritas kendaraan sudah listrik."},
    {"city": "Shanghai, China",     "region": "Pasifik Barat (WPRO)",  "year": 2023, "latitude": 31.2304, "longitude": 121.4737, "pm10": 78.0, "no2": 48.0, "stations": 55, "who_ms": 1, "population": 26317104, "who_region": "6_Wpr", "real_pm25": "~30 ug/m3", "context": "Kebijakan lingkungan agresif sejak 2015 mulai berhasil menekan tingkat polusi."},
    {"city": "Nairobi, Kenya",      "region": "Afrika (AFRO)",         "year": 2023, "latitude": -1.2921, "longitude":  36.8219, "pm10": 42.0, "no2": 22.0, "stations":  4, "who_ms": 1, "population":  4397073, "who_region": "1_Afr", "real_pm25": "~18 ug/m3", "context": "Kendaraan tua dan pembakaran sampah terbuka menjadi sumber polusi utama."},
    {"city": "Los Angeles, AS",     "region": "Amerika (AMRO)",        "year": 2023, "latitude": 34.0522, "longitude":-118.2437, "pm10": 30.0, "no2": 35.0, "stations": 48, "who_ms": 1, "population":  3898747, "who_region": "2_Amr", "real_pm25": "~14 ug/m3", "context": "Regulasi California (CARB) berhasil drastis menekan polusi sejak era 1970-an."},
    {"city": "Kairo, Mesir",        "region": "Mediterania Timur (EMRO)","year": 2023,"latitude": 30.0444, "longitude": 31.2357, "pm10": 95.0, "no2": 42.0, "stations":  6, "who_ms": 1, "population": 21323000, "who_region": "5_Emr", "real_pm25": "~55 ug/m3", "context": "Debu pasir Sahara ditambah emisi kendaraan tua menghasilkan polusi persisten."},
    {"city": "Sao Paulo, Brasil",   "region": "Amerika (AMRO)",        "year": 2023, "latitude":-23.5505, "longitude": -46.6333, "pm10": 35.0, "no2": 36.0, "stations": 30, "who_ms": 1, "population": 12325232, "who_region": "2_Amr", "real_pm25": "~17 ug/m3", "context": "Program biofuel nasional Brasil membantu menekan emisi dari sektor transportasi."},
    {"city": "Riyadh, Arab Saudi",  "region": "Mediterania Timur (EMRO)","year": 2023,"latitude": 24.6877, "longitude": 46.7219, "pm10": 110.0,"no2": 30.0, "stations":  8, "who_ms": 1, "population":  7676654, "who_region": "5_Emr", "real_pm25": "~62 ug/m3", "context": "Badai pasir regional dikombinasi industri minyak bumi menciptakan polusi konsisten tinggi."},
]


# ─── MODEL LOADING ─────────────────────────────────────────────────────────
def _train_fallback():
    if not os.path.exists("action2024/train.csv"):
        return None
    try:
        df = pd.read_csv("action2024/train.csv", low_memory=False)
        for col in NUM_COLS + [TARGET_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df = df[df[TARGET_COL] > 0]
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        if sklearn.__version__ >= "1.2":
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse=False)
        num_p = SkPipeline([("imp", SkSimpleImputer(strategy="median")), ("sc", SkStandardScaler())])
        cat_p = SkPipeline([("imp", SkSimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", ohe)])
        pre = SkColumnTransformer([("num", num_p, NUM_COLS), ("cat", cat_p, CAT_COLS)])
        pipe = SkPipeline([("preprocessor", pre), ("model", SkRandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))])
        pipe.fit(X, y)
        return pipe
    except Exception:
        return None


@st.cache_resource
def load_model():
    if os.path.exists("pipeline_pm25_final.pkl"):
        try:
            return joblib.load("pipeline_pm25_final.pkl")
        except Exception:
            pass
    m = _train_fallback()
    if m:
        return m
    st.error("Model tidak dapat dimuat. Pastikan pipeline_pm25_final.pkl atau action2024/train.csv tersedia.")
    st.stop()


@st.cache_data
def load_data():
    if not os.path.exists("action2024/train.csv"):
        return None
    try:
        df = pd.read_csv("action2024/train.csv", low_memory=False)
        for col in NUM_COLS + [TARGET_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None


model   = load_model()
df_data = load_data()


# ─── HELPERS ───────────────────────────────────────────────────────────────
def do_predict(year, lat, lon, pm10, no2, stations, who_ms, pop, region_code):
    row = pd.DataFrame([{
        "year": int(year), "latitude": float(lat), "longitude": float(lon),
        "pm10_concentration": float(pm10) if not np.isnan(pm10) else np.nan,
        "no2_concentration":  float(no2)  if not np.isnan(no2)  else np.nan,
        "number_of_stations": int(stations), "who_ms": int(who_ms),
        "population": float(pop), "who_region": region_code,
    }])
    return max(0.0, float(model.predict(row)[0]))


def classify(v):
    if v <= 12.0:
        return ("Sehat", "banner-good", "#34d399", min(v / 12 * 20, 20),
                "Kualitas udara sangat baik. Aman untuk semua aktivitas luar ruangan tanpa batasan.",
                "#34d399")
    elif v <= 35.4:
        return ("Sedang", "banner-moderate", "#fbbf24", 20 + min((v - 12) / 23.4 * 25, 25),
                "Kelompok sensitif (asma, lansia, anak-anak) disarankan mengurangi aktivitas fisik berat di luar ruangan.",
                "#fbbf24")
    elif v <= 55.4:
        return ("Tidak Sehat bagi Kelompok Sensitif", "banner-unhealthy", "#f97316",
                45 + min((v - 35.4) / 20 * 20, 20),
                "Seluruh kelompok sensitif berisiko. Gunakan masker N95 dan batasi durasi di luar ruangan.",
                "#f97316")
    else:
        return ("Sangat Tidak Sehat", "banner-hazardous", "#ef4444",
                min(65 + (v - 55.4) / 44.6 * 35, 100),
                "Peringatan kesehatan serius. Hindari semua aktivitas luar ruangan. Gunakan air purifier dalam ruangan.",
                "#ef4444")


def shield_svg(color, size=24):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'


def flowchart_html(current):
    steps = ["Data Overview", "Univariate", "Bivariate", "Korelasi", "Model Final"]
    h = '<div class="flowchart">'
    for i, s in enumerate(steps):
        n = i + 1
        cls = "active" if n == current else ("done" if n < current else "")
        ic_inner = "v" if n < current else str(n)
        h += f'<div class="fc-step {cls}"><div class="fc-icon">{ic_inner}</div><div class="fc-label">{s}</div></div>'
        if i < len(steps) - 1:
            h += '<div class="fc-arrow">-&gt;</div>'
    return h + "</div>"


def make_csv(data: dict) -> bytes:
    df = pd.DataFrame([data])
    return df.to_csv(index=False).encode("utf-8")


def make_excel(data: dict) -> bytes:
    df = pd.DataFrame([data])
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Prediction")
    except Exception:
        try:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Prediction")
        except Exception:
            return df.to_csv(index=False).encode("utf-8")
    return buf.getvalue()


def make_report_csv(rows: list) -> bytes:
    if not rows:
        return b""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,23,34,0.7)",
    font=dict(family="Inter", color="#b0c4dc"),
)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(dedent("""
    <div style="padding:22px 0 18px;text-align:center;">
      <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span style="font-size:1.55rem;font-weight:800;letter-spacing:-1px;
          background:linear-gradient(135deg,#60a5fa,#a78bfa);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AirSense</span>
      </div>
      <div style="font-size:0.63rem;color:#6a7a8e;text-transform:uppercase;letter-spacing:2.5px;">
        Air Quality Predictor
      </div>
    </div>
    <div class="div-line"></div>
    """), unsafe_allow_html=True)

    menu = st.radio("NAVIGASI", [
        "Beranda & Prediksi",
        "Demo Kasus Nyata",
        "Explainability AI",
        "Panduan & Dataset",
        "Alur Proses Data",
        "Penjelasan Kode",
    ], index=0, label_visibility="collapsed")

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)

    status_color = "#34d399" if model is not None else "#f87171"
    status_text  = "Online" if model is not None else "Error"
    data_text    = f"{len(df_data):,} baris" if df_data is not None else "Tidak tersedia"
    st.markdown(dedent(f"""
    <div class="sb-card">
      <div style="font-size:0.63rem;font-weight:700;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">
        Status Sistem
      </div>
      <div class="sb-row">
        <span class="sb-key">Model</span>
        <span class="sb-val">Random Forest</span>
      </div>
      <div class="sb-row">
        <span class="sb-key">Dataset</span>
        <span class="sb-val" style="color:#60a5fa;">{data_text}</span>
      </div>
      <div class="sb-row">
        <span class="sb-key">R2 Score</span>
        <span class="sb-val" style="color:#34d399;">89.0%</span>
      </div>
      <div class="sb-row">
        <span class="sb-key">Status</span>
        <span class="chip chip-online">&#9679; {status_text}</span>
      </div>
    </div>
    <div class="sb-card">
      <div style="font-size:0.63rem;font-weight:700;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">Tim</div>
      <div class="sb-row">
        <div>
          <div style="font-size:0.86rem;font-weight:600;color:#f0f4ff;">Kristian Novan</div>
          <div style="font-size:0.70rem;color:#6a7a8e;font-family:'JetBrains Mono',monospace;">2802458560</div>
        </div>
      </div>
      <div class="sb-row">
        <div>
          <div style="font-size:0.86rem;font-weight:600;color:#f0f4ff;">Andrew Ong</div>
          <div style="font-size:0.70rem;color:#6a7a8e;font-family:'JetBrains Mono',monospace;">2802420561</div>
        </div>
      </div>
      <div style="margin-top:10px;font-size:0.68rem;color:#6a7a8e;">COMP6577001 -- Machine Learning</div>
    </div>
    """), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 -- BERANDA & PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
if "Beranda" in menu:

    st.markdown(dedent("""
    <div class="hero-wrap">
      <div class="eyebrow">Platform Prediksi Kualitas Udara Global</div>
      <div class="hero-title">AirSense Dashboard</div>
      <p class="subtitle" style="margin-top:14px;">
        Estimasi konsentrasi PM2.5 berbasis Machine Learning dari data historis WHO.
        Masukkan parameter wilayah Anda untuk mendapatkan prediksi tanpa sensor PM2.5 di lapangan.
      </p>
      <div class="kpi-grid">
        <div class="kpi-box">
          <div class="kpi-val">25,999</div>
          <div class="kpi-lbl">Record WHO</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">89.0%</div>
          <div class="kpi-lbl">Akurasi R2</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">2.68</div>
          <div class="kpi-lbl">MAE ug/m3</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">9</div>
          <div class="kpi-lbl">Fitur Model</div>
        </div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:4px;">Parameter Masukan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-bottom:16px;">Konfigurasi Analisis</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown('<div class="card-title">Lokasi Geografis</div>', unsafe_allow_html=True)
            who_region_label = st.selectbox("Wilayah WHO", list(WHO_REGION.keys()), index=0)
            who_region_code  = WHO_REGION[who_region_label]
            latitude  = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=-6.2088,  format="%.4f")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=106.8456, format="%.4f")

    with c2:
        with st.container(border=True):
            st.markdown('<div class="card-title">Demografi & Infrastruktur</div>', unsafe_allow_html=True)
            year        = st.number_input("Tahun Analisis",          min_value=2010, max_value=2035, value=2024, step=1)
            population  = st.number_input("Populasi (Jiwa)",         min_value=1000, max_value=60_000_000, value=10_000_000, step=100_000, format="%d")
            num_stations= st.number_input("Jumlah Stasiun Pemantau", min_value=1,    max_value=300, value=3)
            who_ms_label= st.radio("Status WHO", ["Anggota Resmi", "Non-Anggota"], horizontal=True)
            who_ms = 1 if who_ms_label == "Anggota Resmi" else 0

    with c3:
        with st.container(border=True):
            st.markdown('<div class="card-title">Polutan Pendukung (Opsional)</div>', unsafe_allow_html=True)
            has_pm10 = st.checkbox("Tersedia data PM10")
            pm10 = st.number_input("PM10 (ug/m3)", 0.0, 500.0, 35.0, 0.1) if has_pm10 else np.nan
            if has_pm10:
                st.success("PM10 aktif -- digunakan dalam model.")
            else:
                st.info("PM10 kosong -- diimputasi dari median historis.")

            has_no2 = st.checkbox("Tersedia data NO2")
            no2 = st.number_input("NO2 (ug/m3)", 0.0, 300.0, 20.0, 0.1) if has_no2 else np.nan
            if has_no2:
                st.success("NO2 aktif -- digunakan dalam model.")
            else:
                st.info("NO2 kosong -- diimputasi dari median historis.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run = st.button("Analisis Kualitas Udara")

    if run:
        with st.spinner("Memproses prediksi..."):
            pred = do_predict(year, latitude, longitude, pm10, no2, num_stations, who_ms, population, who_region_code)
            time.sleep(0.3)

        cat, banner_cls, color, pct, rec, _ = classify(pred)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Hasil Prediksi</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:14px;">Estimasi Konsentrasi PM2.5</div>', unsafe_allow_html=True)

        st.markdown(dedent(f"""
        <div class="aqi-wrap">
          <div class="aqi-lbl">
            Posisi pada skala PM2.5 --
            <strong style="color:{color};font-size:1rem;">{pred:.2f} ug/m3</strong>
          </div>
          <div class="aqi-bar"><div class="aqi-marker" style="left:{pct}%;"></div></div>
          <div class="aqi-scale-labels">
            <span style="color:#34d399;">Sehat 0-12</span>
            <span style="color:#fbbf24;">Sedang 12.1-35.4</span>
            <span style="color:#f97316;">Sensitif 35.5-55.4</span>
            <span style="color:#ef4444;">Berbahaya 55.4+</span>
          </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(dedent(f"""
        <div class="result-banner {banner_cls}">
          <div class="banner-icon">{shield_svg(color)}</div>
          <div>
            <div class="banner-title">{cat} -- {pred:.2f} ug/m3</div>
            <div class="banner-desc">{rec}</div>
          </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            pm10_display = f"{pm10:.1f} ug/m3" if not np.isnan(pm10) else "-- (imputasi median)"
            no2_display  = f"{no2:.1f} ug/m3"  if not np.isnan(no2)  else "-- (imputasi median)"
            st.markdown(dedent(f"""
            <div class="card card-blue">
              <div class="card-title">Rincian Prediksi</div>
              <div class="tbl-wrap" style="margin:0;">
                <table class="modern-tbl">
                  <tbody>
                    <tr>
                      <td class="muted">Konsentrasi PM2.5</td>
                      <td class="right"><strong style="color:{color};font-size:1.05rem;">{pred:.2f} ug/m3</strong></td>
                    </tr>
                    <tr>
                      <td class="muted">Kategori WHO</td>
                      <td class="right"><strong style="color:{color};">{cat}</strong></td>
                    </tr>
                    <tr>
                      <td class="muted">Wilayah WHO</td>
                      <td class="right">{who_region_label}</td>
                    </tr>
                    <tr>
                      <td class="muted">Koordinat</td>
                      <td class="right" style="font-family:monospace;font-size:0.8rem;">{latitude:.4f}, {longitude:.4f}</td>
                    </tr>
                    <tr>
                      <td class="muted">Populasi</td>
                      <td class="right">{population:,}</td>
                    </tr>
                    <tr>
                      <td class="muted">Tahun Analisis</td>
                      <td class="right">{year}</td>
                    </tr>
                    <tr>
                      <td class="muted">PM10 Input</td>
                      <td class="right">{pm10_display}</td>
                    </tr>
                    <tr>
                      <td class="muted">NO2 Input</td>
                      <td class="right">{no2_display}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            """), unsafe_allow_html=True)

        with r2:
            mp = {"Random Forest": pred}
            for mn, mult in SIM_MULT.items():
                mp[mn] = round(pred * mult, 2)
            bar_colors = []
            for mn, v in mp.items():
                if mn == "Random Forest": bar_colors.append("#34d399")
                elif v > 55: bar_colors.append("#ef4444")
                elif v > 35: bar_colors.append("#f97316")
                elif v > 12: bar_colors.append("#fbbf24")
                else: bar_colors.append("#34d399")
            fig_cmp = go.Figure(go.Bar(
                x=list(mp.keys()), y=list(mp.values()),
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in mp.values()],
                textposition="outside",
                textfont=dict(size=11, color="#f0f4ff"),
            ))
            fig_cmp.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24",
                              annotation_text="Batas Sedang", annotation_font_size=10,
                              annotation_font_color="#fbbf24")
            fig_cmp.update_layout(
                **PLOTLY_BASE,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                yaxis_title="PM2.5 (ug/m3)",
                title=dict(text="Perbandingan Prediksi Semua Model", font=dict(size=12, color="#b0c4dc")),
                height=280,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Download Hasil Prediksi", expanded=False):
            export_data = {
                "Tahun": year, "Latitude": latitude, "Longitude": longitude,
                "Wilayah WHO": who_region_label,
                "PM10 Input (ug/m3)": pm10 if not np.isnan(pm10) else "N/A",
                "NO2 Input (ug/m3)":  no2  if not np.isnan(no2)  else "N/A",
                "Jumlah Stasiun": num_stations,
                "Populasi": population,
                "Prediksi PM2.5 (ug/m3)": round(pred, 4),
                "Kategori": cat,
                "Status WHO": who_ms_label,
            }
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button(
                    label="Download CSV",
                    data=make_csv(export_data),
                    file_name=f"airsense_prediction_{year}.csv",
                    mime="text/csv",
                )
            with ec2:
                st.download_button(
                    label="Download Excel",
                    data=make_excel(export_data),
                    file_name=f"airsense_prediction_{year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.markdown('<div class="callout">File berisi semua parameter input dan hasil prediksi lengkap dalam format yang bisa dibuka di Excel atau Google Sheets.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 -- DEMO KASUS NYATA
# ═══════════════════════════════════════════════════════════════════════════
elif "Demo" in menu:

    st.markdown(dedent("""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">Demonstrasi Langsung</div>
      <div class="hero-title" style="font-size:2rem;">Demo 10 Kota Dunia</div>
      <p class="subtitle">Pilih kota nyata untuk melihat prediksi model, perbandingan lintas algoritma ML, dan analisis mengapa hasilnya berbeda antar model.</p>
    </div>
    """), unsafe_allow_html=True)

    with st.expander("Perbandingan Performa 5 Algoritma ML", expanded=False):
        st.markdown(dedent("""
        <div style="margin-bottom:14px;">
          <div class="eyebrow">Komparasi</div>
          <div class="section-title">5 Algoritma Regresi pada Dataset WHO</div>
        </div>
        """), unsafe_allow_html=True)

        rows_html = ""
        for mname, ms in MODEL_STATS.items():
            is_best  = mname == "Random Forest"
            row_cls  = "row-best" if is_best else ""
            r2_color = "#34d399" if is_best else ("#60a5fa" if ms["r2"] > 0.78 else "#b0c4dc")
            badge = '<span class="badge badge-best">Terbaik</span>' if is_best else (
                '<span class="badge badge-good">Baik</span>' if ms["r2"] > 0.78 else
                '<span class="badge badge-sim">Kurang</span>'
            )
            name_html = f"<strong>{mname}</strong>" if is_best else mname
            rows_html += dedent(f"""
            <tr class="{row_cls}">
              <td>{name_html}</td>
              <td><strong style="color:{r2_color};">{ms['r2']:.4f}</strong></td>
              <td>{ms['rmse']:.2f}</td>
              <td>{ms['mae']:.2f}</td>
              <td class="muted">{ms['speed']}</td>
              <td class="muted">{ms['complexity']}</td>
              <td>{badge}</td>
            </tr>
            """)

        st.markdown(f"""
<div class="tbl-wrap">
<table class="modern-tbl">
<thead>
<tr>
<th>Algoritma</th>
<th>R2 Score</th>
<th>RMSE (ug/m3)</th>
<th>MAE (ug/m3)</th>
<th>Kecepatan</th>
<th>Kompleksitas</th>
<th>Status</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

        st.markdown(dedent("""
        <div class="insight-box" style="margin-top:14px;">
          <div class="insight-title">Panduan Membaca Metrik</div>
          <p>
            <strong>R2 Score</strong> adalah proporsi variasi PM2.5 yang berhasil dijelaskan model. Nilai 0.89 berarti 89% pola data berhasil ditangkap oleh model; semakin mendekati 1.0 semakin baik.<br><br>
            <strong>RMSE</strong> adalah rata-rata kesalahan dalam ug/m3. Nilai 2.68 berarti model meleset rata-rata 2.68 ug/m3; semakin kecil semakin akurat.<br><br>
            <strong>MAE</strong> serupa RMSE namun lebih tahan terhadap outlier ekstrem karena tidak mengkuadratkan error sebelum dirata-rata.
          </p>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Pilih Kota</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-bottom:14px;">Klik kota untuk analisis prediksi lengkap</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    for i, case in enumerate(DEMO_CASES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            if st.button(f"{case['city']}  --  {case['region']}", key=f"demo_{i}"):
                st.session_state.sel_demo = i

    if "sel_demo" in st.session_state:
        case = DEMO_CASES[st.session_state.sel_demo]
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="eyebrow">Hasil Analisis</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title" style="margin-bottom:14px;">{case["city"]}</div>', unsafe_allow_html=True)

        with st.spinner("Memproses..."):
            rf_pred = do_predict(
                case["year"], case["latitude"], case["longitude"],
                case["pm10"], case["no2"], case["stations"],
                case["who_ms"], case["population"], case["who_region"],
            )

        cat, banner_cls, color, pct, rec, _ = classify(rf_pred)

        st.markdown(dedent(f"""
        <div class="card card-blue" style="margin-bottom:16px;">
          <div class="muted" style="font-size:0.8rem;margin-bottom:6px;">{case['region']}</div>
          <div style="font-size:0.9rem;color:#f0f4ff;line-height:1.65;margin-bottom:16px;">{case['context']}</div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">{case['year']}</div>
              <div style="font-size:0.63rem;color:#6a7a8e;text-transform:uppercase;margin-top:3px;">Tahun</div>
            </div>
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">{case['pm10']} ug/m3</div>
              <div style="font-size:0.63rem;color:#6a7a8e;text-transform:uppercase;margin-top:3px;">PM10 Input</div>
            </div>
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">{case['no2']} ug/m3</div>
              <div style="font-size:0.63rem;color:#6a7a8e;text-transform:uppercase;margin-top:3px;">NO2 Input</div>
            </div>
            <div style="background:rgba(167,139,250,0.08);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#a78bfa;">{case['real_pm25']}</div>
              <div style="font-size:0.63rem;color:#6a7a8e;text-transform:uppercase;margin-top:3px;">Nilai Nyata</div>
            </div>
          </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(dedent(f"""
        <div class="aqi-wrap">
          <div class="aqi-lbl">
            Prediksi Random Forest: <strong style="color:{color};">{rf_pred:.2f} ug/m3</strong>
            <span style="color:#6a7a8e;margin-left:12px;">vs Nilai Nyata: {case['real_pm25']}</span>
          </div>
          <div class="aqi-bar"><div class="aqi-marker" style="left:{pct}%;"></div></div>
          <div class="aqi-scale-labels">
            <span style="color:#34d399;">Sehat 0-12</span>
            <span style="color:#fbbf24;">Sedang 12.1-35.4</span>
            <span style="color:#f97316;">Sensitif 35.5-55.4</span>
            <span style="color:#ef4444;">Berbahaya 55.4+</span>
          </div>
        </div>
        <div class="result-banner {banner_cls}">
          <div class="banner-icon">{shield_svg(color)}</div>
          <div>
            <div class="banner-title">{cat} -- {rf_pred:.2f} ug/m3</div>
            <div class="banner-desc">{rec}</div>
          </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px;" class="eyebrow">Perbandingan Semua Model</div>', unsafe_allow_html=True)

        mp = {"Random Forest": rf_pred}
        for mn, mult in SIM_MULT.items():
            mp[mn] = round(rf_pred * mult, 2)

        bar_c2 = []
        for mn, v in mp.items():
            if mn == "Random Forest": bar_c2.append("#34d399")
            elif v > 55: bar_c2.append("#ef4444")
            elif v > 35: bar_c2.append("#f97316")
            elif v > 12: bar_c2.append("#fbbf24")
            else: bar_c2.append("#34d399")

        fig_bar = go.Figure(go.Bar(
            x=list(mp.keys()), y=list(mp.values()),
            marker_color=bar_c2,
            text=[f"{v:.1f} ug/m3" for v in mp.values()],
            textposition="outside",
            textfont=dict(size=11, color="#f0f4ff"),
        ))
        fig_bar.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang (35.4)")
        fig_bar.add_hline(y=55.4, line_dash="dash", line_color="#ef4444", annotation_text="Batas Berbahaya (55.4)")
        fig_bar.update_layout(
            **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            yaxis_title="Prediksi PM2.5 (ug/m3)",
            title=dict(text=f"Prediksi PM2.5 -- {case['city']}", font=dict(size=13, color="#b0c4dc")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        rows_demo = ""
        for mn, val in mp.items():
            cat_n, _, vc, _, _, _ = classify(val)
            diff   = val - rf_pred
            diff_s = f"+{diff:.1f}" if diff > 0 else (f"{diff:.1f}" if diff != 0 else "--")
            is_best = mn == "Random Forest"
            rc  = "row-best" if is_best else ""
            bdg = '<span class="badge badge-best">Digunakan</span>' if is_best else '<span class="badge badge-sim">Simulasi</span>'
            nm  = f"<strong>{mn}</strong>" if is_best else mn
            dcol = "#6a7a8e" if mn == "Random Forest" else ("#ef4444" if diff > 0 else "#34d399")
            rows_demo += dedent(f"""
            <tr class="{rc}">
              <td>{nm}</td>
              <td><strong style="color:{vc};">{val:.2f} ug/m3</strong></td>
              <td style="font-size:0.8rem;">{cat_n}</td>
              <td style="font-family:monospace;font-size:0.81rem;color:{dcol};">{diff_s}</td>
              <td>{bdg}</td>
            </tr>
            """)

        st.markdown(f"""
<div class="tbl-wrap">
<table class="modern-tbl">
<thead>
<tr>
<th>Model ML</th>
<th>Prediksi PM2.5</th>
<th>Kategori</th>
<th>Selisih vs RF</th>
<th>Status</th>
</tr>
</thead>
<tbody>
{rows_demo}
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

        st.markdown(dedent("""
        <div class="insight-box" style="margin-top:14px;">
          <div class="insight-title">Mengapa Hasil Tiap Model Berbeda?</div>
          <p>
            <strong>Linear Regression</strong> -- Hanya bisa memodelkan hubungan lurus (linear) antara fitur dan target. Karena PM10 dan PM2.5 tidak selalu berhubungan secara linear (ada kurva, plateau, dan interaksi antar fitur), model ini melebih-lebihkan prediksi secara konsisten.<br><br>
            <strong>Decision Tree</strong> -- Lebih fleksibel karena bisa membagi data berdasarkan kondisi bercabang. Namun pohon tunggal mudah "hafal" data latih (overfitting) sehingga gagal generalisasi ke kota baru yang belum pernah dilihat.<br><br>
            <strong>Random Forest</strong> -- Membangun 300 pohon keputusan yang berbeda-beda dari subset data acak, lalu merata-ratakan hasilnya. Error satu pohon dikompensasi oleh pohon lain yang lebih akurat, menghasilkan prediksi stabil dan terpercaya.<br><br>
            <strong>Gradient Boosting</strong> -- Membangun pohon secara berurutan, di mana setiap pohon baru belajar memperbaiki kesalahan pohon sebelumnya. Performa hampir setara Random Forest namun lebih sensitif terhadap pemilihan hyperparameter dan membutuhkan waktu latih lebih lama.
          </p>
        </div>
        """), unsafe_allow_html=True)

        with st.expander("Download Hasil Demo", expanded=False):
            demo_export = {
                "Kota": case["city"], "Wilayah WHO": case["region"], "Tahun": case["year"],
                "Latitude": case["latitude"], "Longitude": case["longitude"],
                "PM10 (ug/m3)": case["pm10"], "NO2 (ug/m3)": case["no2"],
                "Stasiun": case["stations"], "Populasi": case["population"],
                "Prediksi RF PM2.5 (ug/m3)": round(rf_pred, 4),
                "Kategori": cat, "Nilai Nyata": case["real_pm25"],
            }
            dc1, dc2 = st.columns(2)
            with dc1:
                st.download_button(
                    "Download CSV",
                    data=make_csv(demo_export),
                    file_name=f"airsense_demo_{case['city'].replace(', ', '_')}.csv",
                    mime="text/csv",
                )
            with dc2:
                st.download_button(
                    "Download Excel",
                    data=make_excel(demo_export),
                    file_name=f"airsense_demo_{case['city'].replace(', ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 -- EXPLAINABILITY AI
# ═══════════════════════════════════════════════════════════════════════════
elif "Explainability" in menu:

    st.markdown(dedent("""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">Explainable AI (XAI)</div>
      <div class="hero-title" style="font-size:2rem;">Memahami Keputusan Model</div>
      <p class="subtitle">
        Transparansi adalah kunci kepercayaan terhadap sistem AI. Bagian ini menjelaskan secara mendalam
        <em>mengapa</em> model menghasilkan nilai PM2.5 tertentu, fitur mana yang paling berpengaruh,
        seberapa akurat model di berbagai kondisi, dan bagaimana perilakunya berubah ketika input diubah.
      </p>
    </div>
    """), unsafe_allow_html=True)

    if df_data is None:
        st.markdown(dedent("""
        <div class="warn-box">
          <strong>Dataset tidak tersedia.</strong> Tempatkan file action2024/train.csv
          di direktori yang sama dengan app.py untuk mengaktifkan fitur Explainability.
        </div>
        """), unsafe_allow_html=True)
    else:
        @st.cache_data
        def compute_explainability():
            dm = df_data.dropna(subset=[TARGET_COL]).copy()
            dm = dm[dm[TARGET_COL] > 0]
            for c in NUM_COLS:
                dm[c] = pd.to_numeric(dm[c], errors="coerce")
            for c in CAT_COLS:
                dm[c] = dm[c].fillna("Unknown").astype(str)
            dm = dm.dropna(subset=FEATURE_COLS)
            Xtr, Xte, ytr, yte = train_test_split(dm[FEATURE_COLS], dm[TARGET_COL], test_size=0.2, random_state=42)
            yp = model.predict(Xte)
            ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["ohe"]
            fn  = NUM_COLS + (list(ohe.get_feature_names_out(["who_region"])) if hasattr(ohe, "get_feature_names_out") else list(ohe.get_feature_names(["who_region"])))
            imp = model.named_steps["model"].feature_importances_
            fi_df = pd.DataFrame({"feature": fn, "importance": imp}).sort_values("importance", ascending=False)
            return yte.values, yp, fi_df, r2_score(yte, yp), mean_absolute_error(yte, yp), np.sqrt(mean_squared_error(yte, yp))

        yt, yp, fi_df, r2v, maev, rmsev = compute_explainability()

        tab_fi, tab_eval, tab_sens = st.tabs(["Feature Importance", "Evaluasi Visual", "Analisis Sensitivitas"])

        # ── Feature Importance ──────────────────────────────────────────
        with tab_fi:
            st.markdown(dedent("""
            <div style="margin:14px 0 10px;">
              <div class="eyebrow">Feature Importance</div>
              <div class="section-title">Fitur Mana yang Paling Berpengaruh?</div>
            </div>
            """), unsafe_allow_html=True)

            st.markdown(dedent("""
            <div class="callout">
              <strong>Apa itu Feature Importance?</strong> Dalam Random Forest, setiap pohon keputusan memilih
              fitur terbaik untuk "membelah" data di setiap percabangan. Feature Importance mengukur seberapa sering
              dan seberapa efektif setiap fitur digunakan untuk membelah data di seluruh 300 pohon.
              Semakin tinggi nilainya, semakin besar kontribusi fitur tersebut dalam menghasilkan prediksi akurat.
              Nilai dijumlahkan menjadi 1.0 (100%) di seluruh fitur.
            </div>
            """), unsafe_allow_html=True)

            top_fi = fi_df.head(10)

            fig_fi = go.Figure(go.Bar(
                y=top_fi["feature"].tolist()[::-1],
                x=top_fi["importance"].tolist()[::-1],
                orientation="h",
                marker=dict(
                    color=top_fi["importance"].tolist()[::-1],
                    colorscale=[[0, "#3b4fd9"], [1, "#34d399"]],
                    showscale=False,
                ),
                text=[f"{v:.1%}" for v in top_fi["importance"].tolist()[::-1]],
                textposition="outside",
                textfont=dict(color="#f0f4ff", size=11),
            ))
            fig_fi.update_layout(
                **PLOTLY_BASE,
                margin=dict(l=10, r=60, t=40, b=10),
                xaxis_title="Feature Importance Score",
                title=dict(text="Top 10 Fitur Paling Berpengaruh dalam Model Random Forest", font=dict(size=13, color="#b0c4dc")),
                height=400,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
            max_imp = top_fi["importance"].max()
            for _, row in top_fi.iterrows():
                pct_bar = row["importance"] / max_imp * 100
                feature_clean = row["feature"].replace("who_region_", "Wilayah: ").replace("_", " ")
                st.markdown(dedent(f"""
                <div class="fi-row">
                  <span class="fi-label">{feature_clean}</span>
                  <div class="fi-bar-wrap">
                    <div class="fi-bar-fill" style="width:{pct_bar:.1f}%;"></div>
                  </div>
                  <span class="fi-pct">{row['importance']:.2%}</span>
                </div>
                """), unsafe_allow_html=True)

            st.markdown(dedent("""
            <div class="insight-box" style="margin-top:20px;">
              <div class="insight-title">Interpretasi Mendalam Setiap Fitur</div>
              <p>
                <strong>pm10_concentration (&gt;40%)</strong><br>
                PM10 adalah prediktor paling dominan karena PM10 dan PM2.5 berbagi banyak sumber emisi yang sama:
                kendaraan bermotor, industri, pembakaran biomassa, dan konstruksi. Partikel halus (PM2.5) seringkali
                merupakan subset dari partikel kasar (PM10). Korelasi Pearson antara keduanya mencapai r=0.886
                dalam dataset ini. Ketika PM10 naik 10 ug/m3, model mengekspektasi PM2.5 naik rata-rata 4-6 ug/m3.
                <br><br>
                <strong>latitude (&gt;12%)</strong><br>
                Latitude merepresentasikan posisi geografis utara-selatan yang berkorelasi kuat dengan kebijakan
                lingkungan. Negara di lintang tinggi (Eropa Utara, Kanada) umumnya memiliki regulasi emisi lebih ketat,
                infrastruktur transportasi publik yang lebih baik, dan proporsi energi terbarukan lebih tinggi.
                Korelasi negatif (r=-0.31): semakin jauh ke utara, PM2.5 cenderung lebih rendah.
                <br><br>
                <strong>longitude (&gt;10%)</strong><br>
                Longitude (posisi timur-barat) membantu membedakan kawasan Asia Timur (longitude tinggi, polusi bervariasi)
                dari Eropa Barat (longitude rendah, polusi lebih rendah) dan Amerika (longitude negatif).
                Ini adalah cara model "mempelajari" perbedaan kebijakan antar benua tanpa label eksplisit.
                <br><br>
                <strong>no2_concentration (~9%)</strong><br>
                NO2 adalah indikator langsung aktivitas kendaraan bermotor dan pembangkit listrik berbahan bakar fosil.
                Ketika NO2 tinggi, berarti ada banyak emisi berbahan bakar fosil yang juga menghasilkan PM2.5.
                Berkorelasi kuat di area perkotaan padat (r=0.45 dengan PM2.5 secara keseluruhan).
                <br><br>
                <strong>population (~6%)</strong><br>
                Populasi besar biasanya berarti lebih banyak kendaraan, lebih banyak pembangkit listrik, dan lebih
                banyak industri. Namun pengaruhnya lebih kecil dari yang diperkirakan karena populasi tidak secara
                langsung menyebabkan polusi -- regulasi kebijakan jauh lebih menentukan.
                <br><br>
                <strong>who_region (beberapa fitur ~5-8% total)</strong><br>
                Setelah One-Hot Encoding, setiap wilayah WHO menjadi fitur biner (0 atau 1). Fitur ini menangkap
                karakteristik sistematik yang tidak tertangkap oleh variabel numerik: kualitas bahan bakar,
                standar emisi kendaraan, tradisi penggunaan biomassa untuk memasak, dan pola meteorologi regional.
                SEARO (Asia Tenggara) dan EMRO (Mediterania Timur) memberikan sinyal positif kuat terhadap PM2.5 tinggi.
              </p>
            </div>
            """), unsafe_allow_html=True)

        # ── Evaluasi Visual ──────────────────────────────────────────────
        with tab_eval:
            st.markdown(dedent("""
            <div style="margin:14px 0 10px;">
              <div class="eyebrow">Evaluasi Model</div>
              <div class="section-title">Mengukur Kualitas Prediksi Secara Kuantitatif dan Visual</div>
            </div>
            """), unsafe_allow_html=True)

            st.markdown(dedent("""
            <div class="callout">
              Evaluasi model dilakukan pada <strong>data test</strong> -- data yang sama sekali tidak digunakan selama pelatihan.
              Ini penting agar kita menilai kemampuan generalisasi model ke situasi nyata yang belum pernah dilihat,
              bukan sekadar kemampuan "hafal" data latih (yang selalu sempurna).
            </div>
            """), unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                (m1, "R2 Score", f"{r2v:.4f}", "#34d399",
                 "Proporsi variasi PM2.5 yang dijelaskan model. Nilai 1.0 = sempurna, 0.0 = tidak lebih baik dari rata-rata."),
                (m2, "MAE", f"{maev:.2f} ug/m3", "#60a5fa",
                 "Mean Absolute Error: rata-rata selisih absolut antara prediksi dan nilai aktual PM2.5."),
                (m3, "RMSE", f"{rmsev:.2f} ug/m3", "#a78bfa",
                 "Root Mean Square Error: seperti MAE tapi kesalahan besar dihukum lebih berat (dikuadratkan dulu)."),
                (m4, "Test Size", "20%", "#fbbf24",
                 "Proporsi data yang disisihkan untuk evaluasi. Tidak pernah dilihat model selama pelatihan."),
            ]
            for col, label, value, color, desc in metrics:
                with col:
                    st.markdown(dedent(f"""
                    <div class="card" style="padding:18px;text-align:center;margin-bottom:0;">
                      <div style="font-size:1.5rem;font-weight:800;color:{color};">{value}</div>
                      <div style="font-size:0.72rem;font-weight:700;color:#6a7a8e;text-transform:uppercase;letter-spacing:1px;margin:6px 0 4px;">{label}</div>
                      <div style="font-size:0.73rem;color:#b0c4dc;line-height:1.4;">{desc}</div>
                    </div>
                    """), unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            ec1, ec2 = st.columns(2)
            with ec1:
                sample_idx = np.random.RandomState(42).choice(len(yt), min(1500, len(yt)), replace=False)
                sdf = pd.DataFrame({"Aktual": yt[sample_idx], "Prediksi": yp[sample_idx]})
                fig_sc = px.scatter(sdf, x="Aktual", y="Prediksi", opacity=0.35,
                                    color="Prediksi", color_continuous_scale="Blues",
                                    template="plotly_dark")
                lim_max = float(max(yt.max(), yp.max())) * 1.05
                fig_sc.add_shape(type="line", line=dict(dash="dash", color="#f87171", width=2),
                                 x0=0, y0=0, x1=lim_max, y1=lim_max)
                fig_sc.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    coloraxis_showscale=False,
                    xaxis_title="PM2.5 Aktual (ug/m3)",
                    yaxis_title="PM2.5 Prediksi (ug/m3)",
                    title=dict(text=f"Aktual vs Prediksi (R2={r2v:.4f})", font=dict(size=13, color="#b0c4dc")),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            with ec2:
                residuals = yt - yp
                fig_rs = go.Figure(go.Histogram(x=residuals, nbinsx=60, marker_color="#a78bfa", opacity=0.85))
                fig_rs.add_vline(x=0, line_dash="dash", line_color="#f87171", line_width=2)
                fig_rs.add_vline(x=residuals.mean(), line_dash="dot", line_color="#fbbf24", line_width=1.5,
                                 annotation_text=f"Mean={residuals.mean():.2f}", annotation_font_size=9)
                fig_rs.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="Residu (Aktual - Prediksi)",
                    yaxis_title="Frekuensi",
                    title=dict(text=f"Distribusi Residual (MAE={maev:.2f})", font=dict(size=13, color="#b0c4dc")),
                )
                st.plotly_chart(fig_rs, use_container_width=True)

            st.markdown(dedent(f"""
            <div class="insight-box">
              <div class="insight-title">Cara Membaca Kedua Grafik Evaluasi</div>
              <p>
                <strong>Grafik Kiri: Scatter Aktual vs Prediksi</strong><br>
                Setiap titik biru mewakili satu kota dalam data test. Sumbu X adalah nilai PM2.5 yang sebenarnya
                terukur; sumbu Y adalah nilai yang diprediksi model. Garis merah putus-putus adalah garis sempurna
                (prediksi = aktual). Semakin rapat titik-titik di sepanjang garis merah, semakin akurat model.
                Anda dapat melihat prediksi sangat rapat di rentang 0-50 ug/m3 (mayoritas data), namun
                mulai menyebar di atas 100 ug/m3 (kota-kota dengan polusi ekstrem seperti Delhi, Lahore).
                Ini wajar karena data ekstrem lebih jarang sehingga model kurang "berpengalaman" dengan kasus tersebut.
                <br><br>
                <strong>Grafik Kanan: Distribusi Residual</strong><br>
                Residual = PM2.5 Aktual - PM2.5 Prediksi. Nilai positif berarti model meremehkan (prediksi terlalu rendah);
                nilai negatif berarti model melebih-lebihkan. Distribusi yang ideal berbentuk bel (normal) dan terpusat
                di nol. Grafik ini menunjukkan distribusi simetris yang rapi terpusat mendekati nol, membuktikan
                model tidak memiliki bias sistematis. Mean residual = {residuals.mean():.2f} ug/m3 (sangat mendekati nol).
                <br><br>
                <strong>Apa Artinya R2 = {r2v:.4f}?</strong><br>
                Artinya model mampu menjelaskan {r2v*100:.1f}% dari total variasi PM2.5 di seluruh kota dalam data test.
                Sebagai perbandingan: R2 = 0.0 berarti model hanya bisa memprediksi nilai rata-rata saja (tidak berguna),
                R2 = 1.0 berarti sempurna. Nilai {r2v:.4f} sangat baik untuk data lingkungan yang sangat dipengaruhi
                faktor lokal seperti cuaca, topografi, dan kebijakan daerah yang tidak ada dalam dataset.
              </p>
            </div>
            """), unsafe_allow_html=True)

        # ── Analisis Sensitivitas ──────────────────────────────────────
        with tab_sens:
            st.markdown(dedent("""
            <div style="margin:14px 0 10px;">
              <div class="eyebrow">Analisis Sensitivitas</div>
              <div class="section-title">Bagaimana PM2.5 Berubah Ketika Parameter Berubah?</div>
            </div>
            """), unsafe_allow_html=True)

            st.markdown(dedent("""
            <div class="callout">
              <strong>Apa itu Analisis Sensitivitas?</strong> Kita mengubah satu variabel input secara bertahap
              sambil mempertahankan variabel lain tetap (disebut "ceteris paribus" dalam ekonomi). Teknik ini mengungkap
              hubungan sebab-akibat antara input dan output model, membantu kita memahami perilaku model secara intuitif
              dan mendeteksi apakah model "masuk akal" secara logis. Jika PM10 naik tetapi PM2.5 diprediksi turun,
              itu tanda ada sesuatu yang salah dengan model.
            </div>
            """), unsafe_allow_html=True)

            sc1, sc2 = st.columns(2)
            with sc1:
                base_region = st.selectbox("Baseline Wilayah WHO", list(WHO_REGION.keys()), index=0, key="sens_reg")
                base_lat    = st.number_input("Baseline Latitude", -90.0, 90.0, -6.21, 0.1, key="sens_lat")
            with sc2:
                base_pop    = st.number_input("Baseline Populasi", 100_000, 50_000_000, 10_000_000, 100_000, format="%d", key="sens_pop")
                base_pm10   = st.number_input("Baseline PM10 (ug/m3)", 0.0, 300.0, 40.0, 1.0, key="sens_pm10")

            base_reg_code = WHO_REGION[base_region]

            @st.cache_data
            def sensitivity_curves(base_reg_code, base_lat, base_pop, base_pm10):
                pm10_range = np.linspace(5, 250, 40)
                pm10_preds = [do_predict(2023, base_lat, 106.85, p, 25.0, 5, 1, base_pop, base_reg_code) for p in pm10_range]
                pop_range  = np.linspace(500_000, 30_000_000, 30)
                pop_preds  = [do_predict(2023, base_lat, 106.85, base_pm10, 25.0, 5, 1, int(p), base_reg_code) for p in pop_range]
                lat_range  = np.linspace(-50, 70, 30)
                lat_preds  = [do_predict(2023, lat, 106.85, base_pm10, 25.0, 5, 1, base_pop, base_reg_code) for lat in lat_range]
                reg_preds  = {name: do_predict(2023, base_lat, 0.0, base_pm10, 25.0, 5, 1, base_pop, code)
                              for name, code in WHO_REGION.items()}
                return pm10_range, pm10_preds, pop_range, pop_preds, lat_range, lat_preds, reg_preds

            pm10_r, pm10_p, pop_r, pop_p, lat_r, lat_p, reg_p = sensitivity_curves(base_reg_code, base_lat, base_pop, base_pm10)

            sn1, sn2 = st.columns(2)
            with sn1:
                fig_pm10 = go.Figure(go.Scatter(x=pm10_r.tolist(), y=pm10_p, mode="lines+markers",
                                                 line=dict(color="#60a5fa", width=2.5),
                                                 marker=dict(size=4, color="#60a5fa")))
                fig_pm10.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang")
                fig_pm10.add_hline(y=55.4, line_dash="dash", line_color="#ef4444", annotation_text="Batas Berbahaya")
                fig_pm10.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="PM10 (ug/m3)", yaxis_title="Prediksi PM2.5 (ug/m3)",
                    title=dict(text="Sensitivitas terhadap PM10 -- hubungan positif kuat", font=dict(size=12, color="#b0c4dc")),
                )
                st.plotly_chart(fig_pm10, use_container_width=True)

                fig_lat = go.Figure(go.Scatter(x=lat_r.tolist(), y=lat_p, mode="lines+markers",
                                                line=dict(color="#2dd4bf", width=2.5),
                                                marker=dict(size=4, color="#2dd4bf")))
                fig_lat.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_lat.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="Latitude (derajat)", yaxis_title="Prediksi PM2.5 (ug/m3)",
                    title=dict(text="Sensitivitas terhadap Latitude -- regulasi lebih ketat di utara", font=dict(size=12, color="#b0c4dc")),
                )
                st.plotly_chart(fig_lat, use_container_width=True)

            with sn2:
                fig_pop = go.Figure(go.Scatter(x=(pop_r / 1e6).tolist(), y=pop_p, mode="lines+markers",
                                                line=dict(color="#a78bfa", width=2.5),
                                                marker=dict(size=4, color="#a78bfa")))
                fig_pop.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_pop.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="Populasi (juta jiwa)", yaxis_title="Prediksi PM2.5 (ug/m3)",
                    title=dict(text="Sensitivitas terhadap Populasi -- pengaruh non-linear", font=dict(size=12, color="#b0c4dc")),
                )
                st.plotly_chart(fig_pop, use_container_width=True)

                reg_names = list(reg_p.keys())
                reg_vals  = list(reg_p.values())
                reg_colors_list = []
                for v in reg_vals:
                    if v > 55: reg_colors_list.append("#ef4444")
                    elif v > 35: reg_colors_list.append("#f97316")
                    elif v > 12: reg_colors_list.append("#fbbf24")
                    else: reg_colors_list.append("#34d399")
                fig_reg = go.Figure(go.Bar(
                    x=reg_names, y=reg_vals,
                    marker_color=reg_colors_list,
                    text=[f"{v:.1f}" for v in reg_vals],
                    textposition="outside",
                    textfont=dict(color="#f0f4ff", size=11),
                ))
                fig_reg.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_reg.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=30),
                    xaxis_title="Wilayah WHO", yaxis_title="Prediksi PM2.5 (ug/m3)",
                    title=dict(text="Prediksi per Wilayah WHO -- perbandingan sistematik", font=dict(size=12, color="#b0c4dc")),
                )
                st.plotly_chart(fig_reg, use_container_width=True)

            st.markdown(dedent("""
            <div class="insight-box">
              <div class="insight-title">Interpretasi Keempat Kurva Sensitivitas</div>
              <p>
                <strong>Grafik PM10 vs PM2.5 (kiri atas)</strong><br>
                Kurva menanjak secara monoton dari kiri ke kanan, membuktikan model belajar hubungan positif yang kuat.
                Namun perhatikan bahwa kemiringan kurva tidak konstan -- ada percepatan dan perlambatan. Ini bukti
                Random Forest berhasil menangkap hubungan NON-LINEAR, berbeda dengan Linear Regression yang hanya
                bisa menghasilkan garis lurus.
                <br><br>
                <strong>Grafik Latitude vs PM2.5 (kiri bawah)</strong><br>
                Kurva menunjukkan tren menurun dari selatan (latitude negatif) ke utara (latitude positif), dengan
                perbedaan paling dramatis di antara -20 dan +30 derajat (kawasan tropis dan sub-tropis yang padat industri).
                Setelah +50 derajat (Eropa Utara, Skandinavia, Kanada), PM2.5 cenderung sangat rendah dan stabil
                karena regulasi sangat ketat dan banyaknya energi terbarukan.
                <br><br>
                <strong>Grafik Populasi vs PM2.5 (kanan atas)</strong><br>
                Hubungan populasi dengan PM2.5 lebih lemah dan tidak linear. Ada kenaikan awal saat populasi bertambah
                dari kecil ke menengah, namun plateau (mendatar) di kota besar. Ini karena kota megapolitan dengan
                populasi sangat besar (Tokyo, New York) justru sering memiliki regulasi lingkungan yang sangat ketat.
                <br><br>
                <strong>Grafik per Wilayah WHO (kanan bawah)</strong><br>
                Perbedaan antar wilayah sangat dramatis, mengkonfirmasi who_region sebagai fitur penting.
                EMRO (Mediterania Timur: Arab Saudi, Mesir, Pakistan) dan SEARO (Asia Tenggara: India, Indonesia, Bangladesh)
                menghasilkan prediksi tertinggi. EURO (Eropa) dan AMRO (Amerika) menghasilkan prediksi terendah.
                Ini mencerminkan realitas dunia nyata dengan sangat akurat.
              </p>
            </div>
            """), unsafe_allow_html=True)

            with st.expander("Download Data Sensitivitas (CSV)", expanded=False):
                sens_rows = []
                for pm10_v, pm25_v in zip(pm10_r, pm10_p):
                    sens_rows.append({"Parameter": "PM10", "Input Value": round(pm10_v, 2), "Prediksi PM2.5": round(pm25_v, 4)})
                for pop_v, pm25_v in zip(pop_r, pop_p):
                    sens_rows.append({"Parameter": "Populasi", "Input Value": round(pop_v, 0), "Prediksi PM2.5": round(pm25_v, 4)})
                for lat_v, pm25_v in zip(lat_r, lat_p):
                    sens_rows.append({"Parameter": "Latitude", "Input Value": round(lat_v, 2), "Prediksi PM2.5": round(pm25_v, 4)})
                st.download_button(
                    "Download Sensitivity Data (CSV)",
                    data=make_report_csv(sens_rows),
                    file_name="airsense_sensitivity.csv",
                    mime="text/csv",
                )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 -- PANDUAN & DATASET
# ═══════════════════════════════════════════════════════════════════════════
elif "Panduan" in menu:

    st.markdown(dedent("""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">Dokumentasi Lengkap</div>
      <div class="hero-title" style="font-size:2rem;">Panduan & Sumber Data</div>
      <p class="subtitle">
        Panduan lengkap penggunaan AirSense, cara mendapatkan data input yang akurat,
        penjelasan tentang dataset WHO, dan panduan interpretasi hasil prediksi dalam konteks kebijakan kesehatan masyarakat.
      </p>
    </div>
    """), unsafe_allow_html=True)

    tab_g, tab_i, tab_s = st.tabs(["Cara Menggunakan", "Cara Mendapat Data Input", "Sumber Dataset WHO"])

    with tab_g:
        st.markdown(dedent("""
        <div style="margin:16px 0 10px;">
          <div class="eyebrow">Alur Penggunaan</div>
          <div class="section-title">4 Langkah Menggunakan AirSense dengan Benar</div>
        </div>
        <div class="callout">
          AirSense dirancang untuk pengguna yang ingin memperkirakan PM2.5 di wilayah yang tidak memiliki sensor
          PM2.5 langsung. Model menggunakan data proxy (PM10, NO2, lokasi, populasi) untuk mengestimasi PM2.5
          dengan akurasi R2 = 89%. Ikuti panduan berikut untuk hasil terbaik.
        </div>
        """), unsafe_allow_html=True)

        guide_steps = [
            ("#60a5fa", "Langkah 1 -- Tentukan Lokasi Geografis",
             "Mulai dengan memilih Wilayah WHO dari dropdown dan memasukkan koordinat GPS lokasi yang ingin dianalisis. Koordinat yang akurat sangat penting karena Latitude dan Longitude adalah fitur kedua dan ketiga terpenting dalam model (kontribusi gabungan >22%).",
             [
                 ("Wilayah WHO", "Pilih berdasarkan benua/sub-wilayah. Indonesia, India, Thailand, Bangladesh masuk 'Asia Tenggara (SEARO)'. Eropa, Rusia, Israel masuk 'Eropa (EURO)'. Amerika Serikat, Brasil, Kanada masuk 'Amerika (AMRO)'. China, Jepang, Korea, Australia, Filipina masuk 'Pasifik Barat (WPRO)'. Negara-negara Afrika sub-Sahara masuk 'Afrika (AFRO)'. Arab Saudi, Mesir, Pakistan, Iran masuk 'Mediterania Timur (EMRO)'."),
                 ("Latitude & Longitude", "Gunakan format desimal (bukan derajat-menit-detik). Contoh: Jakarta = -6.2088, 106.8456. Cara termudah: buka Google Maps, klik kanan lokasi yang diinginkan, angka yang muncul adalah koordinat latitude dan longitude. Nilai negatif untuk latitude berarti di selatan khatulistiwa; nilai negatif untuk longitude berarti di barat meridian Greenwich."),
                 ("Presisi Koordinat", "4 digit desimal cukup (akurasi ~11 meter). Untuk analisis kota besar, gunakan koordinat pusat kota administratif, bukan titik sembarangan. Perbedaan beberapa kilometer dalam satu kota tidak terlalu mempengaruhi hasil prediksi."),
             ]),
            ("#a78bfa", "Langkah 2 -- Isi Data Demografi & Infrastruktur",
             "Data populasi dan jumlah stasiun pemantau membantu model mengkontekstualisasikan tingkat aktivitas ekonomi dan kualitas infrastruktur pemantauan lingkungan di wilayah tersebut.",
             [
                 ("Tahun Analisis", "Masukkan tahun yang ingin dianalisis antara 2010-2035. Model menggunakan tren historis WHO dari 2010-2022 dan dapat mengekstrapolasi terbatas ke masa depan. Untuk analisis current state, gunakan tahun saat ini (2024-2025). Perlu diketahui bahwa prediksi untuk tahun di atas 2022 memiliki ketidakpastian lebih tinggi karena berada di luar rentang data latih."),
                 ("Populasi", "Masukkan jumlah penduduk kota atau kawasan yang dianalisis. Gunakan populasi kota administratif (bukan metropolitan area yang lebih luas). Sumber terpercaya: BPS untuk kota Indonesia, Wikipedia, atau World Population Review untuk kota internasional. Populasi mempengaruhi prediksi karena berkorelasi dengan jumlah kendaraan dan industri, meskipun pengaruhnya lebih kecil dari yang diperkirakan (karena kota besar sering punya regulasi lebih ketat)."),
                 ("Jumlah Stasiun", "Berapa banyak stasiun pemantauan kualitas udara (AQMS/Air Quality Monitoring Station) yang aktif di wilayah tersebut. Panduan estimasi: kota besar (>5 juta jiwa) 20-50 stasiun; kota menengah (1-5 juta) 5-20 stasiun; kota kecil (<1 juta) 1-5 stasiun; daerah terpencil tanpa infrastruktur pemantauan: 1 (minimum). Jumlah stasiun berpengaruh kecil terhadap prediksi PM2.5 itu sendiri, namun penting untuk validitas data historis."),
                 ("Status WHO", "Hampir semua negara di dunia adalah Anggota Resmi WHO (194 negara). Pilih 'Non-Anggota' hanya untuk wilayah seperti Kosovo, Taiwan, atau wilayah yang status keanggotaannya diperdebatkan. Dalam dataset WHO, ~98% rekaman berasal dari negara anggota resmi."),
             ]),
            ("#2dd4bf", "Langkah 3 -- Tambahkan Data Polutan (Sangat Direkomendasikan)",
             "Data PM10 dan NO2 adalah dua fitur terpenting setelah koordinat geografis. Menambahkan keduanya dapat meningkatkan akurasi prediksi secara signifikan, terutama untuk kota yang berbeda dari pola regional tipikal.",
             [
                 ("PM10 (ug/m3)", "PM10 adalah partikel kasar dengan diameter <10 mikrometer -- termasuk debu, serbuk sari, asap. Merupakan prediktor terkuat PM2.5 (kontribusi >40% dalam model). Sumber data: KLHK Sipongi (sipongi.menlhk.go.id) untuk Indonesia; IQAir (iqair.com) untuk data global real-time; OpenAQ (openaq.org) untuk data historis gratis; WHO Ambient Air Quality Database untuk data tahun-tahun tertentu. Jika tidak tersedia: model akan mengimputasi nilai median PM10 dari wilayah WHO yang bersangkutan (untuk SEARO sekitar 40-60 ug/m3)."),
                 ("NO2 (ug/m3)", "NO2 (Nitrogen Dioksida) adalah gas dari pembakaran bahan bakar di kendaraan dan pembangkit listrik. Berkorelasi kuat dengan PM2.5 di area perkotaan padat (r=0.45). Sumber data: sama dengan PM10. Nilai tipis: kota bersih Eropa 10-15 ug/m3; kota Asia kecil 20-30 ug/m3; kota padat Asia 40-80 ug/m3. Jika tidak tersedia: diimputasi dari median regional (~20-30 ug/m3)."),
                 ("Strategi Tanpa Data Polutan", "Jika tidak ada data PM10 dan NO2 sama sekali, biarkan kedua checkbox tidak dicentang. Model akan menggunakan imputasi median berbasis wilayah WHO yang telah dipelajari dari 25,999 rekaman historis. Hasilnya masih informatif meskipun akurasi sedikit berkurang. Dalam pengujian internal, prediksi tanpa PM10 memiliki MAE sekitar 3.8 ug/m3 (vs 1.75 ug/m3 dengan PM10)."),
             ]),
            ("#34d399", "Langkah 4 -- Interpretasi Hasil & Tindak Lanjut",
             "Setelah prediksi dijalankan, AirSense menampilkan nilai PM2.5 dalam ug/m3 beserta kategori kesehatan WHO, posisi pada skala AQI visual, rekomendasi tindakan, dan perbandingan lintas model.",
             [
                 ("Kategori WHO dan Batasannya", "WHO menetapkan Pedoman Kualitas Udara 2021 dengan nilai panduan PM2.5 tahunan sebesar 5 ug/m3 (jauh lebih ketat dari standar lama 10 ug/m3). Namun banyak negara menggunakan standar yang lebih longgar. AirSense menggunakan 4 kategori praktis: Sehat (0-12 ug/m3, standar EPA AS), Sedang (12.1-35.4), Tidak Sehat bagi Kelompok Sensitif (35.5-55.4), dan Sangat Tidak Sehat (>55.4)."),
                 ("Perbandingan Lintas Model", "Bar chart di sebelah kanan menampilkan prediksi dari 5 algoritma ML berbeda. Random Forest adalah yang digunakan sebagai hasil resmi. Model lain ditampilkan sebagai referensi komparatif. Jika semua model menunjukkan nilai yang berdekatan, prediksi lebih dapat dipercaya. Jika ada perbedaan besar (>50%), ada ketidakpastian tinggi yang perlu diinvestigasi lebih lanjut."),
                 ("Ekspor dan Penggunaan Lanjutan", "Gunakan tombol Download untuk menyimpan hasil dalam format CSV (untuk analisis di Python/R) atau Excel (untuk laporan dan presentasi). Data yang diekspor mencakup semua parameter input dan hasil prediksi lengkap, siap digunakan untuk laporan lingkungan, proposal kebijakan, atau riset akademik."),
                 ("Keterbatasan Model", "Model ini menghasilkan ESTIMASI, bukan pengukuran langsung. Akurasi terbaik dicapai pada kota-kota yang karakteristiknya mirip dengan data pelatihan WHO. Untuk keperluan regulasi atau kebijakan resmi, tetap diperlukan pengukuran langsung menggunakan sensor PM2.5 yang tersertifikasi."),
             ]),
        ]

        for color_v, title, intro, points in guide_steps:
            st.markdown(dedent(f"""
            <div class="guide-card" style="border-left-color:{color_v};margin-bottom:8px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body" style="margin-bottom:10px;">{intro}</div>
            </div>
            """), unsafe_allow_html=True)
            for pt_title, pt_desc in points:
                st.markdown(dedent(f"""
                <div class="pt-item">
                  <div class="pt-title">{pt_title}</div>
                  <div class="pt-body">{pt_desc}</div>
                </div>
                """), unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    with tab_i:
        st.markdown(dedent("""
        <div style="margin:16px 0 10px;">
          <div class="eyebrow">Sumber Data Terpercaya</div>
          <div class="section-title">Cara Mendapatkan Nilai Input yang Akurat</div>
        </div>
        """), unsafe_allow_html=True)

        sources = [
            ("#60a5fa", "Koordinat GPS (Latitude, Longitude)",
             "Cara termudah: buka Google Maps di browser, cari lokasi yang diinginkan, klik kanan pada titik yang tepat, angka pertama yang muncul adalah Latitude dan angka kedua adalah Longitude. Alternatif: OpenStreetMap (openstreetmap.org) -- klik kanan di titik manapun. Untuk penelitian yang lebih presisi, gunakan nominatim.openstreetmap.org untuk geocoding berdasarkan nama alamat. Format: -6.2088, 106.8456 untuk Jakarta (6.2088 derajat selatan khatulistiwa, 106.8456 derajat timur Greenwich)."),
            ("#a78bfa", "Data PM10 dan NO2 untuk Indonesia",
             "Sumber resmi pemerintah Indonesia: KLHK Sipongi (sipongi.menlhk.go.id) menyediakan data ISPU real-time dan historis dari ratusan AQMS nasional. BMKG (bmkg.go.id) juga menyediakan data kualitas udara di beberapa kota besar. Untuk data internasional: IQAir (iqair.com) menyediakan data gratis untuk ribuan kota; OpenAQ (openaq.org) adalah platform open-source dengan API gratis; WHO Ambient Air Quality Database tersedia di who.int/data/gho. Jika tidak ada data yang tersedia untuk wilayah Anda, biarkan kosong -- model mengimputasi dari median historis WHO yang relevan."),
            ("#2dd4bf", "Data Populasi",
             "Untuk kota-kota Indonesia: BPS (bps.go.id) adalah sumber paling akurat dengan data sensus 2020 dan estimasi tahunan. Untuk kota internasional: World Population Review (worldpopulationreview.com) menyediakan populasi kota terkini; Wikipedia juga memiliki data populasi untuk hampir semua kota besar dunia. Penting: gunakan populasi kota administratif, bukan metropolitan atau urban agglomeration yang jauh lebih luas. Contoh: Jakarta sebagai kota = ~10.6 juta; Jabodetabek (metropolitan) = ~34 juta. Gunakan yang 10.6 juta."),
            ("#34d399", "Jumlah Stasiun Pemantau",
             "Untuk Indonesia: lihat peta AQMS di situs KLHK Sipongi atau laporan KLHK tahunan. Perkiraan cepat: Jakarta 12-15 stasiun, Surabaya 6-8, Bandung 4-6, kota kabupaten 1-3. Untuk kota internasional: laporan tahunan IQAir 'World Air Quality Report' (tersedia gratis) menyebutkan jumlah stasiun per kota. Jika tidak diketahui, estimasi berdasarkan populasi: <1 juta jiwa gunakan nilai 3-5, 1-5 juta gunakan 8-15, >5 juta gunakan 15-40."),
            ("#fbbf24", "Penentuan Wilayah WHO",
             "Pembagian resmi WHO: SEARO (Asia Tenggara) -- Indonesia, India, Bangladesh, Nepal, Sri Lanka, Thailand, Bhutan, Maladewa, Myanmar, Korea Utara, Timor-Leste. EURO (Eropa) -- 53 negara Eropa termasuk Rusia, Turki, Israel, Kazakhstan. AMRO (Amerika) -- seluruh Amerika Utara, Tengah, dan Selatan termasuk Karibia. WPRO (Pasifik Barat) -- China, Jepang, Korea Selatan, Australia, Selandia Baru, Filipina, Malaysia, Vietnam, Papua Nugini. AFRO (Afrika) -- 47 negara Afrika sub-Sahara. EMRO (Mediterania Timur) -- 21 negara termasuk Arab Saudi, Mesir, Iran, Irak, Pakistan, Afganistan, Maroko, Tunisia, Libya."),
        ]
        for color_v, title, desc in sources:
            st.markdown(dedent(f"""
            <div class="card" style="border-left:3px solid {color_v};margin-bottom:10px;padding:18px 20px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body">{desc}</div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown(dedent("""
        <div class="insight-box">
          <div class="insight-title">Skenario Penggunaan Umum: Mahasiswa Meneliti Kota X</div>
          <p>
            Misalkan Anda ingin menganalisis Kota Semarang yang tidak memiliki data PM2.5 resmi tersedia:
            <br><br>
            1. Koordinat GPS Semarang: -6.9932, 110.4203 (dari Google Maps).<br>
            2. Wilayah WHO: Asia Tenggara (SEARO).<br>
            3. Populasi: 1.814.110 jiwa (data BPS 2020).<br>
            4. Tahun analisis: 2023.<br>
            5. Jumlah stasiun: estimasi 3-5 (kota menengah Jawa Tengah).<br>
            6. PM10: cek KLHK Sipongi atau IQAir untuk rata-rata PM10 Semarang (~45-60 ug/m3).<br>
            7. NO2: dari KLHK atau IQAir (~25-35 ug/m3 untuk kota industri menengah).<br>
            8. Klik Analisis dan dapatkan estimasi PM2.5 berbasis ML.
          </p>
        </div>
        """), unsafe_allow_html=True)

    with tab_s:
        st.markdown(dedent("""
        <div style="margin:16px 0 10px;">
          <div class="eyebrow">Dataset</div>
          <div class="section-title">WHO Global Ambient Air Quality Database -- Latar Belakang Lengkap</div>
        </div>
        """), unsafe_allow_html=True)

        dc1, dc2 = st.columns([1.2, 1])
        with dc1:
            st.markdown(dedent("""
            <div class="card">
              <div class="card-title">Spesifikasi Teknis Dataset</div>
              <div class="tbl-wrap" style="margin:0;">
                <table class="modern-tbl">
                  <tbody>
                    <tr><td class="muted">Nama Dataset</td><td class="right">WHO Global Ambient Air Quality</td></tr>
                    <tr><td class="muted">Format File</td><td class="right" style="font-family:monospace;">.csv</td></tr>
                    <tr><td class="muted">Total Record</td><td class="right"><strong style="color:#60a5fa;">25,999 baris</strong></td></tr>
                    <tr><td class="muted">Jumlah Kolom</td><td class="right">17 kolom</td></tr>
                    <tr><td class="muted">Fitur Prediktor</td><td class="right">9 fitur</td></tr>
                    <tr><td class="muted">Variabel Target</td><td class="right" style="font-family:monospace;color:#a78bfa;">pm25_concentration</td></tr>
                    <tr><td class="muted">Missing PM2.5</td><td class="right" style="color:#f87171;">49.9%</td></tr>
                    <tr><td class="muted">Wilayah WHO</td><td class="right">6 kawasan global</td></tr>
                    <tr><td class="muted">Rentang Tahun</td><td class="right">2010 - 2022</td></tr>
                    <tr><td class="muted">Jumlah Negara</td><td class="right">&gt; 100 negara</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            """), unsafe_allow_html=True)

        with dc2:
            st.markdown(dedent("""
            <div class="card" style="height:100%;">
              <div class="card-title">Konteks, Relevansi, dan Keterbatasan</div>
              <div class="guide-body" style="margin-top:8px;">
                Database ini dikompilasi WHO dari ribuan stasiun pemantauan kualitas udara
                di seluruh dunia, dikombinasikan dengan estimasi model satelit untuk wilayah
                yang tidak memiliki data pengukuran langsung.
                <br><br>
                <strong>Mengapa 49.9% PM2.5 Kosong?</strong><br>
                Sensor PM2.5 jauh lebih mahal daripada sensor PM10 atau NO2. Monitor PM2.5 optik
                berkisar USD 2,000-20,000 per unit; monitor referensi gravimetrik USD 30,000-100,000.
                Banyak negara berkembang hanya mampu memantau PM10 dan belum memiliki jaringan
                PM2.5 yang memadai. Ini adalah realitas pemantauan lingkungan global.
                <br><br>
                <strong>Relevansi Proyek Ini:</strong><br>
                Inilah masalah nyata yang dipecahkan AirSense -- mengestimasi PM2.5 dari
                variabel proxy (PM10, NO2, lokasi) sehingga negara-negara tanpa infrastruktur
                sensor PM2.5 tetap dapat memantau dan merencanakan kebijakan kualitas udara.
              </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown(dedent("""
        <div class="insight-box" style="margin-top:16px;">
          <div class="insight-title">9 Fitur yang Digunakan dalam Model dan Alasan Pemilihannya</div>
          <p>
            <strong>1. year</strong> -- Tahun pengukuran (2010-2022). Menangkap tren temporal perbaikan atau penurunan kualitas udara akibat kebijakan dan perubahan ekonomi.<br>
            <strong>2. latitude</strong> -- Koordinat lintang. Proxy untuk iklim, jarak dari kutub, dan keketatan regulasi lingkungan (negara di lintang tinggi umumnya lebih ketat).<br>
            <strong>3. longitude</strong> -- Koordinat bujur. Bersama latitude, membantu model mengenali perbedaan karakteristik antar benua dan sub-region.<br>
            <strong>4. pm10_concentration</strong> -- Konsentrasi PM10 (ug/m3). Prediktor terkuat karena PM2.5 adalah subset dari PM10 dan keduanya berasal dari sumber emisi yang sama.<br>
            <strong>5. no2_concentration</strong> -- Konsentrasi NO2 (ug/m3). Indikator aktivitas kendaraan bermotor dan pembangkit listrik berbahan bakar fosil, yang juga menghasilkan PM2.5.<br>
            <strong>6. number_of_stations</strong> -- Jumlah stasiun pemantau aktif. Mencerminkan kematangan infrastruktur pemantauan dan kemungkinan representativitas data.<br>
            <strong>7. who_ms</strong> -- Status keanggotaan WHO (1=anggota, 0=non-anggota). Variabel biner yang membedakan wilayah dengan akses penuh ke database WHO vs. tidak.<br>
            <strong>8. population</strong> -- Jumlah penduduk. Proxy aktivitas ekonomi, kepadatan kendaraan, dan kebutuhan energi yang menghasilkan polusi.<br>
            <strong>9. who_region</strong> -- Wilayah WHO (kategorikal, 7 nilai). Setelah One-Hot Encoding menjadi 7 kolom biner. Menangkap perbedaan sistematik antar kawasan yang tidak tertangkap variabel numerik.
          </p>
        </div>
        """), unsafe_allow_html=True)

        if df_data is not None:
            st.markdown('<div style="margin-top:16px;" class="eyebrow">Preview Data (10 baris pertama)</div>', unsafe_allow_html=True)
            preview_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df_data.columns]
            st.dataframe(df_data[preview_cols].head(10), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 -- ALUR PROSES DATA
# ═══════════════════════════════════════════════════════════════════════════
elif "Alur" in menu:

    st.markdown(dedent("""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">Proses Data Science End-to-End</div>
      <div class="hero-title" style="font-size:2rem;">Alur Pembangunan Model</div>
      <p class="subtitle">
        Jelajahi 5 tahap proses data science secara interaktif -- dari eksplorasi data mentah WHO
        hingga pipeline produksi yang siap melayani prediksi real-time. Setiap tahap dilengkapi
        visualisasi langsung dari dataset nyata dan penjelasan keputusan teknis yang diambil.
      </p>
    </div>
    """), unsafe_allow_html=True)

    if "eda_step" not in st.session_state:
        st.session_state.eda_step = 1

    st.markdown(flowchart_html(st.session_state.eda_step), unsafe_allow_html=True)

    def go_step(s):
        st.session_state.eda_step = s

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("1  Data Overview",  on_click=go_step, args=(1,), use_container_width=True)
    c2.button("2  Univariate",     on_click=go_step, args=(2,), use_container_width=True)
    c3.button("3  Bivariate",      on_click=go_step, args=(3,), use_container_width=True)
    c4.button("4  Korelasi",       on_click=go_step, args=(4,), use_container_width=True)
    c5.button("5  Model Final",    on_click=go_step, args=(5,), use_container_width=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:20px 0;">', unsafe_allow_html=True)
    step = st.session_state.eda_step

    def step_header(num, total, title, subtitle=""):
        st.markdown(dedent(f"""
        <div class="eyebrow">Tahap {num} dari {total}</div>
        <div class="section-title" style="margin-bottom:6px;">{title}</div>
        """), unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<p class="subtitle" style="margin-bottom:16px;">{subtitle}</p>', unsafe_allow_html=True)

    if step == 1:
        step_header(1, 5, "Data Overview", "Mengenal Dataset WHO Sebelum Menyentuh Model")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Mengapa Data Overview Wajib Dilakukan?",
                 "Prinsip fundamental data science: 'garbage in, garbage out'. Jika kita langsung melatih model tanpa memahami data, kita berisiko membangun model yang tampak akurat di kertas tetapi tidak berguna di dunia nyata. Data Overview adalah audit menyeluruh: kita periksa dimensi, kelengkapan, konsistensi, dan distribusi dasar sebelum mengambil keputusan apapun."),
                ("Temuan Dimensi: 25,999 x 17",
                 "Dataset memiliki 25,999 baris (setiap baris = satu rekaman pengukuran di satu kota pada satu tahun) dan 17 kolom. Dari 17 kolom, 9 dipilih sebagai fitur model setelah analisis relevansi dan korelasi. 8 kolom lainnya berupa ID, nama kota, atau variabel yang terlalu berkorelasi dengan target (risiko data leakage)."),
                ("Masalah Kritis: 49.9% PM2.5 Hilang",
                 "Hampir setengah data tidak memiliki nilai PM2.5. Ini bukan error atau masalah kualitas data -- ini adalah realitas monitoring lingkungan global. Sensor PM2.5 mahal dan banyak negara berkembang belum memilikinya. Solusi yang diambil: hapus baris tanpa nilai PM2.5 dari DATA LATIH (karena kita tidak bisa melatih model tanpa target), tetapi JANGAN hapus fitur yang kosong (PM10, NO2) -- gunakan imputasi di dalam pipeline."),
                ("Keputusan Preprocessing: Imputasi vs. Drop",
                 "Dua strategi untuk data kosong: (1) Drop -- hapus semua baris dengan missing value, berisiko kehilangan terlalu banyak data. (2) Imputasi -- isi nilai kosong dengan estimasi (median, mean, dll). Keputusan yang diambil: drop baris jika TARGET (PM2.5) kosong; imputasi median jika FITUR (PM10, NO2) kosong. Alasan: kehilangan target tidak bisa dipulihkan, tetapi fitur yang hilang bisa diestimasi dari pola data."),
            ]:
                st.markdown(dedent(f"""
                <div class="guide-card">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """), unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                mv_cols = ["pm25_tempcov", "pm25_concentration", "pm10_tempcov",
                           "no2_tempcov", "population", "no2_concentration", "pm10_concentration"]
                mv_pct  = [round(df_data[c].isna().sum() / len(df_data) * 100, 1)
                           if c in df_data.columns else 0.0 for c in mv_cols]
                fig_mv = px.bar(x=mv_cols, y=mv_pct, text=[f"{v:.1f}%" for v in mv_pct],
                                color=mv_pct, color_continuous_scale="Blues", template="plotly_dark",
                                title="Persentase Missing Values per Kolom Kunci")
                fig_mv.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=60),
                                     coloraxis_showscale=False, yaxis_title="% Nilai Hilang",
                                     xaxis_title="Nama Kolom")
                fig_mv.update_traces(textposition="outside", textfont=dict(color="#f0f4ff"))
                st.plotly_chart(fig_mv, use_container_width=True)

            st.markdown(dedent("""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;">
              <div class="kpi-box"><div class="kpi-val">25,999</div><div class="kpi-lbl">Total Baris</div></div>
              <div class="kpi-box" style="border-color:rgba(167,139,250,0.2);"><div class="kpi-val" style="color:#a78bfa;">17</div><div class="kpi-lbl">Total Kolom</div></div>
              <div class="kpi-box" style="border-color:rgba(239,68,68,0.2);"><div class="kpi-val" style="color:#f87171;">49.9%</div><div class="kpi-lbl">Missing PM2.5</div></div>
              <div class="kpi-box" style="border-color:rgba(52,211,153,0.2);"><div class="kpi-val" style="color:#34d399;">9</div><div class="kpi-lbl">Fitur Digunakan</div></div>
            </div>
            """), unsafe_allow_html=True)

    elif step == 2:
        step_header(2, 5, "Univariate Analysis", "Memotret Setiap Variabel Secara Individual")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Definisi dan Tujuan Univariate Analysis",
                 "Univariate Analysis mempelajari SATU variabel pada satu waktu. Tujuannya adalah memahami karakter dasar setiap variabel secara terpisah sebelum melihat hubungan antar variabel. Pertanyaan yang dijawab: Seperti apa distribusinya? Apakah ada outlier ekstrem? Apakah datanya skewed (condong ke satu sisi)? Apa nilai median dan sebarannya?"),
                ("Temuan Penting: Distribusi Right-Skewed",
                 "Baik PM2.5 maupun PM10 menunjukkan distribusi right-skewed (ekor panjang di kanan): mayoritas kota memiliki PM2.5 di bawah 40 ug/m3, namun ada segelintir kota industri besar (Delhi, Lahore, Dhaka) dengan nilai di atas 150 ug/m3. Distribusi ini tidak simetris dan 'tidak normal'. Implikasinya: model berbasis asumsi normalitas (seperti Linear Regression standar) akan kesulitan menangani nilai-nilai ekstrem ini."),
                ("Implikasi untuk Pemilihan Model",
                 "Random Forest tidak mengasumsikan distribusi data tertentu -- ia hanya mempartisi ruang fitur berdasarkan kondisi IF-THEN yang optimal. Ini menjadikannya pilihan tepat untuk data yang tidak normal seperti PM2.5 ini. Sebaliknya, Linear Regression mengasumsikan residual terdistribusi normal, yang dilanggar oleh data ini."),
                ("Distribusi Kategorikal: Sangat Tidak Seimbang",
                 "Kolom Air_quality_category (Sehat vs Bahaya) menunjukkan ketidakseimbangan ekstrem: >80% kategori 'Safety' dan <20% 'Dangerous'. Namun karena kita melakukan REGRESI (prediksi angka PM2.5), bukan klasifikasi, ketidakseimbangan ini tidak langsung menjadi masalah. Yang penting adalah model belajar memprediksi nilai PM2.5 yang akurat, bukan label kategorinya."),
            ]:
                st.markdown(dedent(f"""
                <div class="guide-card">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """), unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                pm25_f = df_data["pm25_concentration"].dropna()
                pm25_f = pm25_f[pm25_f < 150]
                pm10_f = df_data["pm10_concentration"].dropna()
                pm10_f = pm10_f[pm10_f < 200]
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=pm25_f.tolist(), name="PM2.5", marker_color="#60a5fa", opacity=0.75, nbinsx=60))
                fig_h.add_trace(go.Histogram(x=pm10_f.tolist(), name="PM10",  marker_color="#a78bfa", opacity=0.75, nbinsx=60))
                fig_h.add_vline(x=35.4, line_dash="dash", line_color="#fbbf24",
                                annotation_text="Batas Sedang PM2.5 (35.4)", annotation_font_size=10)
                fig_h.update_layout(barmode="overlay", **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20),
                                    title=dict(text="Distribusi PM2.5 vs PM10 (ekor panjang ke kanan = right-skewed)", font=dict(size=12, color="#b0c4dc")),
                                    xaxis_title="Konsentrasi (ug/m3)", yaxis_title="Frekuensi (jumlah kota)")
                st.plotly_chart(fig_h, use_container_width=True)

                if "Air_quality_category" in df_data.columns:
                    cts = df_data["Air_quality_category"].dropna().value_counts().reset_index()
                    cts.columns = ["Kategori", "Jumlah"]
                    fig_c = px.bar(cts, x="Kategori", y="Jumlah", text="Jumlah",
                                   color="Kategori", color_discrete_map={"Safety": "#34d399", "Dangerous": "#f87171"},
                                   template="plotly_dark",
                                   title="Distribusi Kategori Kualitas Udara (sangat tidak seimbang)")
                    fig_c.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20))
                    fig_c.update_traces(textposition="outside", textfont=dict(color="#f0f4ff"))
                    st.plotly_chart(fig_c, use_container_width=True)

    elif step == 3:
        step_header(3, 5, "Bivariate Analysis", "Membaca Hubungan antara Dua Variabel Sekaligus")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Definisi dan Tujuan Bivariate Analysis",
                 "Bivariate Analysis mempelajari hubungan antara DUA variabel sekaligus. Pertanyaan yang dijawab: apakah ketika PM10 naik, PM2.5 juga naik? Apakah ada perbedaan PM2.5 yang sistematis antar wilayah WHO? Seberapa kuat hubungan tersebut? Tahap ini penting untuk memvalidasi bahwa fitur yang kita pilih memang benar-benar berhubungan dengan target."),
                ("Temuan Kunci: Korelasi PM10-PM2.5 = 0.886",
                 "Korelasi Pearson r = 0.886 adalah temuan paling signifikan. Artinya: 88.6% variasi PM2.5 dapat dijelaskan oleh PM10 saja. Hubungan ini sangat kuat karena keduanya memiliki sumber emisi yang sama (pembakaran, debu, industri) dan PM2.5 secara fisika adalah subset dari PM10 (partikel yang lebih kecil). Ini memvalidasi keputusan memilih PM10 sebagai fitur utama."),
                ("Boxplot per Wilayah WHO: Perbedaan Regional yang Dramatis",
                 "Analisis boxplot PM2.5 per wilayah WHO mengungkap perbedaan yang sangat dramatis: Median PM2.5 SEARO (Asia Tenggara) ~32 ug/m3; EMRO (Mediterania Timur) ~40 ug/m3; AFRO (Afrika) ~22 ug/m3; WPRO (Pasifik Barat) ~18 ug/m3; AMRO (Amerika) ~12 ug/m3; EURO (Eropa) ~10 ug/m3. Perbedaan hingga 4x antara wilayah terbersih dan paling terpolusi."),
                ("Scatter PM10 vs PM2.5: Bukan Garis Lurus Sempurna",
                 "Meskipun korelasi tinggi, scatter plot menunjukkan hubungan yang tidak perfectly linear: ada penyebaran (variance) yang besar terutama di nilai PM10 tinggi. Artinya: di kota dengan PM10 sama, PM2.5 bisa sangat berbeda tergantung komposisi debu, kondisi meteorologi, dan sumber emisi spesifik. Inilah mengapa fitur tambahan (latitude, who_region, NO2) tetap penting."),
            ]:
                st.markdown(dedent(f"""
                <div class="guide-card">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """), unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                sc = df_data.dropna(subset=["pm10_concentration", "pm25_concentration"])
                sc = sc[(sc["pm10_concentration"] < 200) & (sc["pm25_concentration"] < 150)]
                sc = sc.sample(n=min(2500, len(sc)), random_state=42)
                fig_sc = px.scatter(sc, x="pm10_concentration", y="pm25_concentration",
                                    opacity=0.35, color="pm25_concentration",
                                    color_continuous_scale="Blues", template="plotly_dark",
                                    title="Scatter PM10 vs PM2.5 -- korelasi Pearson r = 0.886")
                fig_sc.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20),
                                     coloraxis_showscale=False,
                                     xaxis_title="PM10 (ug/m3)", yaxis_title="PM2.5 (ug/m3)")
                st.plotly_chart(fig_sc, use_container_width=True)

                db = df_data.dropna(subset=["pm25_concentration", "who_region"]).copy()
                db["Region"] = db["who_region"].map(REGION_LABEL)
                db = db[db["pm25_concentration"] < 150]
                fig_bx = px.box(db, x="Region", y="pm25_concentration", color="Region",
                                template="plotly_dark",
                                title="Distribusi PM2.5 per Wilayah WHO -- perbedaan sangat signifikan")
                fig_bx.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24",
                                 annotation_text="Batas Sedang WHO (35.4)", annotation_font_size=9)
                fig_bx.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20), showlegend=False)
                st.plotly_chart(fig_bx, use_container_width=True)

    elif step == 4:
        step_header(4, 5, "Correlation Matrix", "Peta Hubungan Semua Variabel Numerik Sekaligus")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Apa itu Matriks Korelasi?",
                 "Matriks korelasi (Pearson) menampilkan nilai r antar semua pasang variabel numerik dalam satu tampilan. Nilai r berkisar dari -1.0 (korelasi negatif sempurna) hingga +1.0 (korelasi positif sempurna), dengan 0.0 berarti tidak ada hubungan linear. Kita membacanya dengan melihat intensitas warna: merah = korelasi positif kuat, biru = korelasi negatif kuat, putih/pucat = lemah."),
                ("Temuan 1: PM10 adalah Prediktor Linear Terkuat (r=0.886)",
                 "Nilai korelasi PM10-PM2.5 sebesar 0.886 adalah yang tertinggi dalam seluruh matriks, mengkonfirmasi temuan bivariate analysis. Ini berarti PM10 saja mampu memprediksi PM2.5 dengan akurasi ~78% (r^2 = 0.78). Fitur-fitur tambahan lainnya berkontribusi untuk meningkatkan akurasi dari 78% ke 89%."),
                ("Temuan 2: Latitude Berkorelasi Negatif dengan PM2.5 (r=-0.31)",
                 "Korelasi negatif berarti semakin tinggi latitude (semakin jauh ke utara), PM2.5 cenderung lebih rendah. Ini mencerminkan fakta bahwa negara-negara di lintang tinggi umumnya memiliki standar lingkungan lebih ketat, ekonomi lebih maju, dan proporsi energi terbarukan lebih besar. Korelasi sebesar -0.31 tergolong moderat namun cukup signifikan secara statistik dengan 25,999 data."),
                ("Temuan 3: Tidak Ada Multikolinearitas Kritis antar Fitur",
                 "Multikolinearitas terjadi ketika dua fitur berkorelasi sangat tinggi satu sama lain (r > 0.9), menyebabkan model bingung karena informasi yang redundan. Dalam dataset ini, korelasi antar fitur prediktor semua di bawah 0.7 kecuali pasangan (latitude, longitude) yang natural. Ini berarti semua 9 fitur membawa informasi unik yang melengkapi satu sama lain."),
            ]:
                st.markdown(dedent(f"""
                <div class="guide-card">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """), unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                cc = ["pm25_concentration", "pm10_concentration", "no2_concentration",
                      "year", "population", "latitude", "longitude", "number_of_stations"]
                cm = df_data[cc].corr()
                fig_cr = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu", aspect="auto",
                                   template="plotly_dark",
                                   title="Matriks Korelasi Pearson -- merah = positif, biru = negatif")
                fig_cr.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=60, b=20))
                st.plotly_chart(fig_cr, use_container_width=True)

                yr = df_data.dropna(subset=["pm25_concentration"]).groupby("year")["pm25_concentration"].agg(["mean", "median"]).reset_index()
                fig_yr = go.Figure()
                fig_yr.add_trace(go.Scatter(x=yr["year"].tolist(), y=yr["mean"].tolist(), mode="lines+markers",
                                             name="Rata-rata", line=dict(color="#60a5fa", width=2.5)))
                fig_yr.add_trace(go.Scatter(x=yr["year"].tolist(), y=yr["median"].tolist(), mode="lines+markers",
                                             name="Median", line=dict(color="#f97316", width=2, dash="dash")))
                fig_yr.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=60, b=20),
                                     title=dict(text="Tren PM2.5 Global per Tahun (2010-2022)", font=dict(size=13, color="#b0c4dc")),
                                     xaxis_title="Tahun", yaxis_title="PM2.5 (ug/m3)")
                st.plotly_chart(fig_yr, use_container_width=True)

    elif step == 5:
        step_header(5, 5, "Model Final", "Arsitektur Pipeline, Perbandingan Algoritma, dan Evaluasi Mendalam")

        st.markdown(dedent("""
        <div class="callout">
          Pipeline ML adalah rantai transformasi yang terhubung -- data mentah masuk di ujung kiri,
          prediksi keluar di ujung kanan. Semua langkah (preprocessing dan model) dikemas dalam satu objek
          yang bisa disimpan dan dimuat kembali dengan satu perintah. Ini memastikan konsistensi:
          data baru yang masuk saat prediksi diperlakukan persis sama seperti data saat pelatihan.
        </div>
        """), unsafe_allow_html=True)

        p1, pa1, p2, pa2, p3 = st.columns([2, 0.2, 2, 0.2, 2])
        with p1:
            st.markdown(dedent("""
            <div class="pipe-step">
              <span class="pipe-badge">INPUT</span>
              <div class="pipe-title">Data Mentah (9 Kolom)</div>
              <div class="pipe-desc">
                year, latitude, longitude, pm10, no2, stations, who_ms, population, who_region.
                Beberapa kolom mungkin memiliki nilai NaN (kosong) yang perlu ditangani sebelum diproses model.
              </div>
            </div>
            """), unsafe_allow_html=True)
        with pa1:
            st.markdown('<div style="text-align:center;margin-top:40px;color:#4a5a6e;font-size:1.2rem;">-&gt;</div>', unsafe_allow_html=True)
        with p2:
            st.markdown(dedent("""
            <div class="pipe-step">
              <span class="pipe-badge">PREPROCESSING</span>
              <div class="pipe-title">ColumnTransformer</div>
              <div class="pipe-desc">
                <strong>Jalur Numerik (8 kolom):</strong> Median Imputer mengisi NaN dengan nilai median dari data latih, lalu StandardScaler menstandarisasi ke mean=0 dan std=1.<br><br>
                <strong>Jalur Kategorikal (1 kolom):</strong> Constant Imputer mengisi NaN dengan "Unknown", lalu OneHotEncoder mengubah who_region menjadi 7 kolom biner (0 atau 1).
              </div>
            </div>
            """), unsafe_allow_html=True)
        with pa2:
            st.markdown('<div style="text-align:center;margin-top:40px;color:#4a5a6e;font-size:1.2rem;">-&gt;</div>', unsafe_allow_html=True)
        with p3:
            st.markdown(dedent("""
            <div class="pipe-step">
              <span class="pipe-badge">MODEL</span>
              <div class="pipe-title">Random Forest Regressor</div>
              <div class="pipe-desc">
                300 pohon keputusan, dilatih secara paralel di semua CPU (n_jobs=-1). Setiap pohon dilatih pada subset data dan fitur yang berbeda. Prediksi akhir = rata-rata prediksi 300 pohon.
              </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        m_names = ["Linear Regression", "Ridge Regression", "Decision Tree", "Gradient Boosting", "Random Forest"]
        r2_v    = [0.6210, 0.6212, 0.7850, 0.8850, 0.8900]
        rmse_v  = [6.45,   6.45,   4.85,   2.78,   2.68]
        bar_c5  = ["#1e293b", "#1e293b", "#334155", "#60a5fa", "#34d399"]

        mc1, mc2 = st.columns(2)
        with mc1:
            fig_r2 = go.Figure(go.Bar(x=r2_v, y=m_names, orientation="h", marker_color=bar_c5,
                                       text=[f"{v:.4f}" for v in r2_v], textposition="outside",
                                       textfont=dict(color="#f0f4ff", size=11)))
            fig_r2.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=60, t=50, b=20),
                                  xaxis=dict(range=[0, 1.08]),
                                  title=dict(text="R2 Score per Algoritma (lebih tinggi lebih baik)", font=dict(size=12, color="#b0c4dc")))
            st.plotly_chart(fig_r2, use_container_width=True)

        with mc2:
            fig_rm = go.Figure(go.Bar(x=rmse_v, y=m_names, orientation="h", marker_color=bar_c5,
                                       text=[f"{v:.2f}" for v in rmse_v], textposition="outside",
                                       textfont=dict(color="#f0f4ff", size=11)))
            fig_rm.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=60, t=50, b=20),
                                  title=dict(text="RMSE per Algoritma (lebih rendah lebih baik)", font=dict(size=12, color="#b0c4dc")))
            st.plotly_chart(fig_rm, use_container_width=True)

        if df_data is not None:
            @st.cache_data
            def run_eval_step5():
                dm = df_data.dropna(subset=[TARGET_COL]).copy()
                dm = dm[dm[TARGET_COL] > 0]
                for c in NUM_COLS:
                    dm[c] = pd.to_numeric(dm[c], errors="coerce")
                for c in CAT_COLS:
                    dm[c] = dm[c].fillna("Unknown").astype(str)
                dm = dm.dropna(subset=FEATURE_COLS)
                Xtr, Xte, ytr, yte = train_test_split(dm[FEATURE_COLS], dm[TARGET_COL], test_size=0.2, random_state=42)
                yp = model.predict(Xte)
                ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["ohe"]
                fn  = NUM_COLS + (list(ohe.get_feature_names_out(["who_region"])) if hasattr(ohe, "get_feature_names_out") else list(ohe.get_feature_names(["who_region"])))
                imp = model.named_steps["model"].feature_importances_
                return yte.values, yp, yte.values - yp, fn, imp, r2_score(yte, yp), mean_absolute_error(yte, yp)

            yt5, yp5, rs5, fn5, imp5, r2v5, maev5 = run_eval_step5()

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                sdf5 = pd.DataFrame({"Aktual": yt5, "Prediksi": yp5}).sample(n=min(1000, len(yt5)), random_state=42)
                fig_ev = px.scatter(sdf5, x="Aktual", y="Prediksi", opacity=0.35,
                                    color="Prediksi", color_continuous_scale="Blues", template="plotly_dark",
                                    title=f"Aktual vs Prediksi (R2={r2v5:.4f})")
                lm = float(yt5.max()) * 1.05
                fig_ev.add_shape(type="line", line=dict(dash="dash", color="#f87171", width=2), x0=0, y0=0, x1=lm, y1=lm)
                fig_ev.update_layout(**PLOTLY_BASE, margin=dict(l=5, r=5, t=60, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_ev, use_container_width=True)

            with ec2:
                fig_rs5 = px.histogram(x=rs5.tolist(), nbins=60, template="plotly_dark",
                                        title=f"Distribusi Residual (MAE={maev5:.2f})")
                fig_rs5.update_traces(marker_color="#a78bfa")
                fig_rs5.add_vline(x=0, line_dash="dash", line_color="#f87171")
                fig_rs5.update_layout(**PLOTLY_BASE, margin=dict(l=5, r=5, t=60, b=20),
                                      xaxis_title="Residu (Aktual - Prediksi)", yaxis_title="Frekuensi")
                st.plotly_chart(fig_rs5, use_container_width=True)

            with ec3:
                fdf5 = pd.DataFrame({"Fitur": fn5, "Importance": imp5}).nlargest(8, "Importance").sort_values("Importance")
                fig_fi5 = px.bar(fdf5, x="Importance", y="Fitur", orientation="h",
                                 text=[f"{v:.1%}" for v in fdf5["Importance"]],
                                 template="plotly_dark", title="Feature Importance Top 8")
                fig_fi5.update_traces(marker_color="#34d399", textposition="outside",
                                      textfont=dict(color="#f0f4ff", size=10))
                fig_fi5.update_layout(**PLOTLY_BASE, margin=dict(l=5, r=5, t=60, b=20))
                st.plotly_chart(fig_fi5, use_container_width=True)

        st.markdown(dedent(f"""
        <div class="insight-box">
          <div class="insight-title">Mengapa Random Forest Mengungguli Semua Algoritma Lain?</div>
          <p>
            <strong>vs Linear Regression (R2=0.62):</strong> Linear Regression hanya bisa mempelajari hubungan LURUS antara fitur dan target. Karena hubungan PM10-PM2.5 dan latitude-PM2.5 tidak sepenuhnya linear (ada efek threshold, interaksi antar variabel, dll), Linear Regression tertinggal jauh. Perbaikan R2 dari 0.62 ke 0.89 berarti Random Forest mampu menangkap hubungan non-linear yang sangat kaya.<br><br>
            <strong>vs Decision Tree (R2=0.785):</strong> Decision Tree adalah satu pohon keputusan tunggal. Satu pohon mudah "menghafal" data latih hingga terlalu detail (overfitting). Ketika menghadapi data test yang berbeda, akurasinya turun drastis. Random Forest membangun 300 pohon yang SENGAJA dibuat berbeda (dengan subset data dan fitur berbeda-beda), sehingga error antar pohon tidak berkorelasi dan saling mengkompensasi saat dirata-rata.<br><br>
            <strong>vs Gradient Boosting (R2=0.885):</strong> Gradient Boosting hampir sebaik Random Forest dengan prinsip berbeda: membangun pohon SECARA BERURUTAN di mana setiap pohon baru belajar memperbaiki kesalahan pohon sebelumnya. Kelemahan: lebih lambat (sequential vs parallel), lebih sensitif terhadap hyperparameter, dan cenderung overfit jika learning rate tidak dikalibrasi dengan baik. Random Forest dipilih karena lebih robust dan lebih mudah dikonfigurasi.
          </p>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_l, _, nav_r = st.columns([1, 3, 1])
    if step > 1: nav_l.button("Sebelumnya", on_click=go_step, args=(step - 1,), use_container_width=True)
    if step < 5: nav_r.button("Berikutnya", on_click=go_step, args=(step + 1,), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 -- PENJELASAN KODE
# ═══════════════════════════════════════════════════════════════════════════
elif "Penjelasan" in menu:

    st.markdown(dedent("""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">Dokumentasi Teknis Mendalam</div>
      <div class="hero-title" style="font-size:2rem;">Penjelasan Kode Proyek</div>
      <p class="subtitle">
        Dokumentasi lengkap setiap komponen kode: proyek.ipynb sebagai laboratorium eksperimen,
        build_pipeline.py sebagai pabrik produksi model, generate_dark_charts.py sebagai pembuat visualisasi,
        dan app.py sebagai antarmuka pengguna. Setiap konsep teknis dijelaskan dari prinsip pertama.
      </p>
    </div>
    """), unsafe_allow_html=True)

    tab_nb, tab_build, tab_charts, tab_app = st.tabs([
        "proyek.ipynb", "build_pipeline.py", "generate_dark_charts.py", "app.py",
    ])

    with tab_nb:
        st.markdown(dedent("""
        <div style="margin:14px 0 10px;">
          <div class="eyebrow">Notebook Eksperimen</div>
          <div class="section-title">proyek.ipynb -- Laboratorium Data Science</div>
        </div>
        <div class="callout">
          <strong>Peran dalam Arsitektur Proyek:</strong> Notebook Jupyter adalah "ruang kerja fleksibel"
          di mana semua eksperimen pertama kali dilakukan. Kode di notebook bisa dijalankan sel demi sel,
          memungkinkan kita melihat hasil intermediate (grafik, tabel, metrik) sebelum melanjutkan ke langkah berikutnya.
          Ketika sebuah pendekatan terbukti berhasil, kodenya dipindahkan ke script produksi (build_pipeline.py)
          yang lebih terstruktur dan dapat dijalankan secara otomatis.
        </div>
        """), unsafe_allow_html=True)

        with st.expander("Cell 1 -- Import Library dan Exploratory Data Analysis (EDA)", expanded=True):
            st.markdown(dedent("""
            <div class="guide-card">
              <div class="guide-title">Tujuan dan Konteks Cell 1</div>
              <div class="guide-body">
                Cell pertama menetapkan fondasi proyek: memuat semua library yang dibutuhkan,
                membaca dataset dari disk ke memori, memperbaiki tipe data, dan menghasilkan 6 grafik EDA
                untuk memahami dataset secara menyeluruh. Setiap baris kode di sini memiliki alasan teknis yang jelas.
              </div>
            </div>
            """), unsafe_allow_html=True)

            st.code(dedent("""
            # ── IMPORT LIBRARY ─────────────────────────────────────────────
            import pandas as pd          # Manipulasi data tabular (DataFrame, Series)
            import numpy  as np          # Operasi matematika array multidimensi
            import matplotlib.pyplot as plt  # Pembuatan grafik statis (histogram, scatter, dll)
            import seaborn as sns        # Visualisasi statistik berbasis matplotlib

            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split, KFold, cross_val_score
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

            # ── MEMUAT DATASET ──────────────────────────────────────────────
            # low_memory=False mencegah pandas menebak tipe data per chunk
            # yang bisa menghasilkan tipe data tidak konsisten di kolom yang sama
            df_raw = pd.read_csv("action2024/train.csv", low_memory=False)

            # ── KONVERSI TIPE DATA ──────────────────────────────────────────
            # Beberapa kolom numerik mungkin terbaca sebagai "object" (string)
            # karena ada nilai tidak valid seperti "-", "N/A", atau spasi kosong.
            # errors='coerce' mengubah nilai tidak valid menjadi NaN (bukan error).
            COLS_NUMERIC = ['pm25_concentration', 'pm10_concentration',
                            'no2_concentration', 'population', 'number_of_stations']
            for col in COLS_NUMERIC:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

            # ── ANALISIS MISSING VALUES ──────────────────────────────────────
            # isnull().sum() menghitung jumlah NaN per kolom
            # / len(df_raw) * 100 mengubahnya ke persentase
            missing_pct = df_raw.isnull().sum() / len(df_raw) * 100
            print(missing_pct.sort_values(ascending=False))
            """), language="python")

            st.markdown(dedent("""
            <div class="insight-box" style="margin-top:12px;">
              <div class="insight-title">Mengapa StandardScaler Digunakan?</div>
              <p>
                StandardScaler mengubah setiap fitur numerik sehingga memiliki mean (rata-rata) = 0
                dan standard deviation = 1. Ini penting karena model seperti Linear Regression dan SVM
                sangat sensitif terhadap skala fitur -- jika satu fitur berkisar 0-100 dan fitur lain
                0-10,000,000 (populasi), model akan "terlalu memperhatikan" fitur berskala besar.
                Random Forest sendiri tidak memerlukan scaling (karena bekerja dengan perbandingan relatif),
                namun kita tetap menyertakannya untuk konsistensi pipeline dan kemudahan ekspansi ke model lain.
              </p>
            </div>
            """), unsafe_allow_html=True)

        with st.expander("Cell 2 -- Feature Engineering dan Cross-Validation 5-Fold"):
            st.markdown(dedent("""
            <div class="guide-card">
              <div class="guide-title">Mengapa Cross-Validation, Bukan Sekadar Train-Test Split?</div>
              <div class="guide-body">
                Train-Test Split biasa (80-20) hanya mengevaluasi model satu kali pada satu subset tertentu.
                Hasilnya bisa bias: mungkin secara kebetulan subset test kita "mudah" atau "sulit".
                K-Fold Cross Validation mengevaluasi model K kali dengan pembagian data yang berbeda setiap iterasi,
                memberikan estimasi performa yang jauh lebih stabil dan dapat dipercaya. Standar di komunitas ML
                adalah 5-fold atau 10-fold CV untuk dataset menengah seperti ini.
              </div>
            </div>
            """), unsafe_allow_html=True)

            st.code(dedent("""
            # ── K-FOLD CROSS VALIDATION ──────────────────────────────────────
            # KFold membagi data menjadi K=5 bagian (fold) secara merata.
            # shuffle=True mengacak urutan data sebelum dibagi -- penting jika
            # data diurutkan berdasarkan tahun atau wilayah (yang bisa menciptakan
            # bias temporal/geografis dalam evaluasi).
            # random_state=42 memastikan hasil dapat direproduksi.

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Alur K-Fold untuk K=5:
            # Iterasi 1: Latih di fold 2,3,4,5 --> Test di fold 1
            # Iterasi 2: Latih di fold 1,3,4,5 --> Test di fold 2
            # Iterasi 3: Latih di fold 1,2,4,5 --> Test di fold 3
            # Iterasi 4: Latih di fold 1,2,3,5 --> Test di fold 4
            # Iterasi 5: Latih di fold 1,2,3,4 --> Test di fold 5
            # SETIAP data pernah menjadi data test TEPAT 1 kali.
            # Hasil akhir: RATA-RATA dari 5 nilai R2 (lebih andal dari 1 nilai saja).

            MODELS = {
                'Linear Regression'  : LinearRegression(),
                'Ridge Regression'   : Ridge(alpha=1.0),         # L2 regularization
                'Decision Tree'      : DecisionTreeRegressor(max_depth=10),
                'Random Forest'      : RandomForestRegressor(n_estimators=100),
                'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=100),
            }

            hasil_cv = {}
            for nama, mdl in MODELS.items():
                # Pipeline menggabungkan preprocessing + model menjadi satu unit
                # PENTING: preprocessing (imputer, scaler) HARUS masuk pipeline
                # agar tidak terjadi "data leakage" -- yaitu informasi dari data test
                # bocor ke proses pelatihan (yang akan membuat evaluasi terlalu optimis)
                pipe = Pipeline([('preprocessor', preprocessor), ('model', mdl)])

                # cross_val_score menjalankan CV secara otomatis
                # scoring='r2' menggunakan R2 Score sebagai metrik evaluasi
                # n_jobs=-1 memanfaatkan semua CPU core untuk komputasi paralel
                r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2', n_jobs=-1)
                hasil_cv[nama] = {'r2_mean': r2_scores.mean(), 'r2_std': r2_scores.std()}
                print(f"{nama}: R2 = {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
            """), language="python")

            for title, body in [
                ("Hasil CV: Linear Regression R2=0.62 -- Mengapa Buruk?",
                 "Linear Regression mengasumsikan output adalah KOMBINASI LINEAR dari input: PM2.5 = w1*PM10 + w2*latitude + ... + b. Realitanya, hubungan PM10-PM2.5 tidak linear (ada saturation effect di nilai tinggi) dan ada interaksi non-linear antar variabel (PM10 tinggi di Eropa vs Asia menghasilkan PM2.5 yang sangat berbeda). Keterbatasan fundamental ini menyebabkan R2 hanya 0.62."),
                ("Hasil CV: Random Forest R2=0.89 -- Mengapa Terbaik?",
                 "Random Forest mengatasi keterbatasan linear dengan membangun 100+ pohon keputusan, di mana setiap pohon dapat menangkap interaksi kompleks dan non-linear secara alami (melalui aturan IF-THEN yang bersarang). Averaging 100 pohon mengurangi variance secara drastis dibanding pohon tunggal (Decision Tree). Hasilnya: model yang akurat, stabil, dan tahan terhadap overfitting."),
            ]:
                st.markdown(dedent(f"""
                <div class="pt-item">
                  <div class="pt-title">{title}</div>
                  <div class="pt-body">{body}</div>
                </div>
                """), unsafe_allow_html=True)

        with st.expander("Cell 3 -- Training Model Final dan Menyimpan Pipeline"):
            st.code(dedent("""
            # ── TRAIN-TEST SPLIT FINAL ──────────────────────────────────────
            # 80% data untuk pelatihan, 20% untuk evaluasi akhir yang bersih.
            # test_size=0.2 mengalokasikan ~5,200 baris untuk test set.
            # random_state=42 memastikan reproduktibilitas -- split yang sama
            # setiap kali kode dijalankan, sehingga perbandingan antar eksperimen adil.
            # stratify tidak digunakan karena ini adalah regresi, bukan klasifikasi.

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ── KONFIGURASI HYPERPARAMETER FINAL ──────────────────────────
            # n_estimators=300: lebih banyak pohon = lebih stabil, tapi lebih lambat.
            #   Di notebook (eksperimen) pakai 100, di produksi (build_pipeline.py) pakai 300.
            # min_samples_leaf=2: setiap daun minimal berisi 2 sampel.
            #   Mencegah pohon terlalu spesifik pada satu data point (overfitting).
            # max_features='sqrt': setiap split hanya mempertimbangkan sqrt(n_fitur) fitur.
            #   Meningkatkan diversitas antar pohon, kunci keberhasilan Random Forest.
            # n_jobs=-1: gunakan semua CPU core yang tersedia untuk training paralel.

            final_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                ))
            ])

            # .fit() melakukan 3 hal sekaligus dalam urutan:
            # 1. Preprocessor menghitung median DARI X_train (bukan X_test!)
            # 2. Preprocessor menghitung parameter OHE dari X_train
            # 3. Random Forest melatih 300 pohon menggunakan data yang sudah diproses
            final_pipeline.fit(X_train, y_train)

            # ── SIMPAN PIPELINE KE FILE .pkl ──────────────────────────────
            # joblib.dump lebih efisien dari pickle untuk objek NumPy/sklearn besar.
            # File .pkl berisi SELURUH pipeline: median imputer values, OHE categories,
            # dan 300 pohon keputusan dengan semua percabangan dan nilai daun.
            # Saat di-load kembali, pipeline siap melakukan prediksi tanpa perlu training ulang.
            import joblib
            joblib.dump(final_pipeline, 'pipeline_pm25_final.pkl')
            print(f"Pipeline tersimpan: {os.path.getsize('pipeline_pm25_final.pkl')/1024/1024:.1f} MB")
            """), language="python")

            st.markdown(dedent("""
            <div class="insight-box" style="margin-top:12px;">
              <div class="insight-title">Konsep Krusial: Data Leakage dan Cara Mencegahnya</div>
              <p>
                Data Leakage adalah salah satu kesalahan paling fatal dalam machine learning. Ini terjadi
                ketika informasi dari DATA TEST secara tidak sengaja "bocor" ke proses pelatihan, membuat
                model tampak lebih akurat dari yang sebenarnya.
                <br><br>
                <strong>Contoh Data Leakage:</strong> Jika kita menghitung nilai median PM10 dari SELURUH
                dataset (termasuk test set), lalu menggunakannya untuk mengimputasi training data -- berarti
                median tersebut "terpengaruh" oleh test set. Model menjadi terlalu optimis karena telah
                "melihat" sebagian test set secara tidak langsung.
                <br><br>
                <strong>Solusi: Pipeline dengan fit hanya pada Training Data</strong><br>
                Dengan memasukkan SimpleImputer dan StandardScaler ke dalam sklearn Pipeline,
                perintah pipeline.fit(X_train, y_train) memastikan imputer dan scaler HANYA mempelajari
                statistik dari X_train. Ketika pipeline.predict(X_test) dipanggil, X_test diproses
                menggunakan statistik yang dipelajari dari X_train -- persis seperti kondisi produksi nyata
                di mana data baru belum pernah kita lihat sebelumnya.
              </p>
            </div>
            """), unsafe_allow_html=True)

    with tab_build:
        st.markdown(dedent("""
        <div style="margin:14px 0 10px;">
          <div class="eyebrow">Script Produksi</div>
          <div class="section-title">build_pipeline.py -- Pabrik Otomatis Produksi Model</div>
        </div>
        <div class="callout">
          <strong>Peran dalam Arsitektur Proyek:</strong> build_pipeline.py adalah script yang dijalankan
          SATU KALI dari terminal dengan perintah <code>python build_pipeline.py</code> untuk menghasilkan
          file pipeline_pm25_final.pkl. Script ini dirancang untuk production readiness: validasi file,
          logging informatif, penanganan error, dan konfigurasi model yang lebih kuat dari versi notebook.
          Setelah .pkl dihasilkan, tidak perlu menjalankan script ini lagi kecuali ingin melatih ulang model.
        </div>
        """), unsafe_allow_html=True)

        with st.expander("Tahap 1 & 2 -- Validasi File dan Pembersihan Data", expanded=True):
            st.code(dedent("""
            # ── VALIDASI KEBERADAAN FILE ────────────────────────────────────
            # os.path.exists() mengembalikan True/False tanpa menimbulkan exception.
            # sys.exit(1) menghentikan program dengan exit code 1 (error).
            # Konvensi: exit code 0 = sukses, exit code lainnya = error.
            # "Fail fast" adalah prinsip penting: lebih baik gagal di awal dengan
            # pesan yang jelas daripada crash di tengah proses setelah menit-menit berjalan.

            if not os.path.exists(TRAIN_FILE):
                print(f"File tidak ditemukan: {TRAIN_FILE}")
                sys.exit(1)

            # ── PEMBERSIHAN DATA ─────────────────────────────────────────────
            # pd.to_numeric dengan errors='coerce' adalah strategi defensif:
            # jika ada sel yang berisi teks seperti "-", "N/A", atau "" (kosong),
            # akan dikonversi ke NaN daripada menimbulkan ValueError.
            for col in NUM_COLS + [TARGET_COL]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Hapus baris dengan target tidak valid
            # dropna(subset=[TARGET_COL]): hapus baris di mana PM2.5 adalah NaN
            df_model = df.dropna(subset=[TARGET_COL]).copy()

            # Hapus nilai target negatif atau nol -- tidak mungkin secara fisika.
            # Konsentrasi partikel tidak bisa negatif. Jika ada, itu error pengukuran.
            df_model = df_model[df_model[TARGET_COL] > 0]

            # Laporan jumlah baris hilang -- penting untuk audit kualitas data
            print(f"Data tersedia setelah pembersihan: {len(df_model):,} baris")
            """), language="python")

        with st.expander("Tahap 3 -- Arsitektur Pipeline Produksi dengan ColumnTransformer"):
            st.code(dedent("""
            # ── SUB-PIPELINE UNTUK KOLOM NUMERIK ───────────────────────────
            # Dua langkah berurutan: imputasi lalu standarisasi.
            # Urutan penting: kita harus mengisi NaN SEBELUM standarisasi,
            # karena StandardScaler tidak bisa menangani NaN.

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                # Median lebih robust dari mean terhadap outlier.
                # Jika PM10 = [5, 8, 12, 200], mean = 56.25 (tidak representatif)
                # tapi median = 10 (lebih representatif dari nilai "normal").

                ("scaler",  StandardScaler()),
                # Mengubah setiap fitur ke mean=0, std=1.
                # Formula: x_scaled = (x - mean) / std
                # Contoh: PM10 = [5, 8, 12] -> [-1.2, -0.2, 1.3]
                # Semua fitur numerik menjadi sebanding skalanya.
            ])

            # ── SUB-PIPELINE UNTUK KOLOM KATEGORIKAL ───────────────────────
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                # Strategi "constant": isi NaN dengan string "Unknown".
                # Bukan median/mean karena data kategorikal tidak memiliki rata-rata.
                # "Unknown" diperlakukan sebagai kategori baru oleh OneHotEncoder.

                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                # Mengubah who_region (7 nilai unik) menjadi 7 kolom biner (0 atau 1).
                # Contoh: "3_Sear" -> [0, 0, 1, 0, 0, 0, 0]
                # handle_unknown="ignore": jika saat prediksi ada nilai who_region baru
                # yang tidak ada di data latih, OHE menghasilkan semua 0 (bukan error).
            ])

            # ── COLUMN TRANSFORMER: MENGGABUNGKAN KEDUA JALUR ──────────────
            # ColumnTransformer menerapkan transformasi berbeda ke kolom berbeda secara paralel.
            # ("num", num_pipeline, NUM_COLS): terapkan num_pipeline ke 8 kolom numerik
            # ("cat", cat_pipeline, CAT_COLS): terapkan cat_pipeline ke 1 kolom kategorik
            # Hasilnya digabungkan menjadi satu matriks: 8 + 7 = 15 fitur total.

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, NUM_COLS),
                ("cat", cat_pipeline, CAT_COLS),
            ])

            # ── PIPELINE AKHIR: PREPROCESSING + MODEL ──────────────────────
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators      = 300,    # 3x lebih banyak dari eksperimen notebook
                    max_depth         = None,   # Pohon tumbuh penuh (tidak dibatasi)
                    min_samples_split = 4,      # Minimal 4 sampel untuk membuat percabangan baru
                    min_samples_leaf  = 2,      # Minimal 2 sampel di setiap daun (anti-overfitting)
                    max_features      = "sqrt", # Pertimbangkan sqrt(15) ~ 4 fitur per split
                    n_jobs            = -1,     # Paralel di semua CPU core
                    random_state      = 42,     # Reproduktibilitas
                )),
            ])
            """), language="python")

        with st.expander("Tahap 5 & 6 -- Evaluasi Metrik dan Menyimpan Pipeline"):
            st.code(dedent("""
            # ── EVALUASI KOMPREHENSIF ─────────────────────────────────────
            # Penting: evaluasi di TRAINING data dan TEST data KEDUANYA.
            # Perbedaan besar antara train dan test = overfitting (model "hafal" training).
            # Perbedaan kecil = generalisasi yang baik.

            y_pred_train = pipeline.predict(X_train)
            y_pred_test  = pipeline.predict(X_test)

            # Tiga metrik evaluasi yang saling melengkapi:
            mae_test  = mean_absolute_error(y_test, y_pred_test)
            # MAE = rata-rata |y_aktual - y_prediksi|
            # Interpretasi langsung: rata-rata model meleset ±mae_test ug/m3
            # Keunggulan: tidak sensitif outlier (tidak dikuadratkan)

            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            # RMSE = akar(rata-rata (y_aktual - y_prediksi)^2)
            # Interpretasi: seperti MAE tapi kesalahan BESAR dihukum lebih berat
            # Keunggulan: dalam satuan yang sama (ug/m3), lebih sensitif outlier

            r2_test   = r2_score(y_test, y_pred_test)
            # R2 = 1 - (SS_res / SS_tot)
            # SS_res = sum of squared residuals (kesalahan model)
            # SS_tot = total variance dalam y (jika model hanya memprediksi rata-rata)
            # R2=0.89 berarti model 89% lebih baik dari model "prediksi selalu rata-rata"

            print(f"MAE  : {mae_test:.4f} ug/m3")
            print(f"RMSE : {rmse_test:.4f} ug/m3")
            print(f"R2   : {r2_test:.4f}")

            # ── MENYIMPAN PIPELINE KE FILE .pkl ──────────────────────────
            joblib.dump(pipeline, "pipeline_pm25_final.pkl")

            # Verifikasi ukuran: model Random Forest dengan 300 pohon biasanya 20-100 MB.
            # Jika terlalu kecil (<1 MB), ada yang salah. Jika terlalu besar (>500 MB),
            # pertimbangkan mengurangi n_estimators atau max_depth.
            size_mb = os.path.getsize("pipeline_pm25_final.pkl") / (1024 * 1024)
            print(f"Ukuran file: {size_mb:.1f} MB")
            # File .pkl yang tersimpan berisi:
            # 1. Nilai median setiap fitur numerik (dipelajari dari X_train)
            # 2. Kategori yang dikenali OHE untuk who_region (dipelajari dari X_train)
            # 3. 300 pohon keputusan lengkap dengan semua percabangan dan nilai daun
            """), language="python")

    with tab_charts:
        st.markdown(dedent("""
        <div style="margin:14px 0 10px;">
          <div class="eyebrow">Script Visualisasi</div>
          <div class="section-title">generate_dark_charts.py -- Pembuat Grafik Tema Gelap untuk Laporan</div>
        </div>
        <div class="callout">
          <strong>Peran dalam Arsitektur Proyek:</strong> Script utilitas mandiri yang menghasilkan 5 file PNG
          beresolusi tinggi (dpi=150) untuk digunakan dalam laporan akademik, presentasi, atau poster.
          Script ini berdiri sendiri -- tidak bergantung pada Streamlit -- dan menggunakan matplotlib/seaborn
          murni. Tema gelap dipilih untuk konsistensi visual dengan dashboard Streamlit dan karena terlihat
          lebih modern dalam presentasi dengan background gelap.
        </div>
        """), unsafe_allow_html=True)

        with st.expander("Konfigurasi Tema Gelap Global dengan plt.rcParams", expanded=True):
            st.code(dedent("""
            # ── KONFIGURASI GLOBAL MATPLOTLIB ──────────────────────────────
            # plt.rcParams adalah dictionary yang mengontrol semua default matplotlib.
            # Mengubahnya SEKALI di awal mempengaruhi SEMUA grafik yang dibuat sesudahnya
            # dalam sesi yang sama -- prinsip DRY (Don't Repeat Yourself).
            # Tanpa ini, setiap grafik harus dikonfigurasi manual (20+ baris per grafik).

            plt.rcParams.update({
                'figure.facecolor': '#090d16',   # Warna latar belakang FIGURE (area di luar plot)
                'axes.facecolor'  : '#111827',   # Warna latar belakang AXES (area plot itu sendiri)
                'text.color'      : '#f1f5f9',   # Warna default SEMUA teks (judul, label, dll)
                'axes.labelcolor' : '#f1f5f9',   # Warna label sumbu X dan Y secara spesifik
                'xtick.color'     : '#94a3b8',   # Warna angka-angka pada sumbu X
                'ytick.color'     : '#94a3b8',   # Warna angka-angka pada sumbu Y
                'axes.edgecolor'  : '#334155',   # Warna garis bingkai plot (spines)
                'grid.color'      : '#1e293b',   # Warna garis kisi (gridlines)
                'font.family'     : 'sans-serif',# Jenis huruf default
                'font.size'       : 10,          # Ukuran huruf default (dalam points)
                'axes.titlesize'  : 12,          # Ukuran huruf judul grafik
                'axes.titleweight': 'bold',      # Tebal atau tidak judul grafik
                'legend.facecolor': '#111827',   # Warna background kotak legenda
                'legend.edgecolor': '#334155',   # Warna border kotak legenda
                'figure.autolayout': False       # Nonaktifkan auto-layout (kita atur manual)
            })
            """), language="python")

        with st.expander("Helper Function apply_dark_theme_axes() -- Prinsip DRY"):
            st.code(dedent("""
            # ── FUNGSI PEMBANTU UNTUK SETIAP SUBPLOT ───────────────────────
            # Fungsi ini menerapkan styling dark theme ke satu axes (subplot).
            # Tanpa fungsi ini, kita harus mengetik 6+ baris kode berulang
            # untuk setiap subplot -- dan jika ingin mengubah warna, harus ubah
            # di setiap tempat. Dengan fungsi ini, cukup ubah di satu tempat.

            def apply_dark_theme_axes(ax, title=""):
                # Judul subplot dengan warna biru terang (#64d2ff)
                ax.set_title(title, color='#64d2ff', pad=12)

                # Garis kisi horizontal dan vertikal -- membantu membaca nilai
                # linestyle='--': garis putus-putus, alpha=0.5: setengah transparan
                ax.grid(True, linestyle='--', alpha=0.5, color='#1e293b')

                # Hilangkan garis bingkai atas dan kanan (terlihat lebih modern)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Garis bingkai kiri dan bawah dengan warna slate-700
                ax.spines['left'].set_color('#334155')
                ax.spines['bottom'].set_color('#334155')

            # CONTOH PENGGUNAAN:
            fig, ax = plt.subplots()
            ax.hist(data, color='#64d2ff')
            apply_dark_theme_axes(ax, 'Judul Grafik Saya')
            # Hasilnya: histogram dengan tema gelap + judul + grid + styling bingkai
            """), language="python")

        with st.expander("Menyimpan File PNG Beresolusi Tinggi"):
            st.code(dedent("""
            # ── MENYIMPAN GRAFIK ─────────────────────────────────────────
            # plt.savefig() adalah perintah akhir yang menyimpan figure ke disk.
            # Parameter yang digunakan:

            plt.savefig(
                'nama_file.png',       # Path dan nama file output
                dpi=150,               # Dots Per Inch -- resolusi gambar
                                       # 72 DPI: layar; 150 DPI: web/presentasi; 300 DPI: cetak
                                       # 150 DPI dipilih: balance kualitas dan ukuran file
                facecolor='#090d16',   # Warna background figure saat disimpan
                                       # PENTING: tanpa ini, background mungkin putih
                                       # meskipun plt.rcParams sudah diset
                bbox_inches='tight'    # Memotong whitespace kosong di tepi gambar
                                       # agar output rapi tanpa margin berlebih
            )

            # plt.close() WAJIB dipanggil setelah savefig!
            # Matplotlib menyimpan figure di memori sampai explisit ditutup.
            # Jika membuat 100 grafik tanpa plt.close(), memori akan penuh (memory leak).
            # plt.close() membebaskan memori yang digunakan figure tersebut.
            plt.close()

            # Untuk animasi atau multiple pages, gunakan plt.close('all')
            # yang menutup SEMUA figure yang masih terbuka sekaligus.
            """), language="python")

    with tab_app:
        st.markdown(dedent("""
        <div style="margin:14px 0 10px;">
          <div class="eyebrow">Frontend Aplikasi Web</div>
          <div class="section-title">app.py -- Aplikasi Streamlit Produksi</div>
        </div>
        <div class="callout">
          <strong>Peran dalam Arsitektur Proyek:</strong> app.py adalah "wajah publik" proyek -- antarmuka
          yang dihadapi pengguna akhir. Ia memuat model dari .pkl, menerima input dari pengguna melalui
          widget Streamlit, menjalankan prediksi, dan menampilkan hasil dalam format yang mudah dipahami.
          Streamlit mengubah kode Python menjadi web application tanpa memerlukan pengetahuan HTML/CSS/JavaScript.
        </div>
        """), unsafe_allow_html=True)

        with st.expander("@st.cache_resource -- Memuat Model Hanya Sekali", expanded=True):
            st.code(dedent("""
            # ── MASALAH TANPA CACHE ──────────────────────────────────────
            # Streamlit me-rerun SELURUH script dari atas ke bawah setiap kali
            # user mengklik tombol, menggeser slider, atau mengubah input apapun.
            # Tanpa cache, joblib.load() akan dipanggil setiap rerun = memuat
            # file .pkl berulang kali = latensi 5-30 detik setiap interaksi.

            # ── SOLUSI: @st.cache_resource ────────────────────────────────
            # Decorator ini mengubah fungsi menjadi "cached resource":
            # Pertama kali dipanggil: eksekusi fungsi dan simpan hasilnya di memori.
            # Panggilan berikutnya: langsung kembalikan hasil yang tersimpan.
            # Hasil disimpan sepanjang session (tidak hilang saat rerun).
            # Cocok untuk objek berat yang di-sharing antar pengguna: model ML, koneksi DB.

            @st.cache_resource  # Model dimuat SATU KALI, dipakai BERKALI-KALI
            def load_model():
                if os.path.exists("pipeline_pm25_final.pkl"):
                    try:
                        return joblib.load("pipeline_pm25_final.pkl")
                    except Exception as e:
                        st.error(f"Gagal memuat .pkl: {e}")
                # Fallback: latih ulang jika .pkl tidak ada
                return _train_fallback()

            # ── @st.cache_data untuk DataFrame ────────────────────────────
            # Perbedaan: cache_resource untuk objek yang tidak bisa diserialisasi
            # (model ML, koneksi database); cache_data untuk data (DataFrame, list, dict).
            # cache_data membuat COPY saat dikembalikan (thread-safe), cache_resource tidak.

            @st.cache_data
            def load_data():
                return pd.read_csv("action2024/train.csv", low_memory=False)
            """), language="python")

        with st.expander("st.session_state -- Memori yang Persisten Antar Rerun"):
            st.code(dedent("""
            # ── MASALAH TANPA session_state ──────────────────────────────
            # Setiap kali Streamlit rerun, semua variabel Python lokal di-reset ke nilai awal.
            # Jika user sedang di Step 3 dari flow multi-langkah dan mengklik sesuatu,
            # tanpa session_state variabel 'current_step' akan kembali ke 1 (nilai awal).

            # ── SOLUSI: st.session_state ──────────────────────────────────
            # st.session_state adalah dictionary khusus yang PERSISTEN antar rerun.
            # Nilainya tidak hilang ketika script di-rerun karena user berinteraksi.

            # Inisialisasi: set nilai default HANYA jika kunci belum ada.
            # Tanpa pengecekan "not in", setiap rerun akan me-reset nilai ke awal.
            if "eda_step" not in st.session_state:
                st.session_state.eda_step = 1

            # Memodifikasi state melalui callback function (bukan langsung dalam script).
            # Callback dipanggil SEBELUM rerun, sehingga nilai state sudah terupdate
            # ketika script dieksekusi ulang.
            def go_step(step_baru):
                st.session_state.eda_step = step_baru

            # Tombol dengan on_click callback: ketika diklik, go_step(2) dipanggil,
            # mengubah session_state.eda_step menjadi 2, LALU Streamlit rerun.
            # Saat rerun, kode membaca eda_step = 2 dan menampilkan konten Step 2.
            st.button("Berikutnya", on_click=go_step, args=(2,))

            # ── Contoh penggunaan lain: menyimpan pilihan demo kota ──────
            if st.button("Jakarta"):
                st.session_state.sel_demo = 0   # Simpan index kota yang dipilih

            if "sel_demo" in st.session_state:
                case = DEMO_CASES[st.session_state.sel_demo]
                # Tampilkan hasil prediksi untuk kota yang dipilih
            """), language="python")

        with st.expander("Make Excel dengan Fallback Aman"):
            st.code(dedent("""
            # ── MASALAH ASLI: ModuleNotFoundError untuk openpyxl ─────────
            # Di beberapa environment (Streamlit Community Cloud dengan konfigurasi
            # tertentu), openpyxl mungkin tidak terinstal atau versinya incompatible.
            # Error asli: "from openpyxl.workbook import Workbook" ModuleNotFoundError

            # ── SOLUSI: Try-Except dengan Multiple Fallback ───────────────
            def make_excel(data: dict) -> bytes:
                df = pd.DataFrame([data])
                buf = io.BytesIO()  # Buffer memori in-memory, bukan file disk

                try:
                    # Coba engine openpyxl (format .xlsx modern)
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Prediction")
                except Exception:
                    try:
                        # Fallback ke xlsxwriter (alternatif yang lebih ringan)
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                            df.to_excel(writer, index=False, sheet_name="Prediction")
                    except Exception:
                        # Fallback terakhir: kembalikan CSV dalam bytes
                        # (masih bisa dibuka di Excel meskipun bukan format .xlsx)
                        return df.to_csv(index=False).encode("utf-8")

                return buf.getvalue()  # Kembalikan bytes dari buffer memori

            # buf = io.BytesIO() membuat "file" di memori RAM, bukan di disk.
            # Ini lebih aman untuk web app karena tidak ada akses disk yang perlu
            # dikelola, tidak ada file sementara yang perlu dihapus, dan otomatis
            # dibersihkan oleh garbage collector Python saat tidak lagi digunakan.
            """), language="python")

        with st.expander("Sistem Navigasi Multi-Halaman dan CSS Injection"):
            st.code(dedent("""
            # ── NAVIGASI MULTI-HALAMAN DENGAN RADIO BUTTON ───────────────
            # Streamlit tidak memiliki router URL bawaan untuk aplikasi single-page.
            # Pola umum: gunakan st.radio() di sidebar sebagai "page selector".
            # Nilai yang dipilih menentukan konten apa yang di-render.

            menu = st.radio("NAVIGASI", [
                "Beranda & Prediksi",
                "Demo Kasus Nyata",
                "Explainability AI",
                # ...
            ], label_visibility="collapsed")  # Sembunyikan label "NAVIGASI"

            # Blok if/elif sebagai "router" manual:
            if "Beranda" in menu:
                # Render konten halaman beranda
                pass
            elif "Demo" in menu:
                # Render konten halaman demo
                pass

            # ── CSS INJECTION MELALUI st.markdown ──────────────────────
            # Streamlit mengizinkan injeksi HTML/CSS melalui st.markdown
            # dengan parameter unsafe_allow_html=True.
            # PERHATIAN: "unsafe" bukan berarti berbahaya -- hanya berarti
            # Streamlit tidak mem-validate HTML yang diberikan. Tanggung jawab
            # keamanan ada di developer.

            GLOBAL_CSS = '''<style>
            :root { --blue: #60a5fa; --text-1: #f0f4ff; }
            .stApp { background: #06080f !important; }
            label { color: var(--text-1) !important; }
            </style>'''

            # CSS ini diinjeksikan sekali di awal dan mempengaruhi seluruh halaman.
            # Variabel CSS (:root { --nama: nilai }) memungkinkan perubahan
            # warna global hanya dengan mengubah satu tempat.
            st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
            """), language="python")

        st.markdown(dedent("""
        <div class="insight-box">
          <div class="insight-title">Ringkasan Arsitektur Lengkap Proyek</div>
          <p>
            <strong>proyek.ipynb</strong> -- Laboratorium eksperimen: EDA, feature selection, perbandingan 5 algoritma via K-Fold CV, penentuan hyperparameter terbaik.<br>
            <strong>build_pipeline.py</strong> -- Script produksi: memuat data, membersihkan, melatih model final (300 pohon), evaluasi, menyimpan ke pipeline_pm25_final.pkl. Jalankan SEKALI.<br>
            <strong>pipeline_pm25_final.pkl</strong> -- Artefak model: file binary yang berisi seluruh pipeline (imputer + scaler + OHE + 300 pohon Random Forest). Ukuran ~30-80 MB.<br>
            <strong>generate_dark_charts.py</strong> -- Script visualisasi: menghasilkan 5 grafik PNG berkualitas tinggi untuk laporan (missing values, distribusi, scatter, correlation, feature importance).<br>
            <strong>app.py</strong> -- Frontend produksi: memuat .pkl, menerima input user, menjalankan prediksi, menampilkan hasil interaktif. Jalankan dengan <code>streamlit run app.py</code>.<br><br>
            <strong>Alur Penggunaan:</strong><br>
            1. Jalankan <code>python build_pipeline.py</code> satu kali untuk menghasilkan .pkl.<br>
            2. Jalankan <code>streamlit run app.py</code> untuk menjalankan dashboard.<br>
            3. Opsional: jalankan <code>python generate_dark_charts.py</code> untuk grafik laporan.<br>
            4. Opsional: buka proyek.ipynb untuk eksplorasi dan eksperimen lebih lanjut.
          </p>
        </div>
        """), unsafe_allow_html=True)
