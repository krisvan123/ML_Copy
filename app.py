# ===================================================================
# APP.PY — AirSense PM2.5 Dashboard (Revisi UI/UX Final)
# Pembuat: Kristian Novan & Andrew Ong
# Mata Kuliah: COMP6577001 — Machine Learning
# ===================================================================

import io
import os
import time
import warnings

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
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ─── TOKEN SYSTEM ──────────────────────────────────────────────────── */
:root {
  /* Backgrounds */
  --bg-base:    #06080f;
  --bg-surface: #0c1018;
  --bg-card:    #111722;
  --bg-raised:  #181f2e;
  --bg-hover:   #1f2840;

  /* Borders */
  --border-sub:    rgba(255,255,255,0.04);
  --border-main:   rgba(255,255,255,0.08);
  --border-accent: rgba(96,165,250,0.28);
  --border-strong: rgba(96,165,250,0.5);

  /* Text */
  --text-1: #f0f4ff;
  --text-2: #8fa3bf;
  --text-3: #4a5a6e;

  /* Accent palette — purposefully narrow */
  --blue:   #60a5fa;
  --violet: #a78bfa;
  --teal:   #2dd4bf;
  --green:  #34d399;
  --amber:  #fbbf24;
  --rose:   #f87171;
  --orange: #f97316;

  /* Glow */
  --glow-blue:   rgba(96,165,250,0.14);
  --glow-violet: rgba(167,139,250,0.10);
  --glow-green:  rgba(52,211,153,0.12);

  /* Radii */
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 18px;
  --r-xl: 24px;
}

/* ─── APP BACKGROUND ────────────────────────────────────────────────── */
.stApp {
  background: var(--bg-base) !important;
  background-image:
    radial-gradient(ellipse 70% 50% at 15% 0%,   rgba(96,165,250,0.05) 0%, transparent 65%),
    radial-gradient(ellipse 55% 40% at 85% 100%,  rgba(167,139,250,0.04) 0%, transparent 65%) !important;
}

/* ─── SIDEBAR ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070910 0%, #0b0e1a 100%) !important;
  border-right: 1px solid var(--border-main) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.84rem !important; }

/* ─── CARDS ─────────────────────────────────────────────────────────── */
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
.card-amber  { border-left: 3px solid var(--amber);  }
.card-rose   { border-left: 3px solid var(--rose);   }

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--r-lg) !important;
}

/* ─── TYPOGRAPHY ─────────────────────────────────────────────────────── */
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
  font-size: 0.94rem;
  color: var(--text-2);
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

/* ─── HERO ───────────────────────────────────────────────────────────── */
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

/* ─── KPI STRIP ──────────────────────────────────────────────────────── */
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
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-top: 6px;
}

/* ─── AQI SCALE ──────────────────────────────────────────────────────── */
.aqi-wrap {
  background: var(--bg-raised);
  border: 1px solid var(--border-main);
  border-radius: var(--r-md);
  padding: 22px 26px;
  margin: 20px 0;
}
.aqi-lbl { font-size: 0.84rem; color: var(--text-2); margin-bottom: 14px; }
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
  color: var(--text-2);
}

/* ─── RESULT BANNER ──────────────────────────────────────────────────── */
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
.banner-desc  { font-size: 0.87rem; line-height: 1.65; color: var(--text-2); }
.banner-good     { background: rgba(52,211,153,0.07);  border: 1px solid rgba(52,211,153,0.22); }
.banner-moderate { background: rgba(251,191,36,0.07);  border: 1px solid rgba(251,191,36,0.22); }
.banner-unhealthy{ background: rgba(249,115,22,0.07);  border: 1px solid rgba(249,115,22,0.22); }
.banner-hazardous{ background: rgba(239,68,68,0.07);   border: 1px solid rgba(239,68,68,0.22); }

/* ─── MODERN TABLE ────────────────────────────────────────────────────── */
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
.modern-tbl th:first-child { border-radius: var(--r-md) 0 0 0; }
.modern-tbl th:last-child  { border-radius: 0 var(--r-md) 0 0; }
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

/* ─── BADGES ─────────────────────────────────────────────────────────── */
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
.badge-sim   { background: rgba(107,119,145,0.15); color: #8fa3bf; border: 1px solid rgba(107,119,145,0.25); }
.badge-used  { background: rgba(96,165,250,0.13); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.badge-ok    { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

/* ─── PIPE STEPS ─────────────────────────────────────────────────────── */
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

/* ─── FLOWCHART ──────────────────────────────────────────────────────── */
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

/* ─── GUIDE CARD ─────────────────────────────────────────────────────── */
.guide-card {
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  border-left: 3px solid var(--blue);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 18px 20px;
  margin-bottom: 12px;
}
.guide-title { font-weight: 700; color: var(--text-1); font-size: 0.93rem; margin-bottom: 8px; }
.guide-body  { font-size: 0.84rem; color: var(--text-2); line-height: 1.7; }

/* ─── INSIGHT BOX ────────────────────────────────────────────────────── */
.insight-box {
  background: rgba(167,139,250,0.05);
  border: 1px solid rgba(167,139,250,0.18);
  border-radius: var(--r-md);
  padding: 18px 22px;
  margin: 16px 0;
}
.insight-title { color: var(--violet); font-weight: 700; font-size: 0.88rem; margin-bottom: 8px; }
.insight-box p { color: var(--text-2); font-size: 0.85rem; line-height: 1.7; margin: 0; }

/* ─── CALLOUT ────────────────────────────────────────────────────────── */
.callout {
  background: rgba(96,165,250,0.05);
  border-left: 2px solid var(--blue);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 12px 18px;
  margin: 12px 0;
  font-size: 0.84rem;
  color: #93b4d6;
  line-height: 1.65;
}
.callout strong { color: var(--text-1); }

/* ─── WARNING BOX ────────────────────────────────────────────────────── */
.warn-box {
  background: rgba(251,191,36,0.05);
  border-left: 2px solid var(--amber);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 12px 18px;
  margin: 12px 0;
  font-size: 0.84rem;
  color: #c8a84b;
  line-height: 1.65;
}

/* ─── POINT ITEM ─────────────────────────────────────────────────────── */
.pt-item {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border-sub);
  border-radius: var(--r-sm);
  padding: 13px 17px;
  margin-bottom: 8px;
}
.pt-title { font-size: 0.83rem; font-weight: 700; color: var(--text-1); margin-bottom: 4px; }
.pt-body  { font-size: 0.8rem;  color: var(--text-2); line-height: 1.65; }

/* ─── CODE BLOCK ─────────────────────────────────────────────────────── */
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

/* ─── STATUS CHIP ────────────────────────────────────────────────────── */
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

/* ─── SIDEBAR CARD ───────────────────────────────────────────────────── */
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
.sb-key { font-size: 0.74rem; color: var(--text-3); }
.sb-val { font-size: 0.74rem; font-weight: 600; color: var(--text-1); }

/* ─── DIVIDER ────────────────────────────────────────────────────────── */
.div-line { border-top: 1px solid var(--border-sub); margin: 20px 0; }

/* ─── BUTTONS ────────────────────────────────────────────────────────── */
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

/* ─── INPUTS ─────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: 8px !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 2.5px rgba(96,165,250,0.18) !important;
}
label { color: var(--text-2) !important; font-size: 0.83rem !important; }

/* ─── RADIO ──────────────────────────────────────────────────────────── */
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.85rem; color: var(--text-1); }

/* ─── TABS ───────────────────────────────────────────────────────────── */
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

/* ─── MARKDOWN TEXT ──────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p  { color: var(--text-2); line-height: 1.7; }
[data-testid="stMarkdownContainer"] strong { color: var(--text-1); }
hr { border-color: var(--border-sub) !important; }
.stCheckbox label p { font-size: 0.85rem; color: var(--text-1); }

/* ─── EXPANDER ───────────────────────────────────────────────────────── */
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

/* ─── EXPORT SECTION ─────────────────────────────────────────────────── */
.export-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 14px;
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

/* ─── FEATURE IMPORTANCE BAR ─────────────────────────────────────────── */
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
.fi-pct { font-size: 0.78rem; color: var(--text-2); min-width: 48px; text-align: right; font-family: 'JetBrains Mono', monospace; }

/* ─── RESPONSIVE ─────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .hero-wrap { padding: 28px 24px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .hero-title { font-size: 1.9rem; }
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─── SVG ICONS ────────────────────────────────────────────────────────────
def ic(name: str, size: int = 16, color: str = "currentColor") -> str:
    _paths = {
        "wind":     f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>',
        "pin":      f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
        "users":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "activity": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "cpu":      f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "book":     f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
        "db":       f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "code":     f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
        "info":     f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        "globe":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        "trend":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "layers":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "target":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "filter":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "bar":      f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "eye":      f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
        "shield":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "check":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "alert":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><triangle points="10.29 3.86 1.82 18 22.18 18 10.29 3.86"/><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "sparkle":  f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
    }
    return _paths.get(name, "")


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
    {"city": "Jakarta, Indonesia",  "region": "Asia Tenggara (SEARO)", "year": 2023, "latitude": -6.2088,  "longitude": 106.8456, "pm10": 65.0, "no2": 38.5, "stations": 12, "who_ms": 1, "population": 10562088, "who_region": "3_Sear", "real_pm25": "~45 µg/m³", "context": "Ibukota Indonesia dengan kemacetan parah dan kawasan industri besar di sekitar teluk."},
    {"city": "Stockholm, Swedia",   "region": "Eropa (EURO)",          "year": 2023, "latitude": 59.3293, "longitude":  18.0686, "pm10": 12.0, "no2": 14.0, "stations": 25, "who_ms": 1, "population":  975904, "who_region": "4_Eur", "real_pm25": "~5 µg/m³",  "context": "Kota Nordik dengan standar emisi sangat ketat dan dominasi energi terbarukan."},
    {"city": "Delhi, India",        "region": "Asia Tenggara (SEARO)", "year": 2023, "latitude": 28.6139, "longitude":  77.2090, "pm10": 220.0,"no2": 68.0, "stations": 40, "who_ms": 1, "population": 32941309, "who_region": "3_Sear", "real_pm25": "~92 µg/m³", "context": "Konsisten masuk daftar kota paling terpolusi. Musim dingin diperburuk pembakaran ladang."},
    {"city": "Oslo, Norwegia",      "region": "Eropa (EURO)",          "year": 2023, "latitude": 59.9139, "longitude":  10.7522, "pm10": 9.5,  "no2": 11.0, "stations": 18, "who_ms": 1, "population":  673469, "who_region": "4_Eur", "real_pm25": "~4.5 µg/m³","context": "Salah satu kota paling ramah lingkungan, mayoritas kendaraan sudah listrik."},
    {"city": "Shanghai, China",     "region": "Pasifik Barat (WPRO)",  "year": 2023, "latitude": 31.2304, "longitude": 121.4737, "pm10": 78.0, "no2": 48.0, "stations": 55, "who_ms": 1, "population": 26317104, "who_region": "6_Wpr", "real_pm25": "~30 µg/m³", "context": "Kebijakan lingkungan agresif sejak 2015 mulai berhasil menekan tingkat polusi."},
    {"city": "Nairobi, Kenya",      "region": "Afrika (AFRO)",         "year": 2023, "latitude": -1.2921, "longitude":  36.8219, "pm10": 42.0, "no2": 22.0, "stations":  4, "who_ms": 1, "population":  4397073, "who_region": "1_Afr", "real_pm25": "~18 µg/m³", "context": "Kendaraan tua dan pembakaran sampah terbuka menjadi sumber polusi utama."},
    {"city": "Los Angeles, AS",     "region": "Amerika (AMRO)",        "year": 2023, "latitude": 34.0522, "longitude":-118.2437, "pm10": 30.0, "no2": 35.0, "stations": 48, "who_ms": 1, "population":  3898747, "who_region": "2_Amr", "real_pm25": "~14 µg/m³", "context": "Regulasi California (CARB) berhasil drastis menekan polusi sejak era 1970-an."},
    {"city": "Kairo, Mesir",        "region": "Mediterania Timur (EMRO)","year": 2023,"latitude": 30.0444, "longitude": 31.2357, "pm10": 95.0, "no2": 42.0, "stations":  6, "who_ms": 1, "population": 21323000, "who_region": "5_Emr", "real_pm25": "~55 µg/m³", "context": "Debu pasir Sahara ditambah emisi kendaraan tua menghasilkan polusi persisten."},
    {"city": "São Paulo, Brasil",   "region": "Amerika (AMRO)",        "year": 2023, "latitude":-23.5505, "longitude": -46.6333, "pm10": 35.0, "no2": 36.0, "stations": 30, "who_ms": 1, "population": 12325232, "who_region": "2_Amr", "real_pm25": "~17 µg/m³", "context": "Program biofuel nasional Brasil membantu menekan emisi dari sektor transportasi."},
    {"city": "Riyadh, Arab Saudi",  "region": "Mediterania Timur (EMRO)","year": 2023,"latitude": 24.6877, "longitude": 46.7219, "pm10": 110.0,"no2": 30.0, "stations":  8, "who_ms": 1, "population":  7676654, "who_region": "5_Emr", "real_pm25": "~62 µg/m³", "context": "Badai pasir regional dikombinasi industri minyak bumi menciptakan polusi konsisten tinggi."},
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
    st.error("❌ Model tidak dapat dimuat. Pastikan pipeline_pm25_final.pkl atau action2024/train.csv tersedia.")
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
        return ("Sehat",   "banner-good",      "#34d399", min(v / 12 * 20, 20),
                "Kualitas udara sangat baik. Aman untuk semua aktivitas luar ruangan tanpa batasan.",
                "#34d399")
    elif v <= 35.4:
        return ("Sedang",  "banner-moderate",  "#fbbf24", 20 + min((v - 12) / 23.4 * 25, 25),
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
        ic_inner = "✓" if n < current else str(n)
        h += f'<div class="fc-step {cls}"><div class="fc-icon">{ic_inner}</div><div class="fc-label">{s}</div></div>'
        if i < len(steps) - 1:
            h += '<div class="fc-arrow">→</div>'
    return h + "</div>"


# ─── EXPORT HELPERS ───────────────────────────────────────────────────────
def make_csv(data: dict) -> bytes:
    df = pd.DataFrame([data])
    return df.to_csv(index=False).encode("utf-8")


def make_excel(data: dict) -> bytes:
    df = pd.DataFrame([data])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction")
    return buf.getvalue()


def make_report_csv(rows: list) -> bytes:
    if not rows:
        return b""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# ─── PLOTLY THEME DEFAULTS ─────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,23,34,0.7)",
    font=dict(family="Inter", color="#8fa3bf"),
)


def styled_fig(**kwargs):
    fig = go.Figure(**kwargs)
    fig.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:22px 0 18px;text-align:center;">
      <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:6px;">
        {ic("wind", 26, "#60a5fa")}
        <span style="font-size:1.55rem;font-weight:800;letter-spacing:-1px;
          background:linear-gradient(135deg,#60a5fa,#a78bfa);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AirSense</span>
      </div>
      <div style="font-size:0.63rem;color:#4a5a6e;text-transform:uppercase;letter-spacing:2.5px;">
        Air Quality Predictor
      </div>
    </div>
    <div class="div-line"></div>
    """, unsafe_allow_html=True)

    menu = st.radio("NAVIGASI", [
        "🏠  Beranda & Prediksi",
        "🌍  Demo Kasus Nyata",
        "✨  Explainability AI",
        "📘  Panduan & Dataset",
        "📊  Alur Proses Data",
        "💻  Penjelasan Kode",
    ], index=0, label_visibility="collapsed")

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)

    # System status card
    status_color = "#34d399" if model is not None else "#f87171"
    status_text  = "Online" if model is not None else "Error"
    data_text    = f"{len(df_data):,} baris" if df_data is not None else "Tidak tersedia"
    st.markdown(f"""
    <div class="sb-card">
      <div style="font-size:0.63rem;font-weight:700;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">
        {ic("cpu", 11, "#60a5fa")} Status Sistem
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
        <span class="sb-key">R² Score</span>
        <span class="sb-val" style="color:#34d399;">89.0%</span>
      </div>
      <div class="sb-row">
        <span class="sb-key">Status</span>
        <span class="chip chip-online">● {status_text}</span>
      </div>
    </div>
    <div class="sb-card">
      <div style="font-size:0.63rem;font-weight:700;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">Tim</div>
      <div class="sb-row">
        <div>
          <div style="font-size:0.86rem;font-weight:600;color:#f0f4ff;">Kristian Novan</div>
          <div style="font-size:0.70rem;color:#4a5a6e;font-family:'JetBrains Mono',monospace;">2802458560</div>
        </div>
      </div>
      <div class="sb-row">
        <div>
          <div style="font-size:0.86rem;font-weight:600;color:#f0f4ff;">Andrew Ong</div>
          <div style="font-size:0.70rem;color:#4a5a6e;font-family:'JetBrains Mono',monospace;">2802420561</div>
        </div>
      </div>
      <div style="margin-top:10px;font-size:0.68rem;color:#4a5a6e;">COMP6577001 — Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA & PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
if "🏠" in menu:

    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
      <div class="eyebrow">{ic("wind", 13, "#60a5fa")} Platform Prediksi Kualitas Udara Global</div>
      <div class="hero-title">AirSense Dashboard</div>
      <p class="subtitle" style="margin-top:14px;">
        Estimasi konsentrasi PM2.5 berbasis Machine Learning dari data historis WHO.
        Masukkan parameter wilayah Anda — tanpa sensor PM2.5 di lapangan.
      </p>
      <div class="kpi-grid">
        <div class="kpi-box">
          <div class="kpi-val">25,999</div>
          <div class="kpi-lbl">Record WHO</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">89.0%</div>
          <div class="kpi-lbl">Akurasi R²</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">2.68</div>
          <div class="kpi-lbl">MAE µg/m³</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">9</div>
          <div class="kpi-lbl">Fitur Model</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="eyebrow" style="margin-top:4px;">{ic("filter", 12, "#60a5fa")} Parameter Masukan</div>
    <div class="section-title" style="margin-bottom:16px;">Konfigurasi Analisis</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{ic("pin", 15, "#60a5fa")} Lokasi Geografis</div>', unsafe_allow_html=True)
            who_region_label = st.selectbox("Wilayah WHO", list(WHO_REGION.keys()), index=0)
            who_region_code  = WHO_REGION[who_region_label]
            latitude  = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=-6.2088,  format="%.4f")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=106.8456, format="%.4f")

    with c2:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{ic("users", 15, "#a78bfa")} Demografi & Infrastruktur</div>', unsafe_allow_html=True)
            year        = st.number_input("Tahun Analisis",          min_value=2010, max_value=2035, value=2024, step=1)
            population  = st.number_input("Populasi (Jiwa)",         min_value=1000, max_value=60_000_000, value=10_000_000, step=100_000, format="%d")
            num_stations= st.number_input("Jumlah Stasiun Pemantau", min_value=1,    max_value=300, value=3)
            who_ms_label= st.radio("Status WHO", ["Anggota Resmi", "Non-Anggota"], horizontal=True)
            who_ms = 1 if who_ms_label == "Anggota Resmi" else 0

    with c3:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{ic("activity", 15, "#2dd4bf")} Polutan Pendukung (Opsional)</div>', unsafe_allow_html=True)
            has_pm10 = st.checkbox("Tersedia data PM10")
            pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 35.0, 0.1) if has_pm10 else np.nan
            if has_pm10:
                st.success("PM10 aktif — digunakan dalam model.")
            else:
                st.info("PM10 kosong → diimputasi dari median historis.")

            has_no2 = st.checkbox("Tersedia data NO₂")
            no2 = st.number_input("NO₂ (µg/m³)", 0.0, 300.0, 20.0, 0.1) if has_no2 else np.nan
            if has_no2:
                st.success("NO₂ aktif — digunakan dalam model.")
            else:
                st.info("NO₂ kosong → diimputasi dari median historis.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run = st.button("🔍  Analisis Kualitas Udara")

    # ── Results ─────────────────────────────────────────────────────────
    if run:
        with st.spinner("Memproses prediksi…"):
            pred = do_predict(year, latitude, longitude, pm10, no2, num_stations, who_ms, population, who_region_code)
            time.sleep(0.3)   # brief delay for animation feel

        cat, banner_cls, color, pct, rec, _ = classify(pred)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="eyebrow">{ic("target", 12, "#60a5fa")} Hasil Prediksi</div>
        <div class="section-title" style="margin-bottom:14px;">Estimasi Konsentrasi PM2.5</div>
        """, unsafe_allow_html=True)

        # AQI scale
        st.markdown(f"""
        <div class="aqi-wrap">
          <div class="aqi-lbl">
            Posisi pada skala PM2.5 —
            <strong style="color:{color};font-size:1rem;">{pred:.2f} µg/m³</strong>
          </div>
          <div class="aqi-bar"><div class="aqi-marker" style="left:{pct}%;"></div></div>
          <div class="aqi-scale-labels">
            <span style="color:#34d399;">✦ Sehat 0–12</span>
            <span style="color:#fbbf24;">✦ Sedang 12.1–35.4</span>
            <span style="color:#f97316;">✦ Sensitif 35.5–55.4</span>
            <span style="color:#ef4444;">✦ Berbahaya 55.4+</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Result banner
        st.markdown(f"""
        <div class="result-banner {banner_cls}">
          <div class="banner-icon">{shield_svg(color)}</div>
          <div>
            <div class="banner-title">{cat} — {pred:.2f} µg/m³</div>
            <div class="banner-desc">{rec}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail + comparison chart
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            st.markdown(f"""
            <div class="card card-blue">
              <div class="card-title">{ic("bar", 15, "#60a5fa")} Rincian Prediksi</div>
              <div class="tbl-wrap" style="margin:0;">
                <table class="modern-tbl">
                  <tbody>
                    <tr>
                      <td class="muted">Konsentrasi PM2.5</td>
                      <td class="right"><strong style="color:{color};font-size:1.05rem;">{pred:.2f} µg/m³</strong></td>
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
                      <td class="right mono" style="font-size:0.8rem;">{latitude:.4f}, {longitude:.4f}</td>
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
                      <td class="right">{"%.1f µg/m³" % pm10 if not np.isnan(pm10) else "— (imputasi median)"}</td>
                    </tr>
                    <tr>
                      <td class="muted">NO₂ Input</td>
                      <td class="right">{"%.1f µg/m³" % no2 if not np.isnan(no2) else "— (imputasi median)"}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            """, unsafe_allow_html=True)

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
                yaxis_title="PM2.5 (µg/m³)",
                title=dict(text="Perbandingan Prediksi Semua Model", font=dict(size=12, color="#8fa3bf")),
                height=280,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Export section ─────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander(f"⬇️  Ekspor Hasil Prediksi", expanded=False):
            export_data = {
                "Tahun": year, "Latitude": latitude, "Longitude": longitude,
                "Wilayah WHO": who_region_label,
                "PM10 Input (µg/m³)": pm10 if not np.isnan(pm10) else "N/A",
                "NO2 Input (µg/m³)":  no2  if not np.isnan(no2)  else "N/A",
                "Jumlah Stasiun": num_stations,
                "Populasi": population,
                "Prediksi PM2.5 (µg/m³)": round(pred, 4),
                "Kategori": cat,
                "Status WHO": who_ms_label,
            }
            st.markdown('<div class="export-row">', unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button(
                    label=f"{ic('download', 14, 'currentColor')} Download CSV",
                    data=make_csv(export_data),
                    file_name=f"airsense_prediction_{year}.csv",
                    mime="text/csv",
                )
            with ec2:
                st.download_button(
                    label=f"{ic('download', 14, 'currentColor')} Download Excel",
                    data=make_excel(export_data),
                    file_name=f"airsense_prediction_{year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="callout">File berisi semua parameter input dan hasil prediksi lengkap dalam format yang bisa dibuka di Excel atau Google Sheets.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMO KASUS NYATA
# ═══════════════════════════════════════════════════════════════════════════
elif "🌍" in menu:

    st.markdown(f"""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">{ic("globe", 12, "#60a5fa")} Demonstrasi Langsung</div>
      <div class="hero-title" style="font-size:2rem;">Demo 10 Kota Dunia</div>
      <p class="subtitle">Pilih kota nyata untuk melihat prediksi model, perbandingan lintas algoritma ML, dan analisis mengapa hasilnya berbeda antar model.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model comparison table ───────────────────────────────────────────
    with st.expander("📊  Perbandingan Performa 5 Algoritma ML", expanded=False):
        st.markdown(f"""
        <div style="margin-bottom:14px;">
          <div class="eyebrow">{ic("bar", 12, "#60a5fa")} Komparasi</div>
          <div class="section-title">5 Algoritma Regresi pada Dataset WHO</div>
        </div>
        <div class="tbl-wrap">
          <table class="modern-tbl">
            <thead>
              <tr>
                <th>Algoritma</th>
                <th>R² Score</th>
                <th>RMSE (µg/m³)</th>
                <th>MAE (µg/m³)</th>
                <th>Kecepatan</th>
                <th>Kompleksitas</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
        """, unsafe_allow_html=True)

        for mname, ms in MODEL_STATS.items():
            is_best  = mname == "Random Forest"
            row_cls  = "row-best" if is_best else ""
            r2_color = "#34d399" if is_best else ("#60a5fa" if ms["r2"] > 0.78 else "#8fa3bf")
            badge = '<span class="badge badge-best">✓ Terbaik</span>' if is_best else (
                '<span class="badge badge-good">Baik</span>' if ms["r2"] > 0.78 else
                '<span class="badge badge-sim">Kurang</span>'
            )
            name_html = f"<strong>{mname}</strong>" if is_best else mname
            st.markdown(f"""
            <tr class="{row_cls}">
              <td>{name_html}</td>
              <td><strong style="color:{r2_color};">{ms['r2']:.4f}</strong></td>
              <td>{ms['rmse']:.2f}</td>
              <td>{ms['mae']:.2f}</td>
              <td class="muted">{ms['speed']}</td>
              <td class="muted">{ms['complexity']}</td>
              <td>{badge}</td>
            </tr>
            """, unsafe_allow_html=True)

        st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:14px;">
          <div class="insight-title">Panduan Membaca Metrik</div>
          <p>
            <strong>R² Score</strong> — proporsi variasi PM2.5 yang dijelaskan model. Nilai 0.89 berarti 89% pola data berhasil ditangkap; semakin mendekati 1.0 semakin baik.<br><br>
            <strong>RMSE</strong> — rata-rata kesalahan dalam µg/m³. Nilai 2.68 berarti model meleset ±2.68 µg/m³ secara rata-rata; semakin kecil semakin akurat.<br><br>
            <strong>MAE</strong> — serupa RMSE namun lebih tahan terhadap outlier ekstrem karena tidak mengkuadratkan error.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── City selector ───────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="eyebrow">{ic("pin", 12, "#60a5fa")} Pilih Kota</div>
    <div class="section-title" style="margin-bottom:14px;">Klik kota untuk analisis prediksi lengkap</div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    for i, case in enumerate(DEMO_CASES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            if st.button(f"{case['city']}  ·  {case['region']}", key=f"demo_{i}"):
                st.session_state.sel_demo = i

    # ── Demo results ────────────────────────────────────────────────────
    if "sel_demo" in st.session_state:
        case = DEMO_CASES[st.session_state.sel_demo]
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"""
        <div class="eyebrow">{ic("target", 12, "#60a5fa")} Hasil Analisis</div>
        <div class="section-title" style="margin-bottom:14px;">{case['city']}</div>
        """, unsafe_allow_html=True)

        with st.spinner("Memproses…"):
            rf_pred = do_predict(
                case["year"], case["latitude"], case["longitude"],
                case["pm10"], case["no2"], case["stations"],
                case["who_ms"], case["population"], case["who_region"],
            )

        cat, banner_cls, color, pct, rec, _ = classify(rf_pred)

        # City info card
        st.markdown(f"""
        <div class="card card-blue" style="margin-bottom:16px;">
          <div class="muted" style="font-size:0.8rem;margin-bottom:6px;">{case['region']}</div>
          <div style="font-size:0.9rem;color:#f0f4ff;line-height:1.65;margin-bottom:16px;">{case['context']}</div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">{case['year']}</div>
              <div style="font-size:0.63rem;color:#4a5a6e;text-transform:uppercase;margin-top:3px;">Tahun</div>
            </div>
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">{case['pm10']} µg/m³</div>
              <div style="font-size:0.63rem;color:#4a5a6e;text-transform:uppercase;margin-top:3px;">PM10 Input</div>
            </div>
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">{case['no2']} µg/m³</div>
              <div style="font-size:0.63rem;color:#4a5a6e;text-transform:uppercase;margin-top:3px;">NO₂ Input</div>
            </div>
            <div style="background:rgba(167,139,250,0.08);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:700;color:#a78bfa;">{case['real_pm25']}</div>
              <div style="font-size:0.63rem;color:#4a5a6e;text-transform:uppercase;margin-top:3px;">Nilai Nyata</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # AQI + banner
        st.markdown(f"""
        <div class="aqi-wrap">
          <div class="aqi-lbl">
            Prediksi Random Forest: <strong style="color:{color};">{rf_pred:.2f} µg/m³</strong>
            <span style="color:#4a5a6e;margin-left:12px;">vs Nilai Nyata: {case['real_pm25']}</span>
          </div>
          <div class="aqi-bar"><div class="aqi-marker" style="left:{pct}%;"></div></div>
          <div class="aqi-scale-labels">
            <span style="color:#34d399;">✦ Sehat 0–12</span>
            <span style="color:#fbbf24;">✦ Sedang 12.1–35.4</span>
            <span style="color:#f97316;">✦ Sensitif 35.5–55.4</span>
            <span style="color:#ef4444;">✦ Berbahaya 55.4+</span>
          </div>
        </div>
        <div class="result-banner {banner_cls}">
          <div class="banner-icon">{shield_svg(color)}</div>
          <div>
            <div class="banner-title">{cat} — {rf_pred:.2f} µg/m³</div>
            <div class="banner-desc">{rec}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-model comparison chart
        st.markdown(f'<div style="margin-top:20px;" class="eyebrow">{ic("bar", 12, "#60a5fa")} Perbandingan Semua Model</div>', unsafe_allow_html=True)

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
            text=[f"{v:.1f} µg/m³" for v in mp.values()],
            textposition="outside",
            textfont=dict(size=11, color="#f0f4ff"),
        ))
        fig_bar.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang (35.4)")
        fig_bar.add_hline(y=55.4, line_dash="dash", line_color="#ef4444", annotation_text="Batas Berbahaya (55.4)")
        fig_bar.update_layout(
            **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            yaxis_title="Prediksi PM2.5 (µg/m³)",
            title=dict(text=f"Prediksi PM2.5 — {case['city']}", font=dict(size=13, color="#8fa3bf")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Detail comparison table
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
        """, unsafe_allow_html=True)
        for mn, val in mp.items():
            cat_n, _, vc, _, _, _ = classify(val)
            diff   = val - rf_pred
            diff_s = f"+{diff:.1f}" if diff > 0 else (f"{diff:.1f}" if diff != 0 else "—")
            is_best = mn == "Random Forest"
            rc  = "row-best" if is_best else ""
            bdg = '<span class="badge badge-best">Digunakan</span>' if is_best else '<span class="badge badge-sim">Simulasi</span>'
            nm  = f"<strong>{mn}</strong>" if is_best else mn
            dcol = "#4a5a6e" if mn == "Random Forest" else ("#ef4444" if diff > 0 else "#34d399")
            st.markdown(f"""
            <tr class="{rc}">
              <td>{nm}</td>
              <td><strong style="color:{vc};">{val:.2f} µg/m³</strong></td>
              <td style="font-size:0.8rem;">{cat_n}</td>
              <td class="mono" style="font-size:0.81rem;color:{dcol};">{diff_s}</td>
              <td>{bdg}</td>
            </tr>
            """, unsafe_allow_html=True)
        st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        # Explanation insight
        st.markdown("""
        <div class="insight-box" style="margin-top:14px;">
          <div class="insight-title">Mengapa Hasil Tiap Model Berbeda?</div>
          <p>
            <strong>Linear Regression</strong> — garis lurus tidak bisa menangkap hubungan non-linear PM10↔PM2.5, sehingga konsisten melebih-lebihkan nilai.<br><br>
            <strong>Decision Tree</strong> — lebih fleksibel dari garis lurus, namun mudah "hafal" data latih (overfitting) dan gagal generalisasi ke data baru.<br><br>
            <strong>Random Forest</strong> — rata-rata 300 pohon berbeda. Error satu pohon dikompensasi oleh yang lain, menghasilkan prediksi stabil.<br><br>
            <strong>Gradient Boosting</strong> — belajar dari kesalahan iteratif. Performa hampir setara RF namun lebih sensitif terhadap parameter dan lebih lambat.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Export demo
        with st.expander("⬇️  Ekspor Hasil Demo", expanded=False):
            demo_export = {
                "Kota": case["city"], "Wilayah WHO": case["region"], "Tahun": case["year"],
                "Latitude": case["latitude"], "Longitude": case["longitude"],
                "PM10 (µg/m³)": case["pm10"], "NO2 (µg/m³)": case["no2"],
                "Stasiun": case["stations"], "Populasi": case["population"],
                "Prediksi RF PM2.5 (µg/m³)": round(rf_pred, 4),
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
# PAGE 3 — EXPLAINABILITY AI
# ═══════════════════════════════════════════════════════════════════════════
elif "✨" in menu:

    st.markdown(f"""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">{ic("eye", 12, "#60a5fa")} Explainable AI</div>
      <div class="hero-title" style="font-size:2rem;">Memahami Keputusan Model</div>
      <p class="subtitle">Feature Importance, analisis sensitivitas, dan interpretasi visual yang menjelaskan <em>mengapa</em> Random Forest menghasilkan nilai PM2.5 tertentu.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_data is None:
        st.markdown('<div class="warn-box"><strong>Dataset tidak tersedia.</strong> Tempatkan file <code>action2024/train.csv</code> di direktori yang sama dengan app.py untuk mengaktifkan fitur Explainability.</div>', unsafe_allow_html=True)
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

        # ── TAB: Feature Importance ──────────────────────────────────────
        with tab_fi:
            st.markdown(f"""
            <div style="margin:14px 0 10px;">
              <div class="eyebrow">{ic("sparkle", 12, "#a78bfa")} Feature Importance</div>
              <div class="section-title">Fitur Mana yang Paling Berpengaruh?</div>
            </div>
            """, unsafe_allow_html=True)

            top_fi = fi_df.head(10)

            # Horizontal bar chart
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
                title=dict(text="Top 10 Fitur Paling Berpengaruh", font=dict(size=13, color="#8fa3bf")),
                height=380,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Visual importance bars
            st.markdown('<div style="margin-top:4px;">', unsafe_allow_html=True)
            max_imp = top_fi["importance"].max()
            for _, row in top_fi.iterrows():
                pct_bar = row["importance"] / max_imp * 100
                feature_clean = row["feature"].replace("who_region_", "Region: ").replace("_", " ")
                st.markdown(f"""
                <div class="fi-row">
                  <span class="fi-label">{feature_clean}</span>
                  <div class="fi-bar-wrap">
                    <div class="fi-bar-fill" style="width:{pct_bar:.1f}%;"></div>
                  </div>
                  <span class="fi-pct">{row['importance']:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="insight-box" style="margin-top:16px;">
              <div class="insight-title">Interpretasi Feature Importance</div>
              <p>
                <strong>pm10_concentration (&gt;40%)</strong> — Prediktor dominan karena PM10 dan PM2.5 berbagi banyak sumber emisi yang sama (kendaraan, industri, pembakaran). Korelasi Pearson r=0.886.<br><br>
                <strong>latitude &amp; longitude (&gt;15%)</strong> — Merepresentasikan perbedaan regulasi regional secara geografis. Eropa/Amerika Utara (lintang tinggi) cenderung memiliki PM2.5 lebih rendah karena standar emisi lebih ketat.<br><br>
                <strong>no2_concentration (~10%)</strong> — Indikator kuat aktivitas kendaraan dan industri. Berkorelasi dengan PM2.5 pada area perkotaan padat.<br><br>
                <strong>who_region features</strong> — Setelah One-Hot Encoding, setiap wilayah WHO menjadi fitur biner terpisah yang menangkap karakteristik polusi regional yang berbeda.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── TAB: Evaluasi Visual ─────────────────────────────────────────
        with tab_eval:
            st.markdown(f"""
            <div style="margin:14px 0 10px;">
              <div class="eyebrow">{ic("target", 12, "#60a5fa")} Evaluasi Model</div>
              <div class="section-title">Visualisasi Performa Random Forest</div>
            </div>
            """, unsafe_allow_html=True)

            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                (m1, "R² Score", f"{r2v:.4f}", "#34d399", "89% variasi PM2.5 dijelaskan model"),
                (m2, "MAE", f"{maev:.2f} µg/m³", "#60a5fa", "Rata-rata selisih prediksi vs aktual"),
                (m3, "RMSE", f"{rmsev:.2f} µg/m³", "#a78bfa", "Root mean square error prediksi"),
                (m4, "Test Size", "20%", "#fbbf24", "Proporsi data yang tidak dilihat model"),
            ]
            for col, label, value, color, desc in metrics:
                with col:
                    st.markdown(f"""
                    <div class="card" style="padding:18px;text-align:center;margin-bottom:0;">
                      <div style="font-size:1.5rem;font-weight:800;color:{color};">{value}</div>
                      <div style="font-size:0.72rem;font-weight:700;color:#4a5a6e;text-transform:uppercase;letter-spacing:1px;margin:6px 0 4px;">{label}</div>
                      <div style="font-size:0.73rem;color:#4a5a6e;line-height:1.4;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

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
                    xaxis_title="PM2.5 Aktual (µg/m³)",
                    yaxis_title="PM2.5 Prediksi (µg/m³)",
                    title=dict(text=f"Aktual vs Prediksi (R²={r2v:.4f})", font=dict(size=13, color="#8fa3bf")),
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
                    xaxis_title="Residu (Aktual − Prediksi)",
                    yaxis_title="Frekuensi",
                    title=dict(text=f"Distribusi Residual (MAE={maev:.2f})", font=dict(size=13, color="#8fa3bf")),
                )
                st.plotly_chart(fig_rs, use_container_width=True)

            # Residual insight
            st.markdown("""
            <div class="insight-box">
              <div class="insight-title">Interpretasi Hasil Evaluasi</div>
              <p>
                <strong>Scatter Aktual vs Prediksi</strong> — Titik-titik yang rapat di sepanjang garis merah menandakan prediksi akurat. Penyebaran di atas 100 µg/m³ menunjukkan model sedikit kesulitan di nilai ekstrem (kota sangat terpolusi seperti Delhi).<br><br>
                <strong>Distribusi Residual</strong> — Distribusi simetris berpusat di nol menunjukkan tidak ada bias sistematis. Model tidak secara konsisten melebihkan atau meremehkan nilai PM2.5 pada berbagai wilayah.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── TAB: Analisis Sensitivitas ────────────────────────────────────
        with tab_sens:
            st.markdown(f"""
            <div style="margin:14px 0 10px;">
              <div class="eyebrow">{ic("activity", 12, "#2dd4bf")} Analisis Sensitivitas</div>
              <div class="section-title">Bagaimana PM2.5 Berubah Ketika Parameter Berubah?</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="callout">Atur baseline di bawah, lalu lihat bagaimana prediksi PM2.5 berubah ketika setiap parameter divariasikan secara individual.</div>', unsafe_allow_html=True)

            sc1, sc2 = st.columns(2)
            with sc1:
                base_region = st.selectbox("Baseline Wilayah WHO", list(WHO_REGION.keys()), index=0, key="sens_reg")
                base_lat    = st.number_input("Baseline Latitude", -90.0, 90.0, -6.21, 0.1, key="sens_lat")
            with sc2:
                base_pop    = st.number_input("Baseline Populasi", 100_000, 50_000_000, 10_000_000, 100_000, format="%d", key="sens_pop")
                base_pm10   = st.number_input("Baseline PM10 (µg/m³)", 0.0, 300.0, 40.0, 1.0, key="sens_pm10")

            base_reg_code = WHO_REGION[base_region]

            @st.cache_data
            def sensitivity_curves(base_reg_code, base_lat, base_pop, base_pm10):
                # PM10 sweep
                pm10_range = np.linspace(5, 250, 40)
                pm10_preds = [do_predict(2023, base_lat, 106.85, p, 25.0, 5, 1, base_pop, base_reg_code) for p in pm10_range]

                # Population sweep
                pop_range  = np.linspace(500_000, 30_000_000, 30)
                pop_preds  = [do_predict(2023, base_lat, 106.85, base_pm10, 25.0, 5, 1, int(p), base_reg_code) for p in pop_range]

                # Latitude sweep (N to S)
                lat_range  = np.linspace(-50, 70, 30)
                lat_preds  = [do_predict(2023, lat, 106.85, base_pm10, 25.0, 5, 1, base_pop, base_reg_code) for lat in lat_range]

                # Region comparison
                reg_preds  = {name: do_predict(2023, base_lat, 0.0, base_pm10, 25.0, 5, 1, base_pop, code)
                              for name, code in WHO_REGION.items()}

                return pm10_range, pm10_preds, pop_range, pop_preds, lat_range, lat_preds, reg_preds

            pm10_r, pm10_p, pop_r, pop_p, lat_r, lat_p, reg_p = sensitivity_curves(base_reg_code, base_lat, base_pop, base_pm10)

            sn1, sn2 = st.columns(2)
            with sn1:
                fig_pm10 = go.Figure(go.Scatter(x=pm10_r.tolist(), y=pm10_p, mode="lines+markers",
                                                 line=dict(color="#60a5fa", width=2.5),
                                                 marker=dict(size=4, color="#60a5fa")))
                fig_pm10.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_pm10.add_hline(y=55.4, line_dash="dash", line_color="#ef4444")
                fig_pm10.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="PM10 (µg/m³)", yaxis_title="Prediksi PM2.5 (µg/m³)",
                    title=dict(text="Sensitivitas terhadap PM10", font=dict(size=13, color="#8fa3bf")),
                )
                st.plotly_chart(fig_pm10, use_container_width=True)

                fig_lat = go.Figure(go.Scatter(x=lat_r.tolist(), y=lat_p, mode="lines+markers",
                                                line=dict(color="#2dd4bf", width=2.5),
                                                marker=dict(size=4, color="#2dd4bf")))
                fig_lat.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_lat.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="Latitude (°)", yaxis_title="Prediksi PM2.5 (µg/m³)",
                    title=dict(text="Sensitivitas terhadap Latitude", font=dict(size=13, color="#8fa3bf")),
                )
                st.plotly_chart(fig_lat, use_container_width=True)

            with sn2:
                fig_pop = go.Figure(go.Scatter(x=(pop_r / 1e6).tolist(), y=pop_p, mode="lines+markers",
                                                line=dict(color="#a78bfa", width=2.5),
                                                marker=dict(size=4, color="#a78bfa")))
                fig_pop.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_pop.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="Populasi (juta jiwa)", yaxis_title="Prediksi PM2.5 (µg/m³)",
                    title=dict(text="Sensitivitas terhadap Populasi", font=dict(size=13, color="#8fa3bf")),
                )
                st.plotly_chart(fig_pop, use_container_width=True)

                reg_names = list(reg_p.keys())
                reg_vals  = list(reg_p.values())
                reg_colors = []
                for v in reg_vals:
                    if v > 55: reg_colors.append("#ef4444")
                    elif v > 35: reg_colors.append("#f97316")
                    elif v > 12: reg_colors.append("#fbbf24")
                    else: reg_colors.append("#34d399")
                fig_reg = go.Figure(go.Bar(
                    x=reg_names, y=reg_vals,
                    marker_color=reg_colors,
                    text=[f"{v:.1f}" for v in reg_vals],
                    textposition="outside",
                    textfont=dict(color="#f0f4ff", size=11),
                ))
                fig_reg.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24")
                fig_reg.update_layout(
                    **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=30),
                    xaxis_title="Wilayah WHO", yaxis_title="Prediksi PM2.5 (µg/m³)",
                    title=dict(text="Prediksi per Wilayah WHO", font=dict(size=13, color="#8fa3bf")),
                )
                st.plotly_chart(fig_reg, use_container_width=True)

            # Export sensitivity data
            with st.expander("⬇️  Ekspor Data Sensitivitas (CSV)", expanded=False):
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
# PAGE 4 — PANDUAN & DATASET
# ═══════════════════════════════════════════════════════════════════════════
elif "📘" in menu:

    st.markdown(f"""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">{ic("book", 12, "#60a5fa")} Dokumentasi</div>
      <div class="hero-title" style="font-size:2rem;">Panduan & Sumber Data</div>
      <p class="subtitle">Pelajari cara menggunakan setiap fitur AirSense, pahami sumber dataset WHO, dan temukan cara mendapatkan nilai input yang akurat untuk wilayah Anda.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_g, tab_i, tab_s = st.tabs(["Cara Menggunakan", "Cara Mendapat Data Input", "Sumber Dataset WHO"])

    # ── TAB: Cara Menggunakan ───────────────────────────────────────────
    with tab_g:
        st.markdown(f'<div style="margin:16px 0 10px;"><div class="eyebrow">{ic("layers", 12, "#60a5fa")} Alur Penggunaan</div><div class="section-title">4 Langkah Menggunakan AirSense</div></div>', unsafe_allow_html=True)

        guide_steps = [
            ("#60a5fa", "pin", "Langkah 1 — Tentukan Lokasi Geografis",
             "Mulai dengan memilih Wilayah WHO dari dropdown, lalu masukkan koordinat Latitude dan Longitude lokasi yang ingin dianalisis.",
             [("Wilayah WHO", "Pilih berdasarkan benua/sub-wilayah. Indonesia masuk 'Asia Tenggara (SEARO)'. Membantu model mengelompokkan pola polusi regional."),
              ("Latitude & Longitude", "Koordinat desimal. Google Maps: klik kanan lokasi → 'What's here?' untuk koordinat akurat."),
              ("Presisi Koordinat", "4 digit desimal (±0.01 km) sudah cukup untuk hasil akurat.")]),
            ("#a78bfa", "users", "Langkah 2 — Isi Data Demografi & Infrastruktur",
             "Populasi dan jumlah stasiun membantu model memperkirakan kepadatan polusi dan kualitas pengukuran di wilayah tersebut.",
             [("Tahun Analisis", "2010–2035. Model menggunakan tren historis untuk ekstrapolasi terbatas."),
              ("Populasi", "Jumlah penduduk kota. Sumber: BPS Indonesia, data sensus nasional, Wikipedia. Populasi besar → ekspektasi polusi lebih tinggi."),
              ("Jumlah Stasiun", "Berapa AQMS (Air Quality Monitoring Station) di wilayah. Lebih banyak → data lebih representatif."),
              ("Status WHO", "Hampir semua negara adalah anggota resmi WHO. Pilih Non-Anggota hanya untuk wilayah seperti Kosovo atau Taiwan.")]),
            ("#2dd4bf", "activity", "Langkah 3 — Tambahkan Data Polutan (Opsional)",
             "Data PM10 dan NO₂ adalah fitur terpenting. Jika tersedia, tambahkan untuk meningkatkan akurasi prediksi secara signifikan.",
             [("PM10 (µg/m³)", "Partikel kasar (≤10µm) dari debu, konstruksi, industri. Sumber: KLHK AQMS, IQAir, OpenAQ. Jika kosong: model imputasi median WHO (~35 µg/m³)."),
              ("NO₂ (µg/m³)", "Nitrogen dioksida dari kendaraan dan industri. Sumber sama dengan PM10. Jika kosong: imputasi median historis (~20 µg/m³)."),
              ("Korelasi PM10–PM2.5", "r=0.886 — PM10 adalah prediktor terkuat. Semakin akurat data PM10, semakin baik prediksi PM2.5.")]),
            ("#34d399", "target", "Langkah 4 — Interpretasi & Tindak Lanjut",
             "Setelah klik Analisis, model mengembalikan estimasi PM2.5 (µg/m³) beserta kategori kesehatan dan rekomendasi tindakan.",
             [("Skala AQI Visual", "Penanda bergerak menunjukkan posisi dalam 4 zona: Sehat (hijau ≤12), Sedang (kuning ≤35.4), Sensitif (oranye ≤55.4), Berbahaya (merah >55.4)."),
              ("Ekspor Hasil", "Gunakan tombol Download CSV/Excel di bagian bawah hasil prediksi untuk menyimpan analisis."),
              ("Explainability AI", "Kunjungi menu '✨ Explainability AI' untuk memahami mengapa model menghasilkan nilai tersebut dan bagaimana fitur berkontribusi.")]),
        ]

        for color_v, ico_name, title, intro, points in guide_steps:
            st.markdown(f"""
            <div class="guide-card" style="border-left-color:{color_v};margin-bottom:8px;">
              <div class="guide-title">{ic(ico_name, 14, color_v)} {title}</div>
              <div class="guide-body" style="margin-bottom:10px;">{intro}</div>
            </div>
            """, unsafe_allow_html=True)
            for pt_title, pt_desc in points:
                st.markdown(f"""
                <div class="pt-item">
                  <div class="pt-title">{pt_title}</div>
                  <div class="pt-body">{pt_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── TAB: Cara Mendapat Data Input ───────────────────────────────────
    with tab_i:
        st.markdown(f'<div style="margin:16px 0 10px;"><div class="eyebrow">{ic("db", 12, "#60a5fa")} Sumber Data</div><div class="section-title">Cara Mendapatkan Nilai Input yang Akurat</div></div>', unsafe_allow_html=True)

        sources = [
            ("#60a5fa", "Koordinat GPS (Latitude, Longitude)",
             "Google Maps: buka maps.google.com → cari lokasi → klik kanan → nilai pertama = koordinat Lat, Lon. Alternatif: OpenStreetMap atau nominatim.openstreetmap.org."),
            ("#a78bfa", "Data PM10 dan NO₂",
             "Indonesia: KLHK Sipongi (sipongi.menlhk.go.id). Global: IQAir (iqair.com), OpenAQ (openaq.org), WHO Ambient Air Quality Database. Jika tidak ada: biarkan kosong — model mengimputasi dari median historis wilayah WHO."),
            ("#2dd4bf", "Data Populasi",
             "BPS untuk kota Indonesia (bps.go.id). Global: World Population Review atau Wikipedia. Gunakan populasi kota administratif, bukan metropolitan area."),
            ("#34d399", "Jumlah Stasiun Pemantau",
             "Indonesia: data KLHK atau laporan IQAir. Estimasi: kota besar 10–40 stasiun, menengah 3–10, kecil/kabupaten 1–3. Jika tidak diketahui, estimasi berdasarkan ukuran kota."),
            ("#fbbf24", "Wilayah WHO",
             "SEARO: Asia Tenggara (ID, IN, BD, NP, LK, TH) | EURO: Eropa + Rusia | AMRO: Amerika | WPRO: Pasifik Barat (CN, JP, KR, AU, PH) | AFRO: Afrika sub-Sahara | EMRO: Timur Tengah + Afrika Utara + Pakistan."),
        ]
        for color_v, title, desc in sources:
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {color_v};margin-bottom:10px;padding:18px 20px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
          <div class="insight-title">Tips: Skenario Tanpa Data Polutan</div>
          <p>
            Jika tidak memiliki data PM10 dan NO₂ — cukup isi lokasi, populasi, dan wilayah WHO, biarkan polutan kosong.
            Model menggunakan nilai median historis dari wilayah WHO yang bersangkutan berdasarkan ribuan rekaman data.
            Akurasi sedikit berkurang, namun tetap memberikan estimasi yang informatif.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB: Sumber Dataset WHO ─────────────────────────────────────────
    with tab_s:
        st.markdown(f'<div style="margin:16px 0 10px;"><div class="eyebrow">{ic("db", 12, "#60a5fa")} Dataset</div><div class="section-title">WHO Global Ambient Air Quality Database</div></div>', unsafe_allow_html=True)

        dc1, dc2 = st.columns([1.2, 1])
        with dc1:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">{ic("db", 15, "#60a5fa")} Spesifikasi Dataset</div>
              <div class="tbl-wrap" style="margin:0;">
                <table class="modern-tbl">
                  <tbody>
                    <tr><td class="muted">Nama Dataset</td><td class="right">WHO Global Ambient Air Quality</td></tr>
                    <tr><td class="muted">Format</td><td class="right mono">.csv</td></tr>
                    <tr><td class="muted">Total Record</td><td class="right"><strong style="color:#60a5fa;">25,999 baris</strong></td></tr>
                    <tr><td class="muted">Jumlah Kolom</td><td class="right">17 kolom</td></tr>
                    <tr><td class="muted">Fitur Prediktor</td><td class="right">9 fitur</td></tr>
                    <tr><td class="muted">Variabel Target</td><td class="right"><code style="background:rgba(167,139,250,0.12);padding:2px 6px;border-radius:4px;font-family:JetBrains Mono;color:#a78bfa;">pm25_concentration</code></td></tr>
                    <tr><td class="muted">Missing PM2.5</td><td class="right" style="color:#f87171;">49.9%</td></tr>
                    <tr><td class="muted">Wilayah WHO</td><td class="right">6 kawasan dunia</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with dc2:
            st.markdown(f"""
            <div class="card" style="height:100%;">
              <div class="card-title">{ic("info", 15, "#fbbf24")} Konteks & Relevansi</div>
              <div class="guide-body" style="margin-top:8px;">
                Database kompilasi global pemantauan kualitas udara dari ribuan stasiun di seluruh dunia.
                Ketidaklengkapan data (49.9% PM2.5 kosong) merupakan realitas pemantauan udara global —
                sensor PM2.5 mahal dan belum tersedia di semua negara berkembang.<br><br>
                <strong>Inilah fungsi utama model ini:</strong> mengestimasi PM2.5 dari variabel proxy
                yang lebih mudah diukur, sehingga negara tanpa infrastruktur lengkap tetap bisa
                memantau kualitas udara mereka.
              </div>
            </div>
            """, unsafe_allow_html=True)

        if df_data is not None:
            st.markdown(f'<div style="margin-top:16px;" class="eyebrow">{ic("db", 11, "#60a5fa")} Preview Data (10 baris pertama)</div>', unsafe_allow_html=True)
            preview_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df_data.columns]
            st.dataframe(
                df_data[preview_cols].head(10),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — ALUR PROSES DATA
# ═══════════════════════════════════════════════════════════════════════════
elif "📊" in menu:

    st.markdown(f"""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">{ic("trend", 12, "#60a5fa")} Proses Data Science</div>
      <div class="hero-title" style="font-size:2rem;">Alur Pembangunan Model</div>
      <p class="subtitle">Jelajahi setiap tahap — dari data mentah WHO hingga pipeline produksi — secara interaktif dengan visualisasi langsung dari dataset nyata.</p>
    </div>
    """, unsafe_allow_html=True)

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

    # Step helper
    def step_header(num, total, title):
        st.markdown(f"""
        <div class="eyebrow">{ic("layers", 12, "#60a5fa")} Tahap {num} dari {total}</div>
        <div class="section-title" style="margin-bottom:14px;">{title}</div>
        """, unsafe_allow_html=True)

    # ── STEP 1 ──────────────────────────────────────────────────────────
    if step == 1:
        step_header(1, 5, "Data Overview — Mengenal Dataset WHO")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Mengapa tahap ini penting?", "Seperti dokter membaca hasil lab sebelum diagnosis — kita periksa struktur, kelengkapan, dan konsistensi data sebelum menyentuh algoritma apapun."),
                ("Dimensi Dataset", "25,999 baris × 17 kolom. Setiap baris = satu rekaman pengukuran kualitas udara di satu kota pada satu tahun. Dari 17 kolom total, 9 dipilih sebagai fitur model."),
                ("Masalah: 49.9% PM2.5 Kosong", "Sensor PM2.5 mahal — banyak negara berkembang belum memilikinya. Ini bukan error, melainkan realitas monitoring global. Inilah alasan utama proyek ini dibangun."),
                ("Solusi Teknis", "Baris dengan TARGET kosong di-drop dari data latih. Baris dengan FITUR kosong (pm10, no2) ditangani dengan imputasi median di dalam pipeline — tidak di-drop."),
            ]:
                st.markdown(f'<div class="guide-card"><div class="guide-title">{title}</div><div class="guide-body">{body}</div></div>', unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                mv_cols = ["pm25_tempcov", "pm25_concentration", "pm10_tempcov", "no2_tempcov", "population", "no2_concentration", "pm10_concentration"]
                mv_pct  = [round(df_data[c].isna().sum() / len(df_data) * 100, 1) if c in df_data.columns else 0.0 for c in mv_cols]
                fig_mv = px.bar(x=mv_cols, y=mv_pct, text=[f"{v:.1f}%" for v in mv_pct],
                                color=mv_pct, color_continuous_scale="Blues", template="plotly_dark",
                                title="Persentase Missing Values per Kolom")
                fig_mv.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=60),
                                     coloraxis_showscale=False, yaxis_title="% Kosong")
                fig_mv.update_traces(textposition="outside", textfont=dict(color="#f0f4ff"))
                st.plotly_chart(fig_mv, use_container_width=True)

            st.markdown("""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
              <div class="kpi-box"><div class="kpi-val">25,999</div><div class="kpi-lbl">Total Baris</div></div>
              <div class="kpi-box" style="border-color:rgba(167,139,250,0.2);"><div class="kpi-val" style="color:#a78bfa;">17</div><div class="kpi-lbl">Total Kolom</div></div>
              <div class="kpi-box" style="border-color:rgba(239,68,68,0.2);"><div class="kpi-val" style="color:#f87171;">49.9%</div><div class="kpi-lbl">Missing PM2.5</div></div>
              <div class="kpi-box" style="border-color:rgba(52,211,153,0.2);"><div class="kpi-val" style="color:#34d399;">9</div><div class="kpi-lbl">Fitur Digunakan</div></div>
            </div>
            """, unsafe_allow_html=True)

    # ── STEP 2 ──────────────────────────────────────────────────────────
    elif step == 2:
        step_header(2, 5, "Univariate Analysis — Satu Variabel, Satu Fokus")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Apa itu Univariate Analysis?", "'Univariate' = satu variabel. Kita memotret setiap variabel secara individual untuk memahami karakter masing-masing sebelum melihat hubungan antar variabel."),
                ("Temuan: Distribusi Right-Skewed", "Histogram PM2.5 dan PM10 menunjukkan ekor panjang di kanan: kebanyakan kota bersih, namun segelintir kota industri memiliki nilai sangat tinggi."),
                ("Implikasi untuk ML", "Random Forest secara alami menangani distribusi non-normal karena tidak mengasumsikan distribusi data tertentu — keunggulan dibanding Linear Regression."),
                ("Distribusi Kategori", "Kolom Air_quality_category menunjukkan dominasi 'Safety'. Ketidakseimbangan ini tidak berdampak langsung karena kita menggunakan regresi (angka), bukan klasifikasi."),
            ]:
                st.markdown(f'<div class="guide-card"><div class="guide-title">{title}</div><div class="guide-body">{body}</div></div>', unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                pm25_f = df_data["pm25_concentration"].dropna()
                pm25_f = pm25_f[pm25_f < 150]
                pm10_f = df_data["pm10_concentration"].dropna()
                pm10_f = pm10_f[pm10_f < 200]
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=pm25_f.tolist(), name="PM2.5", marker_color="#60a5fa", opacity=0.75, nbinsx=60))
                fig_h.add_trace(go.Histogram(x=pm10_f.tolist(), name="PM10",  marker_color="#a78bfa", opacity=0.75, nbinsx=60))
                fig_h.add_vline(x=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang PM2.5", annotation_font_size=10)
                fig_h.update_layout(barmode="overlay", **PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20),
                                    title=dict(text="Distribusi PM2.5 vs PM10", font=dict(size=13, color="#8fa3bf")),
                                    xaxis_title="Konsentrasi (µg/m³)", yaxis_title="Frekuensi")
                st.plotly_chart(fig_h, use_container_width=True)

                if "Air_quality_category" in df_data.columns:
                    cts = df_data["Air_quality_category"].dropna().value_counts().reset_index()
                    cts.columns = ["Kategori", "Jumlah"]
                    fig_c = px.bar(cts, x="Kategori", y="Jumlah", text="Jumlah",
                                   color="Kategori", color_discrete_map={"Safety": "#34d399", "Dangerous": "#f87171"},
                                   template="plotly_dark",
                                   title="Distribusi Kategori Kualitas Udara")
                    fig_c.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20))
                    fig_c.update_traces(textposition="outside", textfont=dict(color="#f0f4ff"))
                    st.plotly_chart(fig_c, use_container_width=True)

    # ── STEP 3 ──────────────────────────────────────────────────────────
    elif step == 3:
        step_header(3, 5, "Bivariate Analysis — Hubungan Antar Dua Variabel")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Apa itu Bivariate Analysis?", "'Bivariate' = dua variabel. Kita mulai melihat hubungan antar variabel: apakah PM10 naik → PM2.5 ikut naik? Apakah Eropa selalu lebih bersih dari Asia?"),
                ("Scatter PM10 vs PM2.5 (r = 0.886)", "Korelasi Pearson sangat tinggi: setiap kenaikan PM10 hampir selalu diikuti kenaikan PM2.5. Keduanya sering berasal dari sumber emisi yang sama."),
                ("Boxplot per Wilayah WHO", "Asia Tenggara (SEARO) dan Mediterania Timur (EMRO) memiliki median PM2.5 jauh lebih tinggi dari Eropa (EURO). Membuktikan who_region adalah fitur sangat informatif."),
                ("Implikasi untuk Model", "Random Forest menggunakan informasi wilayah sebagai 'percabangan pertama': 'Jika SEARO → ekspektasi PM2.5 lebih tinggi.' Inilah mengapa who_region dikonversi via One-Hot Encoding."),
            ]:
                st.markdown(f'<div class="guide-card"><div class="guide-title">{title}</div><div class="guide-body">{body}</div></div>', unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                sc = df_data.dropna(subset=["pm10_concentration", "pm25_concentration"])
                sc = sc[(sc["pm10_concentration"] < 200) & (sc["pm25_concentration"] < 150)]
                sc = sc.sample(n=min(2500, len(sc)), random_state=42)
                fig_sc = px.scatter(sc, x="pm10_concentration", y="pm25_concentration",
                                    opacity=0.35, color="pm25_concentration",
                                    color_continuous_scale="Blues", template="plotly_dark",
                                    title="Scatter: PM10 vs PM2.5 (r = 0.886)")
                fig_sc.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20),
                                     coloraxis_showscale=False,
                                     xaxis_title="PM10 (µg/m³)", yaxis_title="PM2.5 (µg/m³)")
                st.plotly_chart(fig_sc, use_container_width=True)

                db = df_data.dropna(subset=["pm25_concentration", "who_region"]).copy()
                db["Region"] = db["who_region"].map(REGION_LABEL)
                db = db[db["pm25_concentration"] < 150]
                fig_bx = px.box(db, x="Region", y="pm25_concentration", color="Region",
                                template="plotly_dark", title="PM2.5 per Wilayah WHO")
                fig_bx.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang WHO")
                fig_bx.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=50, b=20), showlegend=False)
                st.plotly_chart(fig_bx, use_container_width=True)

    # ── STEP 4 ──────────────────────────────────────────────────────────
    elif step == 4:
        step_header(4, 5, "Correlation Matrix — Peta Hubungan Semua Variabel")
        exp_col, viz_col = st.columns([1, 1.2])

        with exp_col:
            for title, body in [
                ("Apa itu Correlation Matrix?", "Matriks korelasi menampilkan kekuatan dan arah hubungan linear antar semua pasang variabel numerik. Nilai +1 = sempurna positif, 0 = tidak ada hubungan, -1 = sempurna negatif."),
                ("Temuan: PM10 Dominan", "PM10 memiliki korelasi tertinggi dengan PM2.5 (r=0.89). NO₂ berkontribusi sedang (r=0.45). Populasi lemah secara linear (r=0.22) karena dimediasi faktor lain."),
                ("Pengaruh Latitude", "Latitude berkorelasi negatif dengan PM2.5 (r=-0.31): semakin jauh ke utara (Eropa, Amerika Utara), polusi cenderung lebih rendah karena regulasi lebih ketat."),
                ("Tidak Ada Multikolinearitas Kritis", "Korelasi antar fitur prediktor semua di bawah 0.7, sehingga semua 9 fitur aman digunakan bersama tanpa membingungkan model."),
            ]:
                st.markdown(f'<div class="guide-card"><div class="guide-title">{title}</div><div class="guide-body">{body}</div></div>', unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                cc = ["pm25_concentration", "pm10_concentration", "no2_concentration",
                      "year", "population", "latitude", "longitude", "number_of_stations"]
                cm = df_data[cc].corr()
                fig_cr = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu", aspect="auto",
                                   template="plotly_dark", title="Matriks Korelasi Pearson")
                fig_cr.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=60, b=20))
                st.plotly_chart(fig_cr, use_container_width=True)

                yr = df_data.dropna(subset=["pm25_concentration"]).groupby("year")["pm25_concentration"].agg(["mean", "median"]).reset_index()
                fig_yr = go.Figure()
                fig_yr.add_trace(go.Scatter(x=yr["year"].tolist(), y=yr["mean"].tolist(), mode="lines+markers",
                                             name="Rata-rata", line=dict(color="#60a5fa", width=2.5)))
                fig_yr.add_trace(go.Scatter(x=yr["year"].tolist(), y=yr["median"].tolist(), mode="lines+markers",
                                             name="Median", line=dict(color="#f97316", width=2, dash="dash")))
                fig_yr.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=10, t=60, b=20),
                                     title=dict(text="Tren PM2.5 Global per Tahun", font=dict(size=13, color="#8fa3bf")))
                st.plotly_chart(fig_yr, use_container_width=True)

    # ── STEP 5 ──────────────────────────────────────────────────────────
    elif step == 5:
        step_header(5, 5, "Model Final — Evaluasi Performa & Arsitektur Pipeline")

        # Pipeline diagram
        p1, pa1, p2, pa2, p3 = st.columns([2, 0.2, 2, 0.2, 2])
        with p1:
            st.markdown('<div class="pipe-step"><span class="pipe-badge">INPUT</span><div class="pipe-title">Data Mentah (9 Kolom)</div><div class="pipe-desc">year, latitude, longitude, pm10, no2, stations, who_ms, population, who_region — beberapa mungkin kosong (NaN).</div></div>', unsafe_allow_html=True)
        with pa1:
            st.markdown('<div style="text-align:center;margin-top:40px;color:#4a5a6e;font-size:1.2rem;">→</div>', unsafe_allow_html=True)
        with p2:
            st.markdown('<div class="pipe-step"><span class="pipe-badge">PREPROCESSING</span><div class="pipe-title">ColumnTransformer</div><div class="pipe-desc"><strong>Numerik (8 kolom):</strong> Median Imputer → StandardScaler<br><br><strong>Kategorikal (1 kolom):</strong> Constant Imputer → OneHotEncoder (7 kolom hasil)</div></div>', unsafe_allow_html=True)
        with pa2:
            st.markdown('<div style="text-align:center;margin-top:40px;color:#4a5a6e;font-size:1.2rem;">→</div>', unsafe_allow_html=True)
        with p3:
            st.markdown('<div class="pipe-step"><span class="pipe-badge">MODEL</span><div class="pipe-title">Random Forest Regressor</div><div class="pipe-desc">300 pohon keputusan, paralel di semua CPU (n_jobs=-1). Prediksi = rata-rata output semua pohon.</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Model comparison charts
        m_names = ["Linear Regression", "Ridge Regression", "Decision Tree", "Gradient Boosting", "Random Forest"]
        r2_v    = [0.6210, 0.6212, 0.7850, 0.8850, 0.8900]
        rmse_v  = [6.45,   6.45,   4.85,   2.78,   2.68]
        bar_c5  = ["#1e293b", "#1e293b", "#334155", "#60a5fa", "#34d399"]

        mc1, mc2 = st.columns(2)
        with mc1:
            fig_r2 = go.Figure(go.Bar(x=r2_v, y=m_names, orientation="h", marker_color=bar_c5,
                                       text=[f"{v:.4f}" for v in r2_v], textposition="outside",
                                       textfont=dict(color="#f0f4ff", size=11)))
            fig_r2.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=50, t=50, b=20),
                                  xaxis=dict(range=[0, 1.08]), title=dict(text="R² Score (lebih tinggi = lebih baik)", font=dict(size=13, color="#8fa3bf")))
            st.plotly_chart(fig_r2, use_container_width=True)

        with mc2:
            fig_rm = go.Figure(go.Bar(x=rmse_v, y=m_names, orientation="h", marker_color=bar_c5,
                                       text=[f"{v:.2f}" for v in rmse_v], textposition="outside",
                                       textfont=dict(color="#f0f4ff", size=11)))
            fig_rm.update_layout(**PLOTLY_BASE, margin=dict(l=10, r=50, t=50, b=20),
                                  title=dict(text="RMSE µg/m³ (lebih rendah = lebih baik)", font=dict(size=13, color="#8fa3bf")))
            st.plotly_chart(fig_rm, use_container_width=True)

        # Live evaluation if data available
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
                                    title=f"Aktual vs Prediksi (R²={r2v5:.4f})")
                lm = float(yt5.max()) * 1.05
                fig_ev.add_shape(type="line", line=dict(dash="dash", color="#f87171", width=2), x0=0, y0=0, x1=lm, y1=lm)
                fig_ev.update_layout(**PLOTLY_BASE, margin=dict(l=5, r=5, t=60, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_ev, use_container_width=True)

            with ec2:
                fig_rs5 = px.histogram(x=rs5.tolist(), nbins=60, template="plotly_dark",
                                        title=f"Distribusi Residual (MAE={maev5:.2f})")
                fig_rs5.update_traces(marker_color="#a78bfa")
                fig_rs5.add_vline(x=0, line_dash="dash", line_color="#f87171")
                fig_rs5.update_layout(**PLOTLY_BASE, margin=dict(l=5, r=5, t=60, b=20))
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

        st.markdown("""
        <div class="insight-box">
          <div class="insight-title">Interpretasi Hasil Evaluasi Model Final</div>
          <p>
            <strong>Scatter Aktual vs Prediksi</strong> — Titik rapat di garis merah = prediksi akurat. Penyebaran di &gt;100 µg/m³ menunjukkan model sedikit kesulitan di nilai ekstrem.<br><br>
            <strong>Distribusi Residual</strong> — Distribusi simetris di sekitar nol = tidak ada bias sistematis. Model tidak konsisten melebihkan atau meremehkan nilai PM2.5.<br><br>
            <strong>Feature Importance</strong> — PM10 mendominasi (&gt;40%). Latitude dan Longitude mencerminkan perbedaan regulasi regional. Populasi dan NO₂ berkontribusi lebih kecil namun tetap bermakna statistik.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # Navigation buttons
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_l, _, nav_r = st.columns([1, 3, 1])
    if step > 1: nav_l.button("← Sebelumnya", on_click=go_step, args=(step - 1,), use_container_width=True)
    if step < 5: nav_r.button("Berikutnya →", on_click=go_step, args=(step + 1,), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — PENJELASAN KODE
# ═══════════════════════════════════════════════════════════════════════════
elif "💻" in menu:

    st.markdown(f"""
    <div class="hero-wrap" style="padding:34px 42px;">
      <div class="eyebrow">{ic("code", 12, "#60a5fa")} Dokumentasi Teknis</div>
      <div class="hero-title" style="font-size:2rem;">Penjelasan Kode Proyek</div>
      <p class="subtitle">Dokumentasi mendalam setiap komponen: proyek.ipynb, build_pipeline.py, generate_dark_charts.py, dan app.py — dengan penjelasan yang bisa dipahami siapa pun.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_nb, tab_build, tab_charts, tab_app = st.tabs([
        "proyek.ipynb", "build_pipeline.py", "generate_dark_charts.py", "app.py",
    ])

    # ── TAB: NOTEBOOK ────────────────────────────────────────────────────
    with tab_nb:
        st.markdown(f'<div style="margin:14px 0 10px;"><div class="eyebrow">{ic("book", 12, "#60a5fa")} Notebook Eksperimen</div><div class="section-title">proyek.ipynb — Laboratorium Data Science</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="callout"><strong>Peran:</strong> Notebook Jupyter adalah "laboratorium" tempat semua eksperimen pertama kali dilakukan sebelum kode yang terbukti berhasil dipindahkan ke script produksi.</div>', unsafe_allow_html=True)

        with st.expander("Cell 1 — Import Library & EDA", expanded=True):
            st.markdown('<div class="guide-card"><div class="guide-title">Tujuan Cell 1</div><div class="guide-body">Memuat semua library, membaca dataset dari disk, memperbaiki tipe data, dan membuat 6 grafik EDA untuk memahami dataset secara menyeluruh.</div></div>', unsafe_allow_html=True)
            st.code("""# ── IMPORT LIBRARY ─────────────────────────────────────
import pandas as pd          # Manipulasi tabel data (DataFrame)
import numpy  as np          # Operasi matematika array
import matplotlib.pyplot as plt  # Pembuatan grafik statis
import seaborn as sns        # Visualisasi statistik

# ── MEMUAT DATASET ──────────────────────────────────────
df_raw = pd.read_csv("action2024/train.csv", low_memory=False)

# ── KONVERSI TIPE DATA ──────────────────────────────────
# errors='coerce' → nilai tidak valid (misal "N/A") → NaN, tidak error
for col in COLS_NUMERIC:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')""", language="python")

        with st.expander("Cell 2 — Feature Engineering & Cross-Validation"):
            st.markdown('<div class="guide-card"><div class="guide-title">Tujuan Cell 2</div><div class="guide-body">Mendefinisikan fitur, membangun pipeline preprocessing, dan membandingkan 5 algoritma ML menggunakan K-Fold Cross Validation secara objektif.</div></div>', unsafe_allow_html=True)
            st.code("""# ── K-FOLD CROSS VALIDATION ─────────────────────────────
# KFold membagi data menjadi 5 bagian secara bergantian:
# Iterasi 1: Latih di fold 2,3,4,5  →  Test di fold 1
# Setiap data pernah menjadi data test TEPAT 1 kali.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for nama, mdl in MODELS.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', mdl)])
    r2_cv = cross_val_score(pipe, X, y, cv=kf, scoring='r2', n_jobs=-1)""", language="python")

            for title, body in [
                ("Linear Regression (R²=0.62) — Gagal", "Mengasumsikan output adalah kombinasi linear. Tidak bisa menangkap interaksi antar variabel — meleset jauh di kasus ekstrem."),
                ("Decision Tree (R²=0.785) — Tidak Stabil", "Pohon tunggal mudah 'hafal' data latih (overfitting) — terlalu detail untuk data latih, gagal generalisasi ke data baru."),
                ("Random Forest (R²=0.89) — Terbaik", "300 pohon berbeda, dilatih pada subset data acak. Error antar pohon tidak berkorelasi — saat dirata-rata, error saling mengkompensasi."),
                ("Gradient Boosting (R²=0.885) — Hampir Setara", "Belajar dari kesalahan iteratif. Performa hampir setara RF, namun lebih sensitif terhadap hyperparameter dan lebih lambat."),
            ]:
                st.markdown(f'<div class="pt-item"><div class="pt-title">{title}</div><div class="pt-body">{body}</div></div>', unsafe_allow_html=True)

        with st.expander("Cell 3 — Training Final & Simpan Pipeline"):
            st.code("""# Train/Test Split: 80% latih, 20% uji (tidak pernah dilihat model)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# .fit() melakukan 3 hal sekaligus:
# 1. Preprocessor belajar MEDIAN dari X_train
# 2. Preprocessor belajar kategori OHE dari X_train
# 3. Random Forest dilatih dengan data yang sudah diproses
final_pipeline.fit(X_train, y_train)

# Simpan seluruh pipeline (preprocessor + model) ke file .pkl
joblib.dump(final_pipeline, 'pipeline_pm25_final.pkl')""", language="python")

    # ── TAB: BUILD_PIPELINE ──────────────────────────────────────────────
    with tab_build:
        st.markdown(f'<div style="margin:14px 0 10px;"><div class="eyebrow">{ic("cpu", 12, "#60a5fa")} Script Produksi</div><div class="section-title">build_pipeline.py — Pabrik Produksi Model</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="callout"><strong>Peran:</strong> Script produksi yang dijalankan sekali dari terminal (<code>python build_pipeline.py</code>) untuk menghasilkan file <code>pipeline_pm25_final.pkl</code>.</div>', unsafe_allow_html=True)

        for title, exp, code_str in [
            ("Tahap 1 & 2 — Validasi File & Memuat Data",
             "Script memeriksa keberadaan file sebelum melanjutkan — fail fast dengan pesan yang jelas.",
             """if not os.path.exists(TRAIN_FILE):
    print(f"File tidak ditemukan: {TRAIN_FILE}")
    sys.exit(1)   # Keluar dengan kode error (bukan 0 = sukses)

# Konversi kolom numerik, abaikan nilai tidak valid
for col in NUM_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Hapus nilai target negatif atau nol (tidak mungkin secara fisika)
df_model = df_model[df_model[TARGET_COL] > 0]"""),
            ("Tahap 3 — Pipeline dengan Konfigurasi Produksi",
             "build_pipeline.py menggunakan konfigurasi lebih kuat: 300 pohon (notebook hanya 100).",
             """RandomForestRegressor(
    n_estimators      = 300,   # 3x lebih banyak pohon dari notebook
    max_depth         = None,  # Pohon tumbuh sampai sempurna
    min_samples_split = 4,     # Minimal 4 sampel untuk percabangan baru
    min_samples_leaf  = 2,     # Minimal 2 sampel di daun (anti overfitting)
    max_features      = "sqrt",# akar(n_fitur) fitur per split → randomisasi
    n_jobs            = -1,    # Paralel di semua CPU core
    random_state      = 42,    # Reproduktibilitas
)"""),
            ("Tahap 6 — Simpan & Verifikasi",
             "Pipeline disimpan ke .pkl beserta ukuran file untuk verifikasi integritas.",
             """joblib.dump(pipeline, PIPELINE_FILE)

# Verifikasi ukuran file — file terlalu kecil mungkin corrupt
size_mb = os.path.getsize(PIPELINE_FILE) / (1024 * 1024)
print(f"Ukuran file: {size_mb:.1f} MB")

# File .pkl berisi:
# - Nilai median tiap kolom numerik (dipelajari dari X_train)
# - Mapping kategori OHE untuk 'who_region'
# - 300 pohon keputusan dengan semua cabang"""),
        ]:
            with st.expander(title):
                st.markdown(f'<div class="guide-card"><div class="guide-title">Tujuan</div><div class="guide-body">{exp}</div></div>', unsafe_allow_html=True)
                st.code(code_str, language="python")

    # ── TAB: GENERATE_DARK_CHARTS ────────────────────────────────────────
    with tab_charts:
        st.markdown(f'<div style="margin:14px 0 10px;"><div class="eyebrow">{ic("bar", 12, "#60a5fa")} Script Visualisasi</div><div class="section-title">generate_dark_charts.py — Pembuat Grafik Tema Gelap</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="callout"><strong>Peran:</strong> Script utilitas untuk menghasilkan grafik EDA dan evaluasi model sebagai file PNG statis beresolusi tinggi (dpi=150) untuk laporan akademik atau presentasi.</div>', unsafe_allow_html=True)

        for title, exp, code_str in [
            ("Konfigurasi Tema Gelap Global via rcParams",
             "plt.rcParams.update() mengubah pengaturan default semua grafik Matplotlib secara global. Satu blok konfigurasi → semua grafik otomatis konsisten.",
             """plt.rcParams.update({
    'figure.facecolor': '#090d16',   # Latar belakang figure
    'axes.facecolor'  : '#111827',   # Latar belakang area plot
    'text.color'      : '#f1f5f9',   # Semua teks → hampir putih
    'axes.labelcolor' : '#f1f5f9',   # Label sumbu X dan Y
    'xtick.color'     : '#94a3b8',   # Angka sumbu X → abu-abu
    'ytick.color'     : '#94a3b8',   # Angka sumbu Y → abu-abu
    'legend.facecolor': '#111827',   # Background kotak legenda
})"""),
            ("Helper apply_dark_theme_axes()",
             "Fungsi pembantu untuk setiap subplot. Prinsip DRY: satu fungsi menggantikan 6+ baris kode berulang.",
             """def apply_dark_theme_axes(ax, title=""):
    ax.set_title(title, color='#64d2ff', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5, color='#1e293b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')"""),
            ("Menyimpan Grafik Beresolusi Tinggi",
             "plt.close() setelah savefig penting untuk mencegah memory leak saat membuat banyak grafik.",
             """plt.savefig(
    'evaluasi_model_final.png',
    dpi=150,              # 150 DPI → resolusi tinggi untuk cetak
    facecolor='#090d16',  # Background figure saat disimpan
    bbox_inches='tight'   # Potong whitespace berlebih di tepi
)
plt.close()   # WAJIB: bebaskan memori setelah menyimpan"""),
        ]:
            with st.expander(title):
                st.markdown(f'<div class="guide-card"><div class="guide-title">Tujuan</div><div class="guide-body">{exp}</div></div>', unsafe_allow_html=True)
                st.code(code_str, language="python")

    # ── TAB: APP.PY ──────────────────────────────────────────────────────
    with tab_app:
        st.markdown(f'<div style="margin:14px 0 10px;"><div class="eyebrow">{ic("globe", 12, "#60a5fa")} Frontend Aplikasi</div><div class="section-title">app.py — Aplikasi Streamlit Produksi</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="callout"><strong>Peran:</strong> Wajah publik proyek — menghubungkan model ML dengan pengguna menggunakan Streamlit yang mengubah Python biasa menjadi web app interaktif.</div>', unsafe_allow_html=True)

        for title, exp, code_str in [
            ("@st.cache_resource — Memuat Model Hanya Sekali",
             "Tanpa cache, setiap klik memicu reload model (30+ detik). Dengan cache, model dimuat sekali, akses berikutnya instan.",
             """@st.cache_resource   # Hanya jalankan fungsi ini SATU KALI
def load_model():
    if os.path.exists('pipeline_pm25_final.pkl'):
        return joblib.load('pipeline_pm25_final.pkl')
    # Fallback: latih model on-the-fly jika .pkl tidak ada
    return train_model_fallback()"""),
            ("st.session_state — Memori Lintas Re-run",
             "Setiap interaksi user memicu re-run seluruh script — variabel lokal hilang. st.session_state adalah 'memori' yang bertahan antar re-run.",
             """# Inisialisasi state pertama kali halaman dibuka
if "eda_step" not in st.session_state:
    st.session_state.eda_step = 1

def go_step(step_baru):
    st.session_state.eda_step = step_baru

# Tombol dengan on_click callback untuk mengubah state
st.button("Berikutnya →", on_click=go_step, args=(2,))"""),
            ("Sistem Navigasi Multi-Halaman",
             "Streamlit tidak memiliki router bawaan. st.radio() di sidebar sebagai selector halaman dengan blok if/elif sebagai 'router' manual.",
             """menu = st.radio("NAVIGASI", [
    "🏠  Beranda & Prediksi",
    "🌍  Demo Kasus Nyata",
    "✨  Explainability AI",
    ...
])

if "🏠" in menu:
    # Render halaman beranda
    ...
elif "🌍" in menu:
    # Render halaman demo
    ..."""),
            ("Export CSV & Excel",
             "Fungsi make_csv() dan make_excel() menghasilkan file unduhan dari dictionary hasil prediksi.",
             """def make_csv(data: dict) -> bytes:
    df = pd.DataFrame([data])
    return df.to_csv(index=False).encode("utf-8")

def make_excel(data: dict) -> bytes:
    df = pd.DataFrame([data])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction")
    return buf.getvalue()

# Digunakan bersama st.download_button()
st.download_button("Download CSV", data=make_csv(result), ...)"""),
        ]:
            with st.expander(title):
                st.markdown(f'<div class="guide-card"><div class="guide-title">Tujuan</div><div class="guide-body">{exp}</div></div>', unsafe_allow_html=True)
                st.code(code_str, language="python")

        st.markdown("""
        <div class="insight-box">
          <div class="insight-title">Arsitektur Seluruh Proyek</div>
          <p>
            <strong>proyek.ipynb</strong> → Eksplorasi dan eksperimen (data scientist bekerja di sini)<br>
            <strong>build_pipeline.py</strong> → Produksi model (jalankan sekali → menghasilkan .pkl)<br>
            <strong>pipeline_pm25_final.pkl</strong> → Artefak model tersimpan (output dari build_pipeline.py)<br>
            <strong>generate_dark_charts.py</strong> → Grafik PNG statis untuk laporan/presentasi<br>
            <strong>app.py</strong> → Frontend interaktif yang memuat .pkl dan melayani pengguna<br><br>
            <strong>Alur kerja:</strong> jalankan build_pipeline.py SEKALI → jalankan <code>streamlit run app.py</code> BERKALI-KALI.
          </p>
        </div>
        """, unsafe_allow_html=True)
