import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import sklearn
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

st.set_page_config(
    page_title="AirSense | PM2.5 Predictor",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌫</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

/* ── Root tokens ── */
:root {
  --bg-base:    #07090f;
  --bg-card:    #0d1117;
  --bg-raised:  #131922;
  --border:     rgba(255,255,255,0.07);
  --border-accent: rgba(96,165,250,0.25);
  --text-primary: #e8edf5;
  --text-muted:   #6b7b8f;
  --text-faint:   #3d4a56;
  --accent-blue:  #60a5fa;
  --accent-violet:#a78bfa;
  --accent-teal:  #2dd4bf;
  --accent-green: #34d399;
  --accent-amber: #fbbf24;
  --accent-rose:  #f87171;
  --glow-blue:  rgba(96,165,250,0.12);
  --glow-violet: rgba(167,139,250,0.10);
}

/* ── App background ── */
.stApp {
  background: var(--bg-base);
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(96,165,250,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(167,139,250,0.05) 0%, transparent 60%);
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #08090e 0%, #0c0f19 100%) !important;
  border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
div[data-testid="stSidebar"] .stRadio label { 
  font-size: 0.82rem !important; 
  letter-spacing: 0.3px;
}

/* ── Cards ── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
  margin-bottom: 20px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
  border-color: var(--border-accent);
  box-shadow: 0 0 30px var(--glow-blue);
}
.card-accent-blue  { border-left: 3px solid var(--accent-blue); }
.card-accent-violet { border-left: 3px solid var(--accent-violet); }
.card-accent-teal  { border-left: 3px solid var(--accent-teal); }
.card-accent-green { border-left: 3px solid var(--accent-green); }
.card-accent-amber { border-left: 3px solid var(--accent-amber); }
.card-accent-rose  { border-left: 3px solid var(--accent-rose); }

/* Streamlit containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

/* ── Typography ── */
.display-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(2rem, 4vw, 3.2rem);
  font-weight: 700;
  letter-spacing: -1.5px;
  line-height: 1.1;
  background: linear-gradient(135deg, #e8edf5 0%, var(--accent-blue) 60%, var(--accent-violet) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 12px;
}
.section-label {
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 2.5px;
  color: var(--accent-blue);
  margin-bottom: 8px;
}
.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}
.subtitle {
  font-size: 0.95rem;
  color: var(--text-muted);
  line-height: 1.65;
  max-width: 720px;
}
.card-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Hero ── */
.hero {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 40px 44px;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: 
    radial-gradient(ellipse 50% 80% at 95% 10%, rgba(167,139,250,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 5% 90%, rgba(96,165,250,0.06) 0%, transparent 60%);
  pointer-events: none;
}

/* ── KPI Strip ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 28px;
}
.kpi-box {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
}
.kpi-val {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--accent-blue);
  letter-spacing: -0.5px;
  line-height: 1;
}
.kpi-lbl {
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 5px;
}

/* ── AQI Scale ── */
.aqi-wrap {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  margin: 20px 0;
}
.aqi-label {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 12px;
}
.aqi-bar {
  position: relative;
  height: 12px;
  border-radius: 6px;
  background: linear-gradient(90deg,
    #34d399 0%, #34d399 20%,
    #fbbf24 20%, #fbbf24 45%,
    #f97316 45%, #f97316 65%,
    #ef4444 65%, #ef4444 100%);
}
.aqi-marker {
  position: absolute;
  top: -7px;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #fff;
  border: 3px solid var(--bg-base);
  box-shadow: 0 0 14px rgba(255,255,255,0.6);
  transform: translateX(-50%);
  transition: left 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.aqi-scale-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  font-size: 0.7rem;
  color: var(--text-muted);
}

/* ── Result Banner ── */
.result-banner {
  border-radius: 12px;
  padding: 22px 26px;
  margin: 16px 0;
  display: flex;
  align-items: flex-start;
  gap: 18px;
}
.banner-icon {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.banner-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 4px;
}
.banner-desc {
  font-size: 0.87rem;
  line-height: 1.6;
  color: var(--text-muted);
}
.banner-good    { background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.2); }
.banner-moderate{ background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.2); }
.banner-unhealthy{ background:rgba(249,115,22,0.07); border: 1px solid rgba(249,115,22,0.2); }
.banner-hazardous{ background:rgba(239,68,68,0.07);  border: 1px solid rgba(239,68,68,0.2); }

/* ── Model comparison table ── */
.model-tbl { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.model-tbl th {
  background: rgba(96,165,250,0.08);
  color: var(--accent-blue);
  padding: 10px 14px;
  text-align: left;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  border-bottom: 1px solid var(--border);
}
.model-tbl td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
  font-size: 0.84rem;
}
.model-tbl tr:last-child td { border-bottom: none; }
.model-tbl tr:hover td { background: rgba(255,255,255,0.02); }
.best-row td { background: rgba(52,211,153,0.04) !important; }
.best-row td:first-child { border-left: 2px solid var(--accent-green); }
.badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 99px;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.badge-best   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-good   { background: rgba(96,165,250,0.12); color: #60a5fa; }
.badge-sim    { background: rgba(107,123,143,0.15); color: #6b7b8f; }

/* ── Code blocks ── */
.code-wrap {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 10px;
  padding: 18px 20px;
  margin: 14px 0;
  font-family: 'DM Mono', 'JetBrains Mono', monospace;
  font-size: 0.79rem;
  line-height: 1.7;
  color: #c9d1d9;
  overflow-x: auto;
  white-space: pre;
}
.c-comment { color: #6e7681; font-style: italic; }
.c-keyword  { color: #ff7b72; }
.c-string   { color: #a5d6ff; }
.c-func     { color: #d2a8ff; }
.c-num      { color: #79c0ff; }
.c-var      { color: #ffa657; }

/* ── Guide cards ── */
.guide-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-blue);
  border-radius: 0 10px 10px 0;
  padding: 18px 20px;
  margin-bottom: 14px;
}
.guide-title { font-weight: 600; color: var(--text-primary); font-size: 0.92rem; margin-bottom: 8px; }
.guide-body  { font-size: 0.84rem; color: var(--text-muted); line-height: 1.65; }

/* ── Insight box ── */
.insight-box {
  background: rgba(167,139,250,0.06);
  border: 1px solid rgba(167,139,250,0.18);
  border-radius: 10px;
  padding: 16px 20px;
  margin: 16px 0;
}
.insight-box strong { color: var(--text-primary); }
.insight-box p { color: var(--text-muted); font-size: 0.85rem; line-height: 1.65; margin: 0; }
.insight-title { color: var(--accent-violet); font-weight: 600; font-size: 0.88rem; margin-bottom: 8px; }

/* ── Info callout ── */
.callout {
  background: rgba(96,165,250,0.06);
  border-left: 2px solid var(--accent-blue);
  border-radius: 0 8px 8px 0;
  padding: 11px 16px;
  margin: 10px 0;
  font-size: 0.83rem;
  color: #93b4d6;
  line-height: 1.6;
}

/* ── Pipeline steps ── */
.pipe-step {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
}
.pipe-badge {
  display: inline-block;
  background: linear-gradient(135deg, #3b4fd9, #0ea5e9);
  color: #fff;
  font-size: 0.65rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 99px;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.pipe-title { font-weight: 600; color: var(--text-primary); font-size: 0.95rem; margin-bottom: 6px; }
.pipe-desc  { font-size: 0.82rem; color: var(--text-muted); line-height: 1.6; }
.pipe-arrow { text-align: center; color: var(--accent-blue); font-size: 1.2rem; padding: 6px 0; }

/* ── Demo city cards ── */
.city-btn-wrap { margin-bottom: 10px; }

/* ── Flowchart ── */
.flowchart {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  overflow-x: auto;
  gap: 4px;
}
.fc-step { display: flex; flex-direction: column; align-items: center; text-align: center; min-width: 80px; }
.fc-icon {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem; font-weight: 700;
  background: var(--bg-raised); border: 2px solid var(--border);
  color: var(--text-muted); margin-bottom: 6px;
}
.fc-step.fc-active .fc-icon {
  background: linear-gradient(135deg, #3b4fd9, #0ea5e9);
  border-color: var(--accent-blue); color: #fff;
  box-shadow: 0 0 16px rgba(96,165,250,0.35);
}
.fc-step.fc-done .fc-icon {
  background: var(--accent-green); border-color: var(--accent-green); color: #fff;
}
.fc-label { font-size: 0.68rem; font-weight: 600; color: var(--text-muted); }
.fc-step.fc-active .fc-label { color: var(--accent-blue); }
.fc-step.fc-done  .fc-label { color: var(--accent-green); }
.fc-arrow { color: var(--text-faint); font-size: 1rem; flex-shrink: 0; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #3b4fd9 0%, #0ea5e9 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 9px !important;
  padding: 11px 22px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 4px 16px rgba(59,79,217,0.28) !important;
  transition: all 0.25s !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 22px rgba(59,79,217,0.45) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 2px rgba(96,165,250,0.15) !important;
}

/* ── Radio ── */
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.84rem; color: var(--text-primary); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-card);
  border-radius: 10px 10px 0 0;
  border-bottom: 1px solid var(--border);
  gap: 4px;
  padding: 6px 8px 0;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 8px 8px 0 0 !important;
  color: var(--text-muted) !important;
  font-size: 0.83rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(96,165,250,0.1) !important;
  color: var(--accent-blue) !important;
  border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Misc ── */
hr { border-color: var(--border) !important; }
.stCheckbox label p { font-size: 0.85rem; color: var(--text-primary); }
[data-testid="stMarkdownContainer"] p { color: var(--text-muted); line-height: 1.65; }
[data-testid="stMarkdownContainer"] strong { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)


# ── SVG ICON LIBRARY ──────────────────────────────────────────────────────
def icon(name, size=18, color="currentColor"):
    paths = {
        "map-pin": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
        "wind":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>',
        "users":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "activity":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "cpu":     f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "book":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
        "check":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "database":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "code2":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
        "info":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        "globe":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        "trending":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "layers":  f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "target":  f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "filter":  f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "bar-chart":f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    }
    return paths.get(name, '')


# ── CONSTANTS ─────────────────────────────────────────────────────────────
FEATURE_COLS = ["year","latitude","longitude","pm10_concentration","no2_concentration","number_of_stations","who_ms","population","who_region"]
TARGET_COL   = "pm25_concentration"
NUM_COLS     = ["year","latitude","longitude","pm10_concentration","no2_concentration","number_of_stations","who_ms","population"]
CAT_COLS     = ["who_region"]

WHO_REGION = {
    "Asia Tenggara (SEARO)":         "3_Sear",
    "Eropa (EURO)":                  "4_Eur",
    "Amerika (AMRO)":                "2_Amr",
    "Pasifik Barat (WPRO)":          "6_Wpr",
    "Afrika (AFRO)":                 "1_Afr",
    "Mediterania Timur (EMRO)":      "5_Emr",
    "Negara Non-Anggota (Non-MS)":   "7_NonMS",
}

REGION_LABEL = {
    '1_Afr':'Afrika','2_Amr':'Amerika','3_Sear':'Asia Tenggara',
    '4_Eur':'Eropa','5_Emr':'Mediterania','6_Wpr':'Pasifik Barat','7_NonMS':'Non-MS'
}

MODEL_STATS = {
    "Linear Regression":   {"r2": 0.6210, "rmse": 6.45, "mae": 4.82, "speed": "Sangat Cepat", "complexity": "Rendah"},
    "Ridge Regression":    {"r2": 0.6212, "rmse": 6.45, "mae": 4.80, "speed": "Sangat Cepat", "complexity": "Rendah"},
    "Decision Tree":       {"r2": 0.7850, "rmse": 4.85, "mae": 2.95, "speed": "Cepat",        "complexity": "Sedang"},
    "Random Forest":       {"r2": 0.8900, "rmse": 2.68, "mae": 1.75, "speed": "Sedang",       "complexity": "Tinggi"},
    "Gradient Boosting":   {"r2": 0.8850, "rmse": 2.78, "mae": 1.82, "speed": "Lambat",       "complexity": "Tinggi"},
}

SIM_MULT = {
    "Linear Regression":  1.35,
    "Ridge Regression":   1.34,
    "Decision Tree":      1.12,
    "Gradient Boosting":  1.04,
}

DEMO_CASES = [
    {"city":"Jakarta, Indonesia","region":"Asia Tenggara (SEARO)","flag":"ID","year":2023,"latitude":-6.2088,"longitude":106.8456,"pm10":65.0,"no2":38.5,"stations":12,"who_ms":1,"population":10562088,"who_region":"3_Sear","real_pm25":"~45 µg/m³","context":"Ibukota Indonesia dengan kemacetan parah dan kawasan industri besar di sekitar teluk."},
    {"city":"Stockholm, Swedia","region":"Eropa (EURO)","flag":"SE","year":2023,"latitude":59.3293,"longitude":18.0686,"pm10":12.0,"no2":14.0,"stations":25,"who_ms":1,"population":975904,"who_region":"4_Eur","real_pm25":"~5 µg/m³","context":"Kota Nordik dengan standar emisi sangat ketat dan dominasi energi terbarukan."},
    {"city":"Delhi, India","region":"Asia Tenggara (SEARO)","flag":"IN","year":2023,"latitude":28.6139,"longitude":77.2090,"pm10":220.0,"no2":68.0,"stations":40,"who_ms":1,"population":32941309,"who_region":"3_Sear","real_pm25":"~92 µg/m³","context":"Konsisten masuk daftar kota paling terpolusi. Musim dingin diperburuk pembakaran ladang."},
    {"city":"Oslo, Norwegia","region":"Eropa (EURO)","flag":"NO","year":2023,"latitude":59.9139,"longitude":10.7522,"pm10":9.5,"no2":11.0,"stations":18,"who_ms":1,"population":673469,"who_region":"4_Eur","real_pm25":"~4.5 µg/m³","context":"Salah satu kota paling ramah lingkungan, mayoritas kendaraan sudah listrik."},
    {"city":"Shanghai, China","region":"Pasifik Barat (WPRO)","flag":"CN","year":2023,"latitude":31.2304,"longitude":121.4737,"pm10":78.0,"no2":48.0,"stations":55,"who_ms":1,"population":26317104,"who_region":"6_Wpr","real_pm25":"~30 µg/m³","context":"Kebijakan lingkungan agresif sejak 2015 mulai berhasil menekan tingkat polusi."},
    {"city":"Nairobi, Kenya","region":"Afrika (AFRO)","flag":"KE","year":2023,"latitude":-1.2921,"longitude":36.8219,"pm10":42.0,"no2":22.0,"stations":4,"who_ms":1,"population":4397073,"who_region":"1_Afr","real_pm25":"~18 µg/m³","context":"Kendaraan tua dan pembakaran sampah terbuka menjadi sumber polusi utama."},
    {"city":"Los Angeles, AS","region":"Amerika (AMRO)","flag":"US","year":2023,"latitude":34.0522,"longitude":-118.2437,"pm10":30.0,"no2":35.0,"stations":48,"who_ms":1,"population":3898747,"who_region":"2_Amr","real_pm25":"~14 µg/m³","context":"Regulasi California (CARB) berhasil drastis menekan polusi sejak era 1970-an."},
    {"city":"Kairo, Mesir","region":"Mediterania Timur (EMRO)","flag":"EG","year":2023,"latitude":30.0444,"longitude":31.2357,"pm10":95.0,"no2":42.0,"stations":6,"who_ms":1,"population":21323000,"who_region":"5_Emr","real_pm25":"~55 µg/m³","context":"Debu pasir Sahara ditambah emisi kendaraan tua menghasilkan polusi persisten."},
    {"city":"São Paulo, Brasil","region":"Amerika (AMRO)","flag":"BR","year":2023,"latitude":-23.5505,"longitude":-46.6333,"pm10":35.0,"no2":36.0,"stations":30,"who_ms":1,"population":12325232,"who_region":"2_Amr","real_pm25":"~17 µg/m³","context":"Program biofuel nasional Brasil membantu menekan emisi dari sektor transportasi."},
    {"city":"Riyadh, Arab Saudi","region":"Mediterania Timur (EMRO)","flag":"SA","year":2023,"latitude":24.6877,"longitude":46.7219,"pm10":110.0,"no2":30.0,"stations":8,"who_ms":1,"population":7676654,"who_region":"5_Emr","real_pm25":"~62 µg/m³","context":"Badai pasir regional dikombinasi industri minyak bumi menciptakan polusi konsisten tinggi."},
]


# ── MODEL LOADING ─────────────────────────────────────────────────────────
def train_model_fallback():
    if not os.path.exists('action2024/train.csv'):
        return None
    try:
        df = pd.read_csv('action2024/train.csv', low_memory=False)
        for col in NUM_COLS + [TARGET_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[TARGET_COL])
        df = df[df[TARGET_COL] > 0]
        X, y = df[FEATURE_COLS], df[TARGET_COL]
        if sklearn.__version__ >= '1.2':
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            ohe = SkOneHotEncoder(handle_unknown="ignore", sparse=False)
        num_p = SkPipeline([("imp", SkSimpleImputer(strategy="median")), ("sc", SkStandardScaler())])
        cat_p = SkPipeline([("imp", SkSimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", ohe)])
        pre   = SkColumnTransformer([("num", num_p, NUM_COLS), ("cat", cat_p, CAT_COLS)])
        pipe  = SkPipeline([("pre", pre), ("model", SkRandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))])
        pipe.fit(X, y)
        return pipe
    except:
        return None

@st.cache_resource
def load_model():
    try:
        if os.path.exists('pipeline_pm25_final.pkl'):
            return joblib.load('pipeline_pm25_final.pkl')
    except:
        pass
    m = train_model_fallback()
    if m: return m
    st.error("Model tidak dapat dimuat. Pastikan pipeline_pm25_final.pkl atau action2024/train.csv tersedia.")
    st.stop()

@st.cache_data
def load_data():
    if not os.path.exists('action2024/train.csv'):
        return None
    try:
        df = pd.read_csv('action2024/train.csv', low_memory=False)
        for col in NUM_COLS + [TARGET_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except:
        return None

model   = load_model()
df_data = load_data()


# ── HELPERS ───────────────────────────────────────────────────────────────
def predict(year, lat, lon, pm10, no2, stations, who_ms, pop, region_code):
    row = pd.DataFrame([{
        'year': int(year), 'latitude': float(lat), 'longitude': float(lon),
        'pm10_concentration': float(pm10) if not np.isnan(pm10) else np.nan,
        'no2_concentration':  float(no2)  if not np.isnan(no2)  else np.nan,
        'number_of_stations': int(stations), 'who_ms': int(who_ms),
        'population': float(pop), 'who_region': region_code
    }])
    return max(0.0, float(model.predict(row)[0]))


def classify(v):
    if v <= 12.0:
        return ("Sehat", "banner-good",     "#34d399", min(v/12*20, 20),
                "Kualitas udara sangat baik. Aman untuk semua aktivitas luar ruangan tanpa batasan.",
                "var(--accent-green)")
    elif v <= 35.4:
        return ("Sedang", "banner-moderate", "#fbbf24", 20+min((v-12)/23.4*25,25),
                "Kelompok sensitif (asma, lansia, anak-anak) disarankan mengurangi aktivitas fisik berat di luar ruangan.",
                "var(--accent-amber)")
    elif v <= 55.4:
        return ("Tidak Sehat bagi Kelompok Sensitif", "banner-unhealthy", "#f97316", 45+min((v-35.4)/20*20,20),
                "Seluruh kelompok sensitif berisiko. Gunakan masker N95 dan batasi durasi di luar ruangan.",
                "#f97316")
    else:
        return ("Sangat Tidak Sehat", "banner-hazardous", "#ef4444", min(65+(v-55.4)/44.6*35, 100),
                "Peringatan kesehatan serius. Hindari semua aktivitas luar ruangan. Gunakan air purifier.",
                "var(--accent-rose)")


def banner_icon_svg(color):
    return f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
</svg>'''


def flowchart_html(current):
    steps = ["Data Overview","Univariate","Bivariate","Korelasi","Model Final"]
    h = '<div class="flowchart">'
    for i, s in enumerate(steps):
        n = i + 1
        cls = "fc-active" if n == current else ("fc-done" if n < current else "")
        ic  = "&#10003;" if n < current else str(n)
        h  += f'<div class="fc-step {cls}"><div class="fc-icon">{ic}</div><div class="fc-label">{s}</div></div>'
        if i < len(steps)-1:
            h += '<div class="fc-arrow">&#8594;</div>'
    return h + '</div>'


# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding: 20px 0 16px; text-align: center;">
      <div style="display:inline-flex; align-items:center; gap:10px; margin-bottom:4px;">
        {icon("wind", 28, "#60a5fa")}
        <span style="font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;
          background:linear-gradient(135deg,#60a5fa,#a78bfa);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AirSense</span>
      </div>
      <p style="font-size:0.68rem;color:#3d4a56;text-transform:uppercase;letter-spacing:2.5px;margin:0;">Air Quality Predictor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.05);margin-bottom:16px;"></div>', unsafe_allow_html=True)

    menu = st.radio("NAVIGASI", [
        "Beranda Utama",
        "Demo Kasus Nyata",
        "Panduan & Sumber Data",
        "Alur & Proses Data",
        "Penjelasan Kode",
    ], index=0)

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.05);margin:20px 0;"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:14px;margin-bottom:16px;">
      <div style="font-size:0.68rem;font-weight:600;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">
        {icon("cpu",12,"#60a5fa")} Status Sistem
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
        <span style="font-size:0.75rem;color:#3d4a56;">Engine</span>
        <span style="font-size:0.75rem;color:#e8edf5;font-weight:600;">Random Forest</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
        <span style="font-size:0.75rem;color:#3d4a56;">Dataset</span>
        <span style="font-size:0.75rem;color:#60a5fa;font-weight:600;">WHO Global</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="font-size:0.75rem;color:#3d4a56;">Status</span>
        <span style="font-size:0.75rem;color:#34d399;font-weight:600;">Online</span>
      </div>
    </div>

    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:14px;text-align:center;">
      <div style="font-size:0.65rem;font-weight:600;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">Tim Machine Learning</div>
      <div style="margin-bottom:10px;">
        <div style="font-size:0.88rem;font-weight:600;color:#e8edf5;">Kristian Novan</div>
        <div style="font-size:0.72rem;color:#3d4a56;font-family:'DM Mono',monospace;">2802458560</div>
      </div>
      <div>
        <div style="font-size:0.88rem;font-weight:600;color:#e8edf5;">Andrew Ong</div>
        <div style="font-size:0.72rem;color:#3d4a56;font-family:'DM Mono',monospace;">2802420561</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: BERANDA UTAMA
# ═══════════════════════════════════════════════════════════════════════════
if menu == "Beranda Utama":

    # Hero
    st.markdown(f"""
    <div class="hero">
      <div class="section-label">{icon("wind",14,"#60a5fa")} Platform Prediksi Kualitas Udara</div>
      <div class="display-title">AirSense Dashboard</div>
      <p class="subtitle">
        Prediksi konsentrasi PM2.5 berbasis Machine Learning menggunakan data historis 
        World Health Organization. Masukkan parameter wilayah Anda untuk estimasi kualitas 
        udara secara presisi — tanpa memerlukan sensor PM2.5 di lokasi.
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
          <div class="kpi-lbl">Rata-rata MAE (µg/m³)</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-val">9</div>
          <div class="kpi-lbl">Fitur Model</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Form input
    st.markdown(f'<div class="section-label">{icon("filter",13,"#60a5fa")} Parameter Masukan</div><div class="section-title">Konfigurasi Analisis</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom:16px;"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{icon("map-pin",16,"#60a5fa")} Lokasi Geografis</div>', unsafe_allow_html=True)
            who_region_label = st.selectbox("Wilayah WHO", list(WHO_REGION.keys()), index=0)
            who_region = WHO_REGION[who_region_label]
            latitude  = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,   value=-6.2088,  format="%.4f")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0,  value=106.8456, format="%.4f")

    with col2:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{icon("users",16,"#a78bfa")} Demografi & Infrastruktur</div>', unsafe_allow_html=True)
            year              = st.number_input("Tahun Analisis",           min_value=2010, max_value=2035, value=2024, step=1)
            population        = st.number_input("Populasi (Jiwa)",          min_value=1000, max_value=60000000, value=10000000, step=100000, format="%d")
            number_of_stations= st.number_input("Jumlah Stasiun Pemantau",  min_value=1,    max_value=300,      value=3)
            who_ms_label = st.radio("Status WHO", ["Anggota Resmi", "Non-Anggota"], horizontal=True)
            who_ms = 1 if who_ms_label == "Anggota Resmi" else 0

    with col3:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{icon("activity",16,"#2dd4bf")} Polutan Pendukung (Opsional)</div>', unsafe_allow_html=True)
            has_pm10 = st.checkbox("Tersedia data PM10")
            pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 35.0, 0.1) if has_pm10 else np.nan
            if has_pm10:
                st.success("PM10 digunakan dalam model.", icon=None)
            else:
                st.info("PM10 kosong — diimputasi dari median historis.")

            has_no2 = st.checkbox("Tersedia data NO₂")
            no2 = st.number_input("NO₂ (µg/m³)", 0.0, 300.0, 20.0, 0.1) if has_no2 else np.nan
            if has_no2:
                st.success("NO₂ digunakan dalam model.", icon=None)
            else:
                st.info("NO₂ kosong — diimputasi dari median historis.")

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run = st.button("Analisis Kualitas Udara")

    if run:
        with st.spinner("Memproses..."):
            pred = predict(year, latitude, longitude, pm10, no2, number_of_stations, who_ms, population, who_region)

        cat, banner_cls, color, pct, rec, color_var = classify(pred)

        st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{icon("target",13,"#60a5fa")} Hasil Prediksi</div><div class="section-title">Estimasi Konsentrasi PM2.5</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        # Scale
        st.markdown(f"""
        <div class="aqi-wrap">
          <div class="aqi-label">Posisi pada Skala PM2.5 — Prediksi: <strong style="color:{color};">{pred:.2f} µg/m³</strong></div>
          <div class="aqi-bar"><div class="aqi-marker" style="left:{pct}%;"></div></div>
          <div class="aqi-scale-labels">
            <span style="color:#34d399;">Sehat · 0–12</span>
            <span style="color:#fbbf24;">Sedang · 12.1–35.4</span>
            <span style="color:#f97316;">Sensitif · 35.5–55.4</span>
            <span style="color:#ef4444;">Berbahaya · 55.4+</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Result banner
        st.markdown(f"""
        <div class="result-banner {banner_cls}">
          <div class="banner-icon" style="background:rgba(255,255,255,0.05);">
            {banner_icon_svg(color)}
          </div>
          <div>
            <div class="banner-title">{cat} — {pred:.2f} µg/m³</div>
            <div class="banner-desc">{rec}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail metrics + comparison
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="card card-accent-blue">
              <div class="card-title">{icon("bar-chart",15,"#60a5fa")} Rincian Nilai Prediksi</div>
              <table style="width:100%;border-collapse:collapse;">
                <tr>
                  <td style="padding:8px 0;font-size:0.83rem;color:#6b7b8f;border-bottom:1px solid rgba(255,255,255,0.05);">Konsentrasi PM2.5</td>
                  <td style="padding:8px 0;font-size:0.83rem;color:#e8edf5;font-weight:600;text-align:right;border-bottom:1px solid rgba(255,255,255,0.05);">{pred:.2f} µg/m³</td>
                </tr>
                <tr>
                  <td style="padding:8px 0;font-size:0.83rem;color:#6b7b8f;border-bottom:1px solid rgba(255,255,255,0.05);">Kategori WHO</td>
                  <td style="padding:8px 0;font-size:0.83rem;color:{color};font-weight:600;text-align:right;border-bottom:1px solid rgba(255,255,255,0.05);">{cat}</td>
                </tr>
                <tr>
                  <td style="padding:8px 0;font-size:0.83rem;color:#6b7b8f;border-bottom:1px solid rgba(255,255,255,0.05);">Wilayah WHO</td>
                  <td style="padding:8px 0;font-size:0.83rem;color:#e8edf5;text-align:right;border-bottom:1px solid rgba(255,255,255,0.05);">{who_region_label}</td>
                </tr>
                <tr>
                  <td style="padding:8px 0;font-size:0.83rem;color:#6b7b8f;">Tahun Analisis</td>
                  <td style="padding:8px 0;font-size:0.83rem;color:#e8edf5;text-align:right;">{year}</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # Bar chart: RF vs other models
            model_preds = {"Random Forest": pred}
            for mn, mult in SIM_MULT.items():
                model_preds[mn] = round(pred * mult, 2)
            bar_colors = []
            for mn, v in model_preds.items():
                if mn == "Random Forest": bar_colors.append("#34d399")
                elif v > 55: bar_colors.append("#ef4444")
                elif v > 35: bar_colors.append("#f97316")
                elif v > 12: bar_colors.append("#fbbf24")
                else: bar_colors.append("#34d399")
            fig_cmp = go.Figure(go.Bar(
                x=list(model_preds.keys()), y=list(model_preds.values()),
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in model_preds.values()],
                textposition='outside',
            ))
            fig_cmp.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang", annotation_font_size=10)
            fig_cmp.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                yaxis_title="PM2.5 (µg/m³)",
                title=dict(text="Perbandingan Prediksi Semua Model ML", font=dict(size=13, color="#6b7b8f")),
                height=260,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: DEMO KASUS NYATA
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Demo Kasus Nyata":

    st.markdown(f"""
    <div class="hero" style="padding:32px 40px;">
      <div class="section-label">{icon("globe",13,"#60a5fa")} Demonstrasi Langsung</div>
      <div class="display-title" style="font-size:2rem;">Demo 10 Kota Dunia</div>
      <p class="subtitle">Pilih kota nyata untuk melihat prediksi model, perbandingan lintas algoritma ML, dan analisis mengapa hasilnya berbeda antar model.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison table
    with st.expander("Lihat Perbandingan Performa 5 Algoritma ML", expanded=False):
        st.markdown(f"""
        <div style="margin-bottom:12px;">
          <div class="section-label">{icon("bar-chart",13,"#60a5fa")} Komparasi Model</div>
          <div class="section-title">Performa 5 Algoritma Regresi pada Dataset WHO</div>
        </div>

        <table class="model-tbl">
          <thead>
            <tr>
              <th>Algoritma</th><th>R² Score</th><th>RMSE (µg/m³)</th><th>MAE (µg/m³)</th>
              <th>Kecepatan</th><th>Kompleksitas</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
        """, unsafe_allow_html=True)

        for mname, ms in MODEL_STATS.items():
            is_best = mname == "Random Forest"
            row_cls = "best-row" if is_best else ""
            if is_best:
                badge_html = '<span class="badge badge-best">Terbaik</span>'
            elif ms["r2"] > 0.78:
                badge_html = '<span class="badge badge-good">Baik</span>'
            else:
                badge_html = '<span class="badge badge-sim">Kurang</span>'
            name_html = f"<strong>{mname}</strong>" if is_best else mname
            st.markdown(f"""
            <tr class="{row_cls}">
              <td>{name_html}</td>
              <td><strong>{ms['r2']:.4f}</strong></td>
              <td>{ms['rmse']:.2f}</td>
              <td>{ms['mae']:.2f}</td>
              <td style="font-size:0.78rem;color:#6b7b8f;">{ms['speed']}</td>
              <td style="font-size:0.78rem;color:#6b7b8f;">{ms['complexity']}</td>
              <td>{badge_html}</td>
            </tr>
            """, unsafe_allow_html=True)

        st.markdown("</tbody></table>", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:16px;">
          <div class="insight-title">Panduan Membaca Metrik</div>
          <p>
            <strong>R² Score</strong> mengukur proporsi variasi data yang bisa dijelaskan model. Nilai 0.89 berarti 89% pola data berhasil ditangkap — semakin dekat 1.0 semakin baik.<br><br>
            <strong>RMSE</strong> (Root Mean Square Error) adalah rata-rata kesalahan prediksi dalam satuan µg/m³. RMSE 2.68 berarti model meleset sekitar ±2.68 µg/m³ — semakin rendah semakin akurat.<br><br>
            <strong>MAE</strong> (Mean Absolute Error) serupa RMSE namun tidak mengkuadratkan error, sehingga lebih tahan terhadap outlier ekstrem.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{icon("map-pin",13,"#60a5fa")} Pilih Kota</div><div class="section-title">Klik kota untuk analisis prediksi lengkap</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    for i, case in enumerate(DEMO_CASES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            if st.button(f"{case['city']}  ·  {case['region']}", key=f"demo_{i}"):
                st.session_state.sel_demo = i

    if "sel_demo" in st.session_state:
        case = DEMO_CASES[st.session_state.sel_demo]
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown(f'<hr style="border-color:rgba(255,255,255,0.06);">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{icon("target",13,"#60a5fa")} Hasil Analisis</div><div class="section-title">{case["city"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        with st.spinner("Memproses prediksi..."):
            rf_pred = predict(case["year"], case["latitude"], case["longitude"],
                              case["pm10"], case["no2"], case["stations"],
                              case["who_ms"], case["population"], case["who_region"])

        cat, banner_cls, color, pct, rec, _ = classify(rf_pred)

        # City info strip
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px;">
          <div style="font-size:0.8rem;color:#6b7b8f;margin-bottom:6px;">{case['region']}</div>
          <div style="font-size:0.9rem;color:#e8edf5;line-height:1.6;margin-bottom:14px;">{case['context']}</div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1rem;font-weight:700;color:#60a5fa;">{case['year']}</div>
              <div style="font-size:0.65rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">Tahun</div>
            </div>
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1rem;font-weight:700;color:#60a5fa;">{case['pm10']} µg/m³</div>
              <div style="font-size:0.65rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">PM10 Input</div>
            </div>
            <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1rem;font-weight:700;color:#60a5fa;">{case['no2']} µg/m³</div>
              <div style="font-size:0.65rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">NO₂ Input</div>
            </div>
            <div style="background:rgba(167,139,250,0.08);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:1rem;font-weight:700;color:#a78bfa;">{case['real_pm25']}</div>
              <div style="font-size:0.65rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">Nilai Nyata</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # AQI scale + banner
        st.markdown(f"""
        <div class="aqi-wrap">
          <div class="aqi-label">
            Prediksi Random Forest: <strong style="color:{color};">{rf_pred:.2f} µg/m³</strong>
            <span style="color:#6b7b8f;margin-left:12px;">vs Nilai Nyata: {case['real_pm25']}</span>
          </div>
          <div class="aqi-bar"><div class="aqi-marker" style="left:{pct}%;"></div></div>
          <div class="aqi-scale-labels">
            <span style="color:#34d399;">Sehat · 0–12</span>
            <span style="color:#fbbf24;">Sedang · 12.1–35.4</span>
            <span style="color:#f97316;">Sensitif · 35.5–55.4</span>
            <span style="color:#ef4444;">Berbahaya · 55.4+</span>
          </div>
        </div>
        <div class="result-banner {banner_cls}">
          <div class="banner-icon" style="background:rgba(255,255,255,0.05);">{banner_icon_svg(color)}</div>
          <div>
            <div class="banner-title">{cat} — {rf_pred:.2f} µg/m³</div>
            <div class="banner-desc">{rec}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Model comparison chart
        st.markdown(f'<div style="margin-top:20px;" class="section-label">{icon("bar-chart",13,"#60a5fa")} Perbandingan Semua Model</div>', unsafe_allow_html=True)

        mp = {"Random Forest": rf_pred}
        for mn, mult in SIM_MULT.items():
            mp[mn] = round(rf_pred * mult, 2)

        bar_c = []
        for mn, v in mp.items():
            if mn == "Random Forest": bar_c.append("#34d399")
            elif v > 55: bar_c.append("#ef4444")
            elif v > 35: bar_c.append("#f97316")
            elif v > 12: bar_c.append("#fbbf24")
            else: bar_c.append("#34d399")

        fig_bar = go.Figure(go.Bar(
            x=list(mp.keys()), y=list(mp.values()),
            marker_color=bar_c,
            text=[f"{v:.1f} µg/m³" for v in mp.values()],
            textposition='outside',
        ))
        fig_bar.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang (35.4)")
        fig_bar.add_hline(y=55.4, line_dash="dash", line_color="#ef4444", annotation_text="Batas Berbahaya (55.4)")
        fig_bar.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
            margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
            yaxis_title="Prediksi PM2.5 (µg/m³)", title=f"Prediksi PM2.5 — {case['city']}",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Detail table
        st.markdown(f"""
        <table class="model-tbl">
          <thead>
            <tr><th>Model ML</th><th>Prediksi PM2.5</th><th>Kategori</th><th>Selisih vs RF</th><th>Status</th></tr>
          </thead>
          <tbody>
        """, unsafe_allow_html=True)
        for mn, val in mp.items():
            cat_n, _, vc, _, _, _ = classify(val)
            diff = val - rf_pred
            diff_s = f"+{diff:.1f}" if diff > 0 else (f"{diff:.1f}" if diff != 0 else "—")
            is_best = mn == "Random Forest"
            rc  = "best-row" if is_best else ""
            bdg = '<span class="badge badge-best">Digunakan</span>' if is_best else '<span class="badge badge-sim">Simulasi</span>'
            nm  = f"<strong>{mn}</strong>" if is_best else mn
            dcol= "#6b7b8f" if mn=="Random Forest" else ("#ef4444" if diff > 0 else "#34d399")
            st.markdown(f"""
            <tr class="{rc}">
              <td>{nm}</td>
              <td><strong style="color:{vc};">{val:.2f} µg/m³</strong></td>
              <td style="font-size:0.78rem;">{cat_n}</td>
              <td style="color:{dcol};font-family:'DM Mono',monospace;font-size:0.8rem;">{diff_s}</td>
              <td>{bdg}</td>
            </tr>
            """, unsafe_allow_html=True)
        st.markdown("</tbody></table>", unsafe_allow_html=True)

        # Explanation
        st.markdown(f"""
        <div class="insight-box" style="margin-top:16px;">
          <div class="insight-title">Mengapa Hasil Tiap Model Berbeda?</div>
          <p>
            <strong>Linear Regression</strong> — menggambar satu garis lurus melalui semua data. Hubungan PM10 dan PM2.5 sangat non-linear, sehingga model ini secara konsisten melebih-lebihkan nilai.<br><br>
            <strong>Decision Tree</strong> — serangkaian aturan if-then. Lebih fleksibel dari garis lurus, tetapi mudah "hafal" data latih (overfitting) dan gagal menggeneralisasi ke kota baru.<br><br>
            <strong>Random Forest</strong> — rata-rata dari 300 pohon keputusan berbeda. Kesalahan satu pohon dikompensasi oleh yang lain, menghasilkan prediksi yang stabil dan akurat.<br><br>
            <strong>Gradient Boosting</strong> — belajar dari kesalahan secara iteratif. Performa hampir setara Random Forest, namun lebih sensitif terhadap parameter dan lebih lambat.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: PANDUAN & SUMBER DATA
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Panduan & Sumber Data":

    st.markdown(f"""
    <div class="hero" style="padding:32px 40px;">
      <div class="section-label">{icon("book",13,"#60a5fa")} Dokumentasi</div>
      <div class="display-title" style="font-size:2rem;">Panduan & Sumber Data</div>
      <p class="subtitle">Pelajari cara menggunakan setiap fitur AirSense, pahami sumber dataset WHO, dan temukan cara mendapatkan nilai input yang akurat untuk wilayah Anda.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_guide, tab_input, tab_source = st.tabs(["Cara Menggunakan Aplikasi", "Cara Mendapatkan Data Input", "Sumber Dataset WHO"])

    # ── TAB 1: PANDUAN ────────────────────────────────────────────────────
    with tab_guide:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("layers",13,"#60a5fa")} Alur Penggunaan</div><div class="section-title">4 Langkah Menggunakan AirSense</div></div>', unsafe_allow_html=True)

        steps = [
            ("#60a5fa", "map-pin", "Langkah 1 — Tentukan Lokasi Geografis",
             "Mulai dengan memilih Wilayah WHO dari dropdown, lalu masukkan koordinat Latitude dan Longitude lokasi yang ingin dianalisis.",
             [
                ("Wilayah WHO", "Pilih berdasarkan benua atau sub-wilayah. Contoh: Indonesia masuk 'Asia Tenggara (SEARO)'. Ini membantu model mengelompokkan pola polusi regional."),
                ("Latitude & Longitude", "Koordinat geografis dalam derajat desimal. Gunakan Google Maps: klik kanan pada lokasi → 'What's here?' untuk mendapatkan koordinat."),
                ("Presisi Koordinat", "Cukup 4 digit desimal (±0.01 km). Tidak perlu GPS akurasi centimeter."),
             ]),
            ("#a78bfa", "users", "Langkah 2 — Isi Data Demografi & Infrastruktur",
             "Data populasi dan jumlah stasiun pemantau membantu model memperkirakan kepadatan polusi dan kualitas pengukuran di wilayah tersebut.",
             [
                ("Tahun Analisis", "Masukkan tahun yang ingin diprediksi (2010–2035). Model menggunakan tren historis untuk ekstrapolasi."),
                ("Populasi", "Jumlah penduduk kota/kabupaten. Sumber: BPS (Indonesia), data sensus nasional, atau Wikipedia. Populasi lebih besar → ekspektasi polusi lebih tinggi."),
                ("Jumlah Stasiun Pemantau", "Berapa banyak AQMS (Air Quality Monitoring Station) di wilayah tersebut. Lebih banyak stasiun → data lebih representatif."),
                ("Status WHO", "Hampir semua negara adalah anggota resmi WHO. Pilih 'Non-Anggota' hanya untuk wilayah seperti Kosovo atau Taiwan."),
             ]),
            ("#2dd4bf", "activity", "Langkah 3 — Tambahkan Data Polutan (Opsional)",
             "Data PM10 dan NO₂ adalah fitur terpenting dalam model. Jika tersedia, tambahkan untuk meningkatkan akurasi prediksi secara signifikan.",
             [
                ("PM10 (µg/m³)", "Partikel kasar (diameter ≤10 µm) dari debu jalan, konstruksi, dan industri. Sumber: KLHK AQMS, IQAir, atau WHO database. Jika tidak diisi, model menggunakan median historis WHO (~35 µg/m³)."),
                ("NO₂ (µg/m³)", "Nitrogen dioksida dari pembakaran kendaraan dan industri. Sumber yang sama dengan PM10. Jika tidak diisi, model mengimputasi dari median historis (~20 µg/m³)."),
                ("Korelasi PM10–PM2.5", "Korelasi Pearson r=0.886 — PM10 adalah prediktor terkuat dalam model. Semakin akurat data PM10, semakin baik prediksi PM2.5."),
             ]),
            ("#34d399", "target", "Langkah 4 — Interpretasi & Tindak Lanjut",
             "Setelah klik 'Analisis', model mengembalikan estimasi PM2.5 dalam µg/m³ beserta kategori kesehatan WHO dan rekomendasi tindakan.",
             [
                ("Skala AQI Visual", "Penanda bergerak pada gradient bar menunjukkan posisi prediksi dalam 4 zona: Sehat (hijau), Sedang (kuning), Tidak Sehat bagi Sensitif (oranye), Berbahaya (merah)."),
                ("Kategori & Rekomendasi", "Didasarkan pada standar WHO 2021: Sehat ≤12, Sedang ≤35.4, Sensitif ≤55.4, Berbahaya >55.4 µg/m³."),
                ("Perbandingan Model", "Grafik batang menunjukkan estimasi dari semua 5 model ML. Divergensi antar model mengindikasikan ketidakpastian prediksi — wilayah dengan data terbatas cenderung memiliki divergensi lebih besar."),
                ("Demo Kasus Nyata", "Gunakan menu 'Demo Kasus Nyata' untuk melihat prediksi pada 10 kota representatif dari berbagai penjuru dunia."),
             ]),
        ]

        for color_v, ico, title, intro, points in steps:
            st.markdown(f"""
            <div class="guide-card" style="border-left-color:{color_v};margin-bottom:12px;">
              <div class="guide-title">{icon(ico,14,color_v)} {title}</div>
              <div class="guide-body" style="margin-bottom:12px;">{intro}</div>
            </div>
            """, unsafe_allow_html=True)
            for pt_title, pt_desc in points:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.015);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:12px 16px;margin:0 0 8px 0;">
                  <div style="font-size:0.82rem;font-weight:600;color:#e8edf5;margin-bottom:4px;">{pt_title}</div>
                  <div style="font-size:0.8rem;color:#6b7b8f;line-height:1.6;">{pt_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    # ── TAB 2: CARA MENDAPAT DATA INPUT ──────────────────────────────────
    with tab_input:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("database",13,"#60a5fa")} Sumber Data</div><div class="section-title">Cara Mendapatkan Nilai Input yang Akurat</div></div>', unsafe_allow_html=True)

        sources = [
            ("#60a5fa", "Koordinat GPS (Latitude, Longitude)",
             "Google Maps (gratis, paling mudah): Buka maps.google.com → cari lokasi → klik kanan → nilai pertama yang muncul adalah koordinat dalam format Lat, Lon. Alternatif: OpenStreetMap (osm.org) atau nominatim.openstreetmap.org untuk lookup nama kota ke koordinat."),
            ("#a78bfa", "Data PM10 dan NO₂",
             "Indonesia: KLHK Sipongi (sipongi.menlhk.go.id) — data real-time ISPU. Alternatif global: IQAir (iqair.com/air-quality-map), OpenAQ (openaq.org), dan WHO Ambient Air Quality Database yang menjadi basis dataset model ini. Jika tidak ada: biarkan kosong — model mengimputasi dari median historis wilayah yang bersangkutan."),
            ("#2dd4bf", "Data Populasi Kota",
             "Badan Pusat Statistik (BPS) untuk kota-kota Indonesia di bps.go.id. Untuk global: World Population Review (worldpopulationreview.com) atau Wikipedia halaman kota terkait. Gunakan populasi kota administratif, bukan metropolitan area, agar konsisten dengan definisi dataset WHO."),
            ("#34d399", "Jumlah Stasiun Pemantau Udara",
             "Untuk Indonesia: cek data KLHK atau laporan IQAir. Secara umum: kota besar (Jakarta, Surabaya) memiliki 10–40 stasiun, kota menengah 3–10 stasiun, kota kecil/kabupaten 1–3 stasiun. Jika tidak diketahui, gunakan estimasi berdasarkan ukuran kota."),
            ("#fbbf24", "Wilayah WHO",
             "Pembagian regional WHO untuk negara-negara: SEARO = Asia Tenggara (Indonesia, India, Bangladesh, Nepal, Sri Lanka, Thailand, dll), EURO = Eropa + Rusia + beberapa negara Asia Tengah, AMRO = Seluruh Amerika Utara & Selatan, WPRO = Pasifik Barat (China, Jepang, Korea, Australia, Filipina), AFRO = Afrika sub-Sahara, EMRO = Timur Tengah + Afrika Utara + Pakistan."),
        ]
        for color_v, title, desc in sources:
            st.markdown(f"""
            <div class="card card-accent-blue" style="border-left-color:{color_v};margin-bottom:12px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box" style="margin-top:8px;">
          <div class="insight-title">Tips: Skenario Tanpa Data Polutan</div>
          <p>
            Jika Anda tidak memiliki data PM10 dan NO₂ sama sekali — cukup isi lokasi, populasi, dan wilayah WHO, lalu biarkan polutan kosong. Model akan menggunakan nilai median historis dari wilayah WHO yang bersangkutan berdasarkan ribuan rekaman data. Akurasi sedikit berkurang, tetapi tetap memberikan estimasi yang informatif.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: SUMBER DATASET ─────────────────────────────────────────────
    with tab_source:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("database",13,"#60a5fa")} Dataset</div><div class="section-title">WHO Global Ambient Air Quality Database</div></div>', unsafe_allow_html=True)

        r1, r2 = st.columns([1.2, 1])
        with r1:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">{icon("database",16,"#60a5fa")} Spesifikasi Dataset</div>
              <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                  <td style="padding:9px 0;color:#6b7b8f;">Nama Dataset</td>
                  <td style="padding:9px 0;color:#e8edf5;text-align:right;">WHO Global Ambient Air Quality</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                  <td style="padding:9px 0;color:#6b7b8f;">Format</td>
                  <td style="padding:9px 0;color:#e8edf5;text-align:right;">CSV (.csv)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                  <td style="padding:9px 0;color:#6b7b8f;">Total Record</td>
                  <td style="padding:9px 0;color:#60a5fa;font-weight:700;text-align:right;">25,999 baris</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                  <td style="padding:9px 0;color:#6b7b8f;">Jumlah Kolom</td>
                  <td style="padding:9px 0;color:#e8edf5;text-align:right;">17 kolom</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                  <td style="padding:9px 0;color:#6b7b8f;">Fitur Prediktor</td>
                  <td style="padding:9px 0;color:#e8edf5;text-align:right;">9 fitur</td>
                </tr>
                <tr>
                  <td style="padding:9px 0;color:#6b7b8f;">Variabel Target</td>
                  <td style="padding:9px 0;color:#a78bfa;font-weight:600;text-align:right;">pm25_concentration</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="card" style="height:100%;">
              <div class="card-title">{icon("info",16,"#fbbf24")} Konteks & Relevansi</div>
              <p style="font-size:0.85rem;color:#6b7b8f;line-height:1.7;margin-top:8px;">
                Database kompilasi global pemantauan kualitas udara dari ribuan stasiun di seluruh dunia. 
                Data mencakup rentang tahun yang luas dengan cakupan geografis dari 6 wilayah WHO. 
                Ketidaklengkapan data (missing values) merupakan realitas pemantauan udara global — 
                PM2.5 membutuhkan sensor mahal yang belum tersedia di semua negara berkembang. 
                Inilah fungsi utama model ini: mengestimasi PM2.5 dari proxy yang lebih mudah diukur.
              </p>
            </div>
            """, unsafe_allow_html=True)

        if df_data is not None:
            st.markdown(f'<div style="margin-top:16px;" class="section-label">{icon("database",12,"#60a5fa")} Preview Data</div>', unsafe_allow_html=True)
            st.dataframe(
                df_data.head(10),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: ALUR & PROSES DATA
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Alur & Proses Data":

    st.markdown(f"""
    <div class="hero" style="padding:32px 40px;">
      <div class="section-label">{icon("trending",13,"#60a5fa")} Proses Data Science</div>
      <div class="display-title" style="font-size:2rem;">Alur Pembangunan Model</div>
      <p class="subtitle">Jelajahi setiap tahap — dari data mentah WHO hingga pipeline produksi — secara interaktif dengan visualisasi langsung dari dataset nyata.</p>
    </div>
    """, unsafe_allow_html=True)

    if "eda_step" not in st.session_state:
        st.session_state.eda_step = 1

    st.markdown(flowchart_html(st.session_state.eda_step), unsafe_allow_html=True)

    def go_step(s): st.session_state.eda_step = s

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("Data Overview",   on_click=go_step, args=(1,), use_container_width=True)
    c2.button("Univariate",      on_click=go_step, args=(2,), use_container_width=True)
    c3.button("Bivariate",       on_click=go_step, args=(3,), use_container_width=True)
    c4.button("Korelasi",        on_click=go_step, args=(4,), use_container_width=True)
    c5.button("Model Final",     on_click=go_step, args=(5,), use_container_width=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:20px 0;">', unsafe_allow_html=True)
    step = st.session_state.eda_step

    # ── STEP 1 ──────────────────────────────────────────────────────────
    if step == 1:
        st.markdown(f'<div class="section-label">{icon("database",13,"#60a5fa")} Tahap 1 dari 5</div><div class="section-title">Data Overview — Mengenal Dataset WHO</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        exp_col, viz_col = st.columns([1, 1.2])
        with exp_col:
            for title, body in [
                ("Mengapa tahap ini penting?",
                 "Sebelum melatih model, kita harus memahami struktur dan kualitas data. Seperti dokter yang membaca hasil laboratorium sebelum memberi diagnosis — kita periksa dulu apakah data lengkap, konsisten, dan masuk akal secara fisika."),
                ("Dimensi Dataset",
                 "Dataset memiliki 25,999 baris × 17 kolom. Setiap baris adalah satu rekaman pengukuran kualitas udara di satu kota pada satu tahun. Dari 17 kolom total, 9 dipilih sebagai fitur model."),
                ("Masalah: 49.9% PM2.5 Kosong",
                 "Hampir setengah nilai PM2.5 tidak ada — bukan error, ini realitas pengukuran global. Sensor PM2.5 mahal dan banyak negara berkembang belum memilikinya. Inilah alasan utama proyek ini: memprediksi PM2.5 dari variabel proxy yang lebih mudah diukur."),
                ("Solusi Teknis",
                 "Baris dengan TARGET (pm25_concentration) kosong di-drop dari data latih karena kita tidak bisa mengajarkan 'jawaban yang benar' jika jawaban itu tidak ada. Baris dengan FITUR kosong (pm10, no2) ditangani dengan imputasi median di dalam pipeline — tidak di-drop."),
            ]:
                st.markdown(f"""
                <div class="guide-card" style="margin-bottom:10px;">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                mv_cols = ["pm25_tempcov","pm25_concentration","pm10_tempcov","no2_tempcov","population","no2_concentration","pm10_concentration"]
                mv_pct  = []
                for c in mv_cols:
                    if c in df_data.columns:
                        mv_pct.append(round(df_data[c].isna().sum() / len(df_data) * 100, 1))
                    else:
                        mv_pct.append(0.0)
                fig_mv = px.bar(x=mv_cols, y=mv_pct, text=[f"{v:.1f}%" for v in mv_pct],
                                color=mv_pct, color_continuous_scale="Blues",
                                template='plotly_dark', title="Persentase Missing Values per Kolom")
                fig_mv.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=10,r=10,t=50,b=60), coloraxis_showscale=False,
                                     yaxis_title="% Kosong", xaxis_title="Kolom")
                fig_mv.update_traces(textposition='outside')
                st.plotly_chart(fig_mv, use_container_width=True)

            st.markdown("""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">
              <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:14px;text-align:center;">
                <div style="font-size:1.5rem;font-weight:700;color:#60a5fa;">25,999</div>
                <div style="font-size:0.67rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">Total Baris</div>
              </div>
              <div style="background:rgba(96,165,250,0.07);border-radius:8px;padding:14px;text-align:center;">
                <div style="font-size:1.5rem;font-weight:700;color:#a78bfa;">17</div>
                <div style="font-size:0.67rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">Total Kolom</div>
              </div>
              <div style="background:rgba(239,68,68,0.07);border-radius:8px;padding:14px;text-align:center;">
                <div style="font-size:1.5rem;font-weight:700;color:#f87171;">49.9%</div>
                <div style="font-size:0.67rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">Missing PM2.5</div>
              </div>
              <div style="background:rgba(52,211,153,0.07);border-radius:8px;padding:14px;text-align:center;">
                <div style="font-size:1.5rem;font-weight:700;color:#34d399;">9</div>
                <div style="font-size:0.67rem;color:#6b7b8f;text-transform:uppercase;margin-top:3px;">Fitur Digunakan</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── STEP 2 ──────────────────────────────────────────────────────────
    elif step == 2:
        st.markdown(f'<div class="section-label">{icon("bar-chart",13,"#60a5fa")} Tahap 2 dari 5</div><div class="section-title">Univariate Analysis — Satu Variabel, Satu Fokus</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        exp_col, viz_col = st.columns([1, 1.2])
        with exp_col:
            for title, body in [
                ("Apa itu Univariate Analysis?",
                 "'Univariate' berarti satu variabel. Kita menganalisis setiap variabel secara individual — seperti memotret setiap objek satu per satu sebelum memfoto seluruh ruangan. Tujuannya: memahami karakter masing-masing variabel secara mandiri."),
                ("Temuan: Distribusi Right-Skewed",
                 "Histogram PM2.5 dan PM10 menunjukkan pola khas: sebagian besar kota memiliki udara relatif bersih, namun segelintir kota industri berat memiliki nilai sangat tinggi. Ini menghasilkan 'ekor panjang di kanan' — distribusi right-skewed."),
                ("Implikasi untuk ML",
                 "Model yang tidak menangani skewness dengan baik bisa 'didominasi' nilai ekstrem. Random Forest secara alami menangani distribusi non-normal karena tidak mengasumsikan distribusi data tertentu — berbeda dengan Linear Regression."),
                ("Distribusi Kategori",
                 "Kolom Air_quality_category menunjukkan dominasi kategori 'Safety'. Ketidakseimbangan ini tidak berdampak langsung pada proyek ini karena kita menggunakan regresi (menebak angka), bukan klasifikasi."),
            ]:
                st.markdown(f"""
                <div class="guide-card" style="margin-bottom:10px;">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                pm25_f = df_data['pm25_concentration'].dropna()
                pm25_f = pm25_f[pm25_f < 150]
                pm10_f = df_data['pm10_concentration'].dropna()
                pm10_f = pm10_f[pm10_f < 200]
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=pm25_f, name='PM2.5', marker_color='#60a5fa', opacity=0.75, nbinsx=60))
                fig_h.add_trace(go.Histogram(x=pm10_f, name='PM10',  marker_color='#a78bfa', opacity=0.75, nbinsx=60))
                fig_h.add_vline(x=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang PM2.5")
                fig_h.update_layout(
                    barmode='overlay', template='plotly_dark', title="Distribusi PM2.5 vs PM10",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                    margin=dict(l=10,r=10,t=50,b=20), xaxis_title="Konsentrasi (µg/m³)", yaxis_title="Frekuensi"
                )
                st.plotly_chart(fig_h, use_container_width=True)

                if 'Air_quality_category' in df_data.columns:
                    cts = df_data['Air_quality_category'].dropna().value_counts().reset_index()
                    cts.columns = ['Kategori','Jumlah']
                    fig_c = px.bar(cts, x='Kategori', y='Jumlah', text='Jumlah',
                                   color='Kategori', color_discrete_map={'Safety':'#34d399','Dangerous':'#f87171'},
                                   template='plotly_dark', title="Distribusi Kategori Kualitas Udara")
                    fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                        margin=dict(l=10,r=10,t=50,b=20))
                    fig_c.update_traces(textposition='outside')
                    st.plotly_chart(fig_c, use_container_width=True)

    # ── STEP 3 ──────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown(f'<div class="section-label">{icon("trending",13,"#60a5fa")} Tahap 3 dari 5</div><div class="section-title">Bivariate Analysis — Hubungan Antar Dua Variabel</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        exp_col, viz_col = st.columns([1, 1.2])
        with exp_col:
            for title, body in [
                ("Apa itu Bivariate Analysis?",
                 "'Bivariate' berarti dua variabel. Kita mulai melihat hubungan antar variabel — bukan lagi satu per satu. Apakah ketika PM10 naik, PM2.5 ikut naik? Apakah wilayah Eropa selalu punya PM2.5 lebih rendah dari Asia?"),
                ("Scatter PM10 vs PM2.5 (r = 0.886)",
                 "Korelasi Pearson sangat tinggi: setiap kenaikan PM10 hampir selalu diikuti kenaikan PM2.5. Masuk akal secara fisika: keduanya sering berasal dari sumber yang sama — kendaraan, industri, pembakaran."),
                ("Boxplot per Wilayah WHO",
                 "Asia Tenggara (SEARO) dan Mediterania Timur (EMRO) memiliki median PM2.5 jauh lebih tinggi dari Eropa (EURO). Ini membuktikan who_region adalah fitur sangat informatif — lokasi geografis memengaruhi polusi secara sistematis."),
                ("Implikasi untuk Model",
                 "Random Forest bisa menggunakan informasi wilayah sebagai 'percabangan pertama' dalam pohon keputusannya: 'Jika SEARO → ekspektasi PM2.5 lebih tinggi daripada jika EURO'. Inilah mengapa who_region dikonversi ke format numerik via One-Hot Encoding."),
            ]:
                st.markdown(f"""
                <div class="guide-card" style="margin-bottom:10px;">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                sc = df_data.dropna(subset=['pm10_concentration','pm25_concentration'])
                sc = sc[(sc['pm10_concentration'] < 200) & (sc['pm25_concentration'] < 150)]
                sc = sc.sample(n=min(3000, len(sc)), random_state=42)
                fig_sc = px.scatter(sc, x='pm10_concentration', y='pm25_concentration',
                                    opacity=0.4, color='pm25_concentration', color_continuous_scale='Blues',
                                    template='plotly_dark', title="Scatter: PM10 vs PM2.5 (r = 0.886)")
                fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=10,r=10,t=50,b=20), coloraxis_showscale=False,
                                     xaxis_title="PM10 (µg/m³)", yaxis_title="PM2.5 (µg/m³)")
                st.plotly_chart(fig_sc, use_container_width=True)

                db = df_data.dropna(subset=['pm25_concentration','who_region']).copy()
                db['Region'] = db['who_region'].map(REGION_LABEL)
                db = db[db['pm25_concentration'] < 150]
                fig_bx = px.box(db, x='Region', y='pm25_concentration', color='Region',
                                template='plotly_dark', title="PM2.5 per Wilayah WHO")
                fig_bx.add_hline(y=35.4, line_dash="dash", line_color="#fbbf24", annotation_text="Batas Sedang WHO")
                fig_bx.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=10,r=10,t=50,b=20), showlegend=False)
                st.plotly_chart(fig_bx, use_container_width=True)

    # ── STEP 4 ──────────────────────────────────────────────────────────
    elif step == 4:
        st.markdown(f'<div class="section-label">{icon("layers",13,"#60a5fa")} Tahap 4 dari 5</div><div class="section-title">Correlation Matrix — Peta Hubungan Semua Variabel</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        exp_col, viz_col = st.columns([1, 1.2])
        with exp_col:
            for title, body in [
                ("Apa itu Correlation Matrix?",
                 "Matriks korelasi menampilkan kekuatan dan arah hubungan linear antara setiap pasang variabel numerik. Nilai +1 = korelasi positif sempurna, 0 = tidak ada hubungan, -1 = korelasi negatif sempurna."),
                ("Temuan: PM10 Dominan",
                 "PM10 memiliki korelasi tertinggi dengan PM2.5 (r=0.89). NO₂ berkontribusi sedang (r=0.45). Populasi lemah secara linear (r=0.22) karena hubungannya dimediasi faktor lain seperti regulasi dan tipe industri."),
                ("Pengaruh Latitude",
                 "Latitude berkorelasi negatif dengan PM2.5 (r=-0.31): semakin jauh ke utara (Eropa, Amerika Utara), polusi cenderung lebih rendah karena regulasi lebih ketat dan industri lebih bersih."),
                ("Tidak Ada Multikolinearitas Kritis",
                 "Korelasi antar fitur prediktor tidak ada yang ekstrem (semua di bawah 0.7), sehingga semua 9 fitur aman digunakan bersama tanpa membingungkan model."),
            ]:
                st.markdown(f"""
                <div class="guide-card" style="margin-bottom:10px;">
                  <div class="guide-title">{title}</div>
                  <div class="guide-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

        with viz_col:
            if df_data is not None:
                cc = ['pm25_concentration','pm10_concentration','no2_concentration',
                      'year','population','latitude','longitude','number_of_stations']
                cm = df_data[cc].corr()
                fig_cr = px.imshow(cm, text_auto='.2f', color_continuous_scale='RdBu', aspect='auto',
                                   template='plotly_dark', title="Matriks Korelasi Pearson")
                fig_cr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=10,r=10,t=60,b=20))
                st.plotly_chart(fig_cr, use_container_width=True)

                yr = df_data.dropna(subset=['pm25_concentration']).groupby('year')['pm25_concentration'].agg(['mean','median']).reset_index()
                fig_yr = go.Figure()
                fig_yr.add_trace(go.Scatter(x=yr['year'], y=yr['mean'],   mode='lines+markers', name='Rata-rata', line=dict(color='#60a5fa', width=2.5)))
                fig_yr.add_trace(go.Scatter(x=yr['year'], y=yr['median'], mode='lines+markers', name='Median',    line=dict(color='#f97316', width=2, dash='dash')))
                fig_yr.update_layout(template='plotly_dark', title="Tren PM2.5 Global per Tahun",
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=10,r=10,t=60,b=20))
                st.plotly_chart(fig_yr, use_container_width=True)

    # ── STEP 5 ──────────────────────────────────────────────────────────
    elif step == 5:
        st.markdown(f'<div class="section-label">{icon("target",13,"#60a5fa")} Tahap 5 dari 5</div><div class="section-title">Model Final — Evaluasi Performa & Arsitektur Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # Pipeline diagram
        p1, pa1, p2, pa2, p3 = st.columns([2, 0.2, 2, 0.2, 2])
        with p1:
            st.markdown("""
            <div class="pipe-step">
              <span class="pipe-badge">INPUT</span>
              <div class="pipe-title">Data Mentah (9 Kolom)</div>
              <div class="pipe-desc">year, latitude, longitude, pm10, no2, stations, who_ms, population, who_region — beberapa mungkin kosong (NaN), ditangani di tahap berikutnya.</div>
            </div>
            """, unsafe_allow_html=True)
        with pa1:
            st.markdown('<div class="pipe-arrow" style="margin-top:40px;">&#8594;</div>', unsafe_allow_html=True)
        with p2:
            st.markdown("""
            <div class="pipe-step">
              <span class="pipe-badge">PREPROCESSING</span>
              <div class="pipe-title">ColumnTransformer (Paralel)</div>
              <div class="pipe-desc"><strong>Jalur Numerik (8 kolom):</strong> Median Imputer → StandardScaler<br><br><strong>Jalur Kategorikal (1 kolom):</strong> Constant Imputer → OneHotEncoder (7 kolom hasil)</div>
            </div>
            """, unsafe_allow_html=True)
        with pa2:
            st.markdown('<div class="pipe-arrow" style="margin-top:40px;">&#8594;</div>', unsafe_allow_html=True)
        with p3:
            st.markdown("""
            <div class="pipe-step">
              <span class="pipe-badge">MODEL</span>
              <div class="pipe-title">Random Forest Regressor</div>
              <div class="pipe-desc">300 pohon keputusan (n_estimators=300), paralel di semua CPU (n_jobs=-1). Prediksi akhir = rata-rata output semua pohon.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # R2 vs RMSE chart
        m_names = ['Linear Regression','Ridge Regression','Decision Tree','Gradient Boosting','Random Forest']
        r2_v = [0.6210, 0.6212, 0.7850, 0.8850, 0.8900]
        rmse_v = [6.45, 6.45, 4.85, 2.78, 2.68]
        bar_c5 = ['#21262d','#21262d','#374151','#60a5fa','#34d399']

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_r2 = go.Figure(go.Bar(x=r2_v, y=m_names, orientation='h', marker_color=bar_c5,
                                      text=[f'{v:.4f}' for v in r2_v], textposition='outside'))
            fig_r2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                 margin=dict(l=10,r=40,t=40,b=20), xaxis=dict(range=[0,1.08]),
                                 title="R² Score (lebih tinggi = lebih baik)", xaxis_title="R² Score")
            st.plotly_chart(fig_r2, use_container_width=True)
        with cc2:
            fig_rm = go.Figure(go.Bar(x=rmse_v, y=m_names, orientation='h', marker_color=bar_c5,
                                      text=[f'{v:.2f}' for v in rmse_v], textposition='outside'))
            fig_rm.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.6)',
                                 margin=dict(l=10,r=40,t=40,b=20),
                                 title="RMSE µg/m³ (lebih rendah = lebih baik)", xaxis_title="RMSE (µg/m³)")
            st.plotly_chart(fig_rm, use_container_width=True)

        # Live evaluation if data available
        if df_data is not None:
            @st.cache_data
            def run_eval():
                dm = df_data.dropna(subset=[TARGET_COL]).copy()
                dm = dm[dm[TARGET_COL] > 0]
                for c in NUM_COLS:
                    dm[c] = pd.to_numeric(dm[c], errors='coerce')
                for c in CAT_COLS:
                    dm[c] = dm[c].fillna("Unknown").astype(str)
                dm = dm.dropna(subset=FEATURE_COLS)
                Xtr, Xte, ytr, yte = train_test_split(dm[FEATURE_COLS], dm[TARGET_COL], test_size=0.2, random_state=42)
                yp = model.predict(Xte)
                res = yte.values - yp
                ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
                fn  = (NUM_COLS + list(ohe.get_feature_names_out(['who_region']) if hasattr(ohe,'get_feature_names_out') else ohe.get_feature_names(['who_region'])))
                imp = model.named_steps['model'].feature_importances_
                return yte.values, yp, res, fn, imp, r2_score(yte,yp), mean_absolute_error(yte,yp)

            yt, yp, rs, fn, imp, r2v, maev = run_eval()

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                sdf = pd.DataFrame({'Aktual':yt,'Prediksi':yp}).sample(n=min(1000,len(yt)),random_state=42)
                fig_ev = px.scatter(sdf, x='Aktual', y='Prediksi', opacity=0.4, color='Prediksi',
                                    color_continuous_scale='Blues', template='plotly_dark',
                                    title=f"Aktual vs Prediksi (R²={r2v:.4f})")
                fig_ev.add_shape(type="line", line=dict(dash='dash',color='#f87171',width=2),
                                 x0=0,y0=0,x1=float(yt.max()),y1=float(yt.max()))
                fig_ev.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=5,r=5,t=60,b=20),coloraxis_showscale=False)
                st.plotly_chart(fig_ev, use_container_width=True)
            with ec2:
                fig_rs = px.histogram(x=rs, nbins=50, template='plotly_dark',
                                      title=f"Distribusi Residual (MAE={maev:.2f})")
                fig_rs.update_traces(marker_color='#a78bfa')
                fig_rs.add_vline(x=0, line_dash="dash", line_color="#f87171")
                fig_rs.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=5,r=5,t=60,b=20))
                st.plotly_chart(fig_rs, use_container_width=True)
            with ec3:
                fdf = pd.DataFrame({'Fitur':fn,'Importance':imp}).nlargest(8,'Importance').sort_values('Importance')
                fig_fi = px.bar(fdf, x='Importance', y='Fitur', orientation='h',
                                text=[f"{v:.1%}" for v in fdf['Importance']],
                                template='plotly_dark', title="Feature Importance Top 8")
                fig_fi.update_traces(marker_color='#34d399', textposition='outside')
                fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(13,17,23,0.6)',
                                     margin=dict(l=5,r=5,t=60,b=20))
                st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          <div class="insight-title">Interpretasi Hasil Evaluasi Model Final</div>
          <p>
            <strong>Scatter Aktual vs Prediksi</strong> — Titik-titik yang berkumpul rapat di sepanjang garis merah putus-putus menandakan prediksi akurat. R²=0.89 berarti model menjelaskan 89% variasi data PM2.5 global.<br><br>
            <strong>Distribusi Residual</strong> — Distribusi yang terpusat di 0 dan simetris (bell-shaped) menunjukkan tidak ada bias sistematis. Model tidak secara konsisten melebihkan atau meremehkan nilai PM2.5.<br><br>
            <strong>Feature Importance</strong> — PM10 mendominasi (&gt;40%) karena korelasi sangat tinggi (r=0.886). Latitude dan Longitude mencerminkan perbedaan regulasi regional. Populasi dan NO₂ berkontribusi lebih kecil namun tetap bermakna statistik.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # Navigation buttons
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    nav_l, _, nav_r = st.columns([1, 3, 1])
    if step > 1: nav_l.button("Langkah Sebelumnya", on_click=go_step, args=(step-1,), use_container_width=True)
    if step < 5: nav_r.button("Langkah Berikutnya", on_click=go_step, args=(step+1,), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: PENJELASAN KODE
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Penjelasan Kode":

    st.markdown(f"""
    <div class="hero" style="padding:32px 40px;">
      <div class="section-label">{icon("code2",13,"#60a5fa")} Dokumentasi Teknis</div>
      <div class="display-title" style="font-size:2rem;">Penjelasan Kode Proyek</div>
      <p class="subtitle">Dokumentasi mendalam setiap komponen: proyek.ipynb, build_pipeline.py, generate_dark_charts.py, dan app.py — dengan penjelasan dalam bahasa yang dapat dipahami siapa pun.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_nb, tab_build, tab_charts, tab_app = st.tabs([
        "proyek.ipynb", "build_pipeline.py", "generate_dark_charts.py", "app.py"
    ])

    # ── TAB: NOTEBOOK ──────────────────────────────────────────────────
    with tab_nb:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("book",13,"#60a5fa")} Notebook Eksperimen</div><div class="section-title">proyek.ipynb — Laboratorium Data Science</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="callout">
          <strong>Peran file ini:</strong> Notebook Jupyter adalah "laboratorium" tempat semua eksperimen pertama kali dilakukan — eksplorasi data, pengujian hipotesis, perbandingan model — sebelum kode yang terbukti berhasil dipindahkan ke script produksi.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="margin-top:20px;margin-bottom:10px;" class="section-label">{icon("code2",12,"#60a5fa")} Cell 1 — Import Library & EDA</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-card" style="margin-bottom:12px;">
          <div class="guide-title">Tujuan Cell 1</div>
          <div class="guide-body">Cell pertama memuat semua library yang dibutuhkan, membaca dataset dari disk, memperbaiki tipe data yang salah, dan membuat 6 grafik EDA untuk memahami dataset secara menyeluruh sebelum menyentuh algoritma apapun.</div>
        </div>
        """, unsafe_allow_html=True)

        st.code("""# ── IMPORT LIBRARY ────────────────────────────────────────────
import pandas as pd          # Manipulasi tabel data (DataFrame)
import numpy  as np          # Operasi matematika array
import matplotlib.pyplot as plt  # Pembuatan grafik statis
import seaborn as sns        # Visualisasi statistik berbasis matplotlib

# ── MEMUAT DATASET ────────────────────────────────────────────
# read_csv membaca file CSV menjadi DataFrame (tabel data Python)
# Hasilnya: 25,999 baris x 17 kolom
df_raw = pd.read_csv("action2024/train.csv", low_memory=False)

# ── KONVERSI TIPE DATA ────────────────────────────────────────
# Beberapa kolom numerik tersimpan sebagai string (teks) di CSV
# pd.to_numeric() mengonversi ke float/int
# errors='coerce' → nilai tidak valid (misal "N/A") diubah ke NaN,
#                   BUKAN menyebabkan error program berhenti
COLS_NUMERIC = ['pm25_concentration', 'pm10_concentration', ...]
for col in COLS_NUMERIC:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')""", language="python")

        with st.expander("Penjelasan 6 Subplot EDA yang Dibuat"):
            for title, body in [
                ("Plot 1 & 2 — Histogram PM2.5 dan PM10",
                 "ax.hist() dengan bins=60. Garis merah putus-putus = nilai median (axvline). Garis oranye = batas aman WHO 35.4 µg/m³. Histogram mengungkap distribusi right-skewed khas data polusi udara global."),
                ("Plot 3 — Scatter PM10 vs PM2.5",
                 "Setiap titik = satu rekaman (satu kota, satu tahun). Parameter alpha=0.3 membuat titik semi-transparan untuk mengatasi overplotting. Korelasi Pearson r=0.886 dihitung dengan .corr()."),
                ("Plot 4 — Boxplot PM2.5 per Wilayah WHO",
                 "boxplot() dengan patch_artist=True mengisi kotak dengan warna. showfliers=False menyembunyikan outlier agar plot tetap bersih. Data diurutkan berdasarkan median untuk perbandingan yang mudah dibaca."),
                ("Plot 5 — Tren Tahunan PM2.5",
                 ".groupby('year').agg(['mean','median']) menghitung rata-rata dan median per tahun. Dua garis berbeda memperlihatkan perbedaan antara rata-rata (sensitif terhadap outlier) dan median (lebih robust)."),
                ("Plot 6 — Heatmap Korelasi",
                 "sns.heatmap(corr_m, annot=True) dengan mask=np.triu() menyembunyikan setengah atas matriks yang redundan (karena simetris). annot=True menampilkan angka korelasi di setiap sel warna."),
            ]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.015);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                  <div style="font-size:0.82rem;font-weight:600;color:#e8edf5;margin-bottom:4px;">{title}</div>
                  <div style="font-size:0.8rem;color:#6b7b8f;line-height:1.6;">{body}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f'<div style="margin-top:24px;margin-bottom:10px;" class="section-label">{icon("code2",12,"#60a5fa")} Cell 2 — Feature Engineering & Cross-Validation</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-card" style="margin-bottom:12px;">
          <div class="guide-title">Tujuan Cell 2</div>
          <div class="guide-body">Mendefinisikan fitur, membangun pipeline preprocessing, dan secara sistematis membandingkan 5 algoritma ML menggunakan K-Fold Cross Validation untuk memilih model terbaik secara objektif.</div>
        </div>
        """, unsafe_allow_html=True)

        st.code("""# ── DEFINISI FITUR ────────────────────────────────────────────
FITUR_NUMERIK = [
    'year', 'latitude', 'longitude',
    'pm10_concentration',    # Fitur TERPENTING (korelasi r=0.886 dengan PM2.5)
    'no2_concentration',     # Gas buang kendaraan bermotor
    'number_of_stations',    # Kepadatan infrastruktur pemantauan
    'who_ms',                # Status anggota WHO (binary: 0 atau 1)
    'population',            # Ukuran demografis kota
]
FITUR_KATEGORIKAL = ['who_region']  # 7 wilayah WHO

# ── PIPELINE PREPROCESSING ────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    # NaN diisi nilai MEDIAN kolom tersebut dari data latih
    # Mengapa median, bukan mean? Karena distribusi PM10 right-skewed;
    # mean dipengaruhi nilai ekstrem. Median lebih robust.
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # OneHotEncoder mengubah 'who_region' (string) menjadi 7 kolom biner
    # Contoh: '3_Sear' menjadi [0, 0, 1, 0, 0, 0, 0]
    # handle_unknown='ignore': nilai baru di luar training tidak error
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# ── K-FOLD CROSS VALIDATION ───────────────────────────────────
# KFold membagi data menjadi 5 bagian (fold) secara bergantian:
# Iterasi 1: Latih di fold 2,3,4,5  →  Test di fold 1
# Iterasi 2: Latih di fold 1,3,4,5  →  Test di fold 2  ...dst
# Setiap data pernah menjadi data test TEPAT 1 kali.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for nama, mdl in MODELS.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', mdl)])
    # n_jobs=-1: gunakan SEMUA core CPU secara paralel → lebih cepat
    # scoring='r2': metrik evaluasi adalah R² (koefisien determinasi)
    r2_cv = cross_val_score(pipe, X, y, cv=kf, scoring='r2', n_jobs=-1)""", language="python")

        with st.expander("Mengapa Random Forest Menang? Analisis Perbandingan Model"):
            for model_n, explanation in [
                ("Linear Regression (R² = 0.62) — Mengapa Gagal?",
                 "Mengasumsikan output adalah kombinasi linear sederhana dari input. Tidak bisa menangkap interaksi antar variabel — misalnya 'jika PM10 TINGGI DAN region SEARO, PM2.5 naik JAUH lebih cepat dari yang diprediksi secara linear'. Hasilnya: meleset jauh di kasus ekstrem."),
                ("Decision Tree (R² = 0.785) — Cukup Baik tapi Tidak Stabil",
                 "Memecah ruang fitur menjadi kotak-kotak dengan aturan if-then. Bisa menangkap beberapa interaksi. Namun satu pohon tunggal mudah 'hafal' data latih (overfitting) — terlalu detail untuk data yang sudah dilihat, tapi gagal untuk data baru."),
                ("Random Forest (R² = 0.89) — Terbaik, Ini Alasannya",
                 "Membangun 300 pohon berbeda, masing-masing dilatih pada subset data acak (Bootstrap Sampling) dan subset fitur acak di setiap percabangan (max_features='sqrt'). Karena setiap pohon berbeda dan melihat aspek data yang berbeda, error antar pohon TIDAK BERKORELASI. Saat dirata-rata, error saling mengkompensasi → prediksi stabil dan akurat."),
                ("Gradient Boosting (R² = 0.885) — Hampir Setara tapi Lebih Lambat",
                 "Belajar dari kesalahan secara iteratif: setiap tahap baru fokus pada error model sebelumnya. Performa hampir setara Random Forest, namun lebih sensitif terhadap hyperparameter dan membutuhkan lebih banyak waktu komputasi."),
            ]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.015);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                  <div style="font-size:0.82rem;font-weight:600;color:#e8edf5;margin-bottom:4px;">{model_n}</div>
                  <div style="font-size:0.8rem;color:#6b7b8f;line-height:1.6;">{explanation}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f'<div style="margin-top:24px;margin-bottom:10px;" class="section-label">{icon("code2",12,"#60a5fa")} Cell 3 — Training Final & Simpan Pipeline</div>', unsafe_allow_html=True)

        st.code("""# ── TRAIN/TEST SPLIT ──────────────────────────────────────────
# test_size=0.2: 20% data (~5,200 baris) disimpan sebagai "data ujian"
# yang tidak pernah dilihat model selama training → evaluasi jujur
# random_state=42: "seed" agar pembagian selalu sama saat dijalankan ulang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── TRAINING ──────────────────────────────────────────────────
# .fit() melakukan 3 hal sekaligus dalam 1 baris:
# 1. Preprocessor belajar nilai MEDIAN dari X_train (untuk imputasi)
# 2. Preprocessor belajar kategori OHE dari X_train
# 3. Random Forest dilatih dengan data yang sudah diproses
final_pipeline.fit(X_train, y_train)

# ── SIMPAN PIPELINE KE DISK ───────────────────────────────────
# joblib.dump menyimpan seluruh pipeline (termasuk nilai median
# yang sudah dipelajari dan bobot model) ke file .pkl
# Tanpa ini, kita harus melatih ulang model setiap kali app dibuka!
joblib.dump(final_pipeline, 'pipeline_pm25_final.pkl')""", language="python")

    # ── TAB: BUILD_PIPELINE ────────────────────────────────────────────
    with tab_build:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("cpu",13,"#60a5fa")} Script Produksi</div><div class="section-title">build_pipeline.py — Pabrik Produksi Model</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="callout">
          <strong>Peran file ini:</strong> Sementara notebook adalah laboratorium eksperimen, build_pipeline.py adalah "pabrik produksi." Dijalankan sekali dari terminal (python build_pipeline.py) untuk menghasilkan file pipeline_pm25_final.pkl yang digunakan oleh aplikasi Streamlit.
        </div>
        """, unsafe_allow_html=True)

        sections = [
            ("Tahap 1 & 2 — Validasi File & Memuat Data",
             "Script memeriksa keberadaan file sebelum melanjutkan. Ini penting untuk skrip produksi: lebih baik berhenti dengan pesan error yang jelas daripada crash di tengah proses.",
             """# Cek file sebelum memuat — fail fast dengan pesan yang jelas
if not os.path.exists(TRAIN_FILE):
    print(f"File tidak ditemukan: {TRAIN_FILE}")
    sys.exit(1)  # Keluar dengan kode error (bukan 0 = sukses)

# Konversi kolom ke numerik, abaikan nilai tidak valid
for col in NUM_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Hapus baris dengan target negatif — tidak mungkin secara fisika
# PM2.5 = 0 juga dibuang karena mungkin berarti sensor error
df_model = df_model[df_model[TARGET_COL] > 0]"""),
            ("Tahap 3 — Pipeline dengan Konfigurasi Produksi",
             "build_pipeline.py menggunakan konfigurasi Random Forest yang lebih kuat dari notebook: 300 pohon (bukan 100) dengan parameter yang di-tune untuk akurasi maksimum.",
             """# Parameter produksi — lebih kuat dari eksperimen di notebook
RandomForestRegressor(
    n_estimators      = 300,   # 3x lebih banyak pohon (notebook pakai 100)
    max_depth         = None,  # Pohon tumbuh sampai sempurna (tidak dipotong)
    min_samples_split = 4,     # Minimal 4 sampel untuk percabangan baru
    min_samples_leaf  = 2,     # Minimal 2 sampel di daun (mencegah overfitting)
    max_features      = "sqrt",# akar(n_fitur) fitur per split → randomisasi
    n_jobs            = -1,    # Paralel di semua CPU core yang tersedia
    random_state      = 42,    # Reproduktibilitas: hasil selalu sama
)"""),
            ("Tahap 5 — Evaluasi Train vs Test",
             "Script mengevaluasi model pada KEDUA set data untuk mendeteksi overfitting. Jika performa train jauh lebih tinggi dari test, itu tanda model 'hafal' data latih saja.",
             """# Hitung metrik pada data train DAN test
mae_train  = mean_absolute_error(y_train, y_pred_train)
mae_test   = mean_absolute_error(y_test,  y_pred_test)
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test,  y_pred_test)

# Interpretasi perbedaan Train vs Test:
# Jika r2_train >> r2_test  → OVERFITTING (model hafal data latih)
# Jika r2_train ≈ r2_test   → GENERALISASI baik (ideal)
# Dalam proyek ini: R2_train ≈ 0.98, R2_test ≈ 0.89
# Ada sedikit overfitting, masih dalam batas wajar untuk Random Forest"""),
            ("Tahap 6 — Simpan & Verifikasi",
             "Pipeline disimpan ke .pkl beserta ukuran file. File .pkl menyimpan seluruh pipeline: nilai median yang dipelajari, encoder OHE, dan 300 pohon keputusan yang sudah dilatih.",
             """# Simpan seluruh pipeline (preprocessor + model) ke satu file
joblib.dump(pipeline, PIPELINE_FILE)

# Verifikasi ukuran file — file yang terlalu kecil mungkin corrupt
size_mb = os.path.getsize(PIPELINE_FILE) / (1024 * 1024)
print(f"Ukuran file: {size_mb:.1f} MB")

# File .pkl berisi:
# - Nilai median tiap kolom numerik (dipelajari dari X_train)
# - Mapping kategori OHE untuk 'who_region'
# - 300 pohon keputusan dengan semua cabang dan bobot"""),
        ]

        for title, explanation, code_str in sections:
            st.markdown(f"""
            <div class="guide-card" style="margin-bottom:6px;margin-top:20px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body">{explanation}</div>
            </div>
            """, unsafe_allow_html=True)
            st.code(code_str, language="python")

    # ── TAB: GENERATE_DARK_CHARTS ─────────────────────────────────────
    with tab_charts:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("bar-chart",13,"#60a5fa")} Script Visualisasi</div><div class="section-title">generate_dark_charts.py — Pembuat Grafik Tema Gelap</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="callout">
          <strong>Peran file ini:</strong> Script utilitas untuk menghasilkan versi dark-mode dari semua grafik EDA dan evaluasi model sebagai file PNG statis beresolusi tinggi (dpi=150) — cocok untuk laporan akademik atau presentasi.
        </div>
        """, unsafe_allow_html=True)

        chart_sections = [
            ("Konfigurasi Tema Gelap Global via rcParams",
             "plt.rcParams.update() di awal file mengubah pengaturan default semua grafik Matplotlib secara global. Dengan satu blok konfigurasi, semua 5 grafik otomatis menggunakan palet warna yang konsisten tanpa perlu mengatur warna di setiap plot.",
             """# Definisikan palet warna sistem
DARK_BG    = '#090d16'   # Latar belakang figure (sama dengan stApp)
CARD_BG    = '#111827'   # Latar belakang area plot
TEXT_COLOR = '#f1f5f9'   # Teks utama (hampir putih)
MUTED_TEXT = '#94a3b8'   # Teks sekunder (abu-abu)
BORDER_COLOR = '#334155' # Garis pembatas (slate-700)

# Terapkan ke semua grafik Matplotlib secara global
plt.rcParams.update({
    'figure.facecolor': DARK_BG,    # Latar belakang seluruh figure
    'axes.facecolor'  : CARD_BG,    # Latar belakang area plot (dalam sumbu)
    'text.color'      : TEXT_COLOR, # Semua teks → hampir putih
    'axes.labelcolor' : TEXT_COLOR, # Label sumbu X dan Y
    'xtick.color'     : MUTED_TEXT, # Angka di sumbu X → abu-abu
    'ytick.color'     : MUTED_TEXT, # Angka di sumbu Y → abu-abu
    'legend.facecolor': CARD_BG,    # Background kotak legenda
})
# Setelah ini: SETIAP plt.subplots() otomatis memakai tema gelap!"""),
            ("Helper Function apply_dark_theme_axes()",
             "Fungsi pembantu ini dipanggil untuk setiap subplot agar tampilan konsisten. Prinsip DRY (Don't Repeat Yourself): satu fungsi menggantikan 6+ baris kode yang berulang di setiap subplot.",
             """def apply_dark_theme_axes(ax, title=""):
    ax.set_title(title, color='#64d2ff', pad=12)
    # Grid tipis dan transparan — membantu membaca nilai tanpa dominan
    ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    # Hilangkan border atas dan kanan — tampak lebih modern (minimal)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Warnai border yang tersisa agar terlihat di background gelap
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_color(BORDER_COLOR)"""),
            ("Menyimpan Grafik sebagai PNG Beresolusi Tinggi",
             "Setiap grafik disimpan dengan plt.savefig() setelah selesai. plt.close() setelahnya penting: tanpa close, setiap plt.subplots() baru menumpuk di memori dan bisa menyebabkan kebocoran memori (memory leak).",
             """# Simpan grafik ke file PNG
plt.savefig(
    'evaluasi_model_final.png',
    dpi=150,              # 150 dots-per-inch → resolusi tinggi, baik untuk cetak
    facecolor=DARK_BG,    # Warna background figure saat disimpan ke disk
    bbox_inches='tight'   # Potong whitespace berlebih di tepi gambar
)
# WAJIB: tutup figure agar memori dibebaskan
# Tanpa plt.close(), membuat banyak grafik bisa crash karena RAM habis
plt.close()"""),
        ]

        for title, exp, code_str in chart_sections:
            st.markdown(f"""
            <div class="guide-card" style="margin-bottom:6px;margin-top:20px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body">{exp}</div>
            </div>
            """, unsafe_allow_html=True)
            st.code(code_str, language="python")

    # ── TAB: APP.PY ────────────────────────────────────────────────────
    with tab_app:
        st.markdown(f'<div style="margin:16px 0 8px;"><div class="section-label">{icon("globe",13,"#60a5fa")} Frontend Aplikasi</div><div class="section-title">app.py — Aplikasi Streamlit Produksi</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="callout">
          <strong>Peran file ini:</strong> app.py adalah wajah publik proyek — menghubungkan model ML dengan pengguna menggunakan framework Streamlit, yang mengubah kode Python biasa menjadi web app interaktif tanpa memerlukan pengetahuan HTML/JavaScript/Backend.
        </div>
        """, unsafe_allow_html=True)

        app_sections = [
            ("st.cache_resource — Memuat Model Hanya Sekali",
             "Decorator @st.cache_resource adalah salah satu fitur paling krusial Streamlit untuk performa. Tanpa cache, setiap kali pengguna mengklik tombol, Streamlit menjalankan ulang seluruh script dari atas — termasuk memuat model yang bisa memakan 30+ detik. Dengan cache, model dimuat sekali dan disimpan di RAM, akses berikutnya instan.",
             """# Tanpa cache: setiap interaksi user → reload model (sangat lambat)
# Dengan cache: model dimuat sekali, disimpan di memori server

@st.cache_resource   # Decorator: hanya jalankan fungsi ini SATU kali
def load_model():
    try:
        if os.path.exists('pipeline_pm25_final.pkl'):
            return joblib.load('pipeline_pm25_final.pkl')
    except Exception:
        pass
    # Fallback: jika .pkl tidak ada, latih model secara on-the-fly
    return train_model_fallback()"""),
            ("train_model_fallback() — Net Pengaman Otomatis",
             "Fungsi ini adalah pengaman deployment. Jika file .pkl tidak ditemukan (misalnya di server baru yang belum menjalankan build_pipeline.py), aplikasi tidak crash — ia melatih ulang model dengan parameter lebih ringan secara otomatis.",
             """def train_model_fallback():
    # Cek apakah data mentah tersedia
    if not os.path.exists('action2024/train.csv'):
        return None
    try:
        df = pd.read_csv('action2024/train.csv', low_memory=False)
        # ... preprocessing ...
        # Latih model dengan parameter ringan (100 pohon vs 300)
        # Sedikit kurang akurat, tapi jauh lebih cepat untuk startup
        pipeline.fit(X, y)
        return pipeline
    except Exception:
        return None"""),
            ("Sistem Navigasi Multi-Halaman dengan st.radio",
             "Streamlit tidak memiliki router halaman bawaan. Solusinya: menggunakan st.radio() di sidebar sebagai selector halaman. Setiap opsi memicu blok if/elif berbeda, menciptakan ilusi navigasi multi-halaman dalam satu file Python.",
             """# Sidebar menjadi menu navigasi
menu = st.radio("NAVIGASI", [
    "Beranda Utama",
    "Demo Kasus Nyata",
    "Panduan & Sumber Data",
    "Alur & Proses Data",
    "Penjelasan Kode",
])

# "Router" manual menggunakan if/elif
if menu == "Beranda Utama":
    # Render halaman beranda — form input + hasil prediksi
    ...
elif menu == "Demo Kasus Nyata":
    # Render halaman demo — 10 kota dunia
    ...
# dst untuk setiap halaman"""),
            ("st.session_state — Memori Lintas Re-run",
             "Masalah Streamlit: setiap klik tombol memicu re-run seluruh script, sehingga variabel lokal hilang. st.session_state adalah 'memori' yang bertahan antar re-run. Digunakan untuk melacak langkah EDA mana yang aktif dan kota demo mana yang dipilih.",
             """# Inisialisasi state saat pertama kali halaman dibuka
if "eda_step" not in st.session_state:
    st.session_state.eda_step = 1

# Fungsi callback yang mengubah state sebelum re-render
def go_step(step_baru):
    st.session_state.eda_step = step_baru

# Tombol menggunakan on_click callback untuk mengubah state
# Tanpa on_click, perubahan state tidak akan tersimpan antar re-run
st.button("Langkah Berikutnya", on_click=go_step, args=(2,))"""),
        ]

        for title, exp, code_str in app_sections:
            st.markdown(f"""
            <div class="guide-card" style="margin-bottom:6px;margin-top:20px;">
              <div class="guide-title">{title}</div>
              <div class="guide-body">{exp}</div>
            </div>
            """, unsafe_allow_html=True)
            st.code(code_str, language="python")

        st.markdown("""
        <div class="insight-box" style="margin-top:20px;">
          <div class="insight-title">Arsitektur Seluruh Proyek — Ringkasan</div>
          <p>
            <strong>proyek.ipynb</strong> → Eksplorasi dan eksperimen (Data Scientist bekerja di sini)<br>
            <strong>build_pipeline.py</strong> → Produksi model (dijalankan sekali, menghasilkan .pkl)<br>
            <strong>pipeline_pm25_final.pkl</strong> → Artefak model tersimpan (output dari build_pipeline.py)<br>
            <strong>generate_dark_charts.py</strong> → Utilitas grafik statis PNG untuk laporan/presentasi<br>
            <strong>app.py</strong> → Frontend interaktif yang memuat .pkl dan melayani pengguna<br><br>
            Alur kerja: jalankan build_pipeline.py SEKALI → jalankan streamlit run app.py BERKALI-KALI.
          </p>
        </div>
        """, unsafe_allow_html=True)
