
# ===================================================================
# BUILD_PIPELINE.PY — Script Pembuat Pipeline PM2.5
# Pembuat: Kristian Novan
# Mata Kuliah: COMP6577001 — Machine Learning
#
# Jalankan script ini SEKALI untuk membuat file pipeline_pm25_final.pkl
# yang digunakan oleh aplikasi Streamlit (app.py).
#
# Cara penggunaan:
#   python build_pipeline.py
# ===================================================================

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# 0. Konfigurasi & Konstanta
# ─────────────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "action2024")
TRAIN_FILE       = os.path.join(DATA_DIR, "train.csv")
PIPELINE_FILE    = "pipeline_pm25_final.pkl"
RANDOM_STATE     = 42

# Fitur yang digunakan (harus sama persis dengan app.py)
FEATURE_COLS = [
    "year",
    "latitude",
    "longitude",
    "pm10_concentration",
    "no2_concentration",
    "number_of_stations",
    "who_ms",
    "population",
    "who_region",
]

TARGET_COL = "pm25_concentration"

# Kolom numerik & kategorikal
NUM_COLS = [
    "year",
    "latitude",
    "longitude",
    "pm10_concentration",
    "no2_concentration",
    "number_of_stations",
    "who_ms",
    "population",
]
CAT_COLS = ["who_region"]


def print_header(title: str) -> None:
    """Cetak header yang rapi ke terminal."""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def print_step(step: str) -> None:
    print(f"  ➤  {step}")


# ─────────────────────────────────────────────────────────────────
# 1. Memuat & Pra-pemrosesan Dataset
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 1 — Memuat Dataset")

if not os.path.exists(TRAIN_FILE):
    print(f"  ❌  File tidak ditemukan: {TRAIN_FILE}")
    print("  Pastikan folder 'action2024/' berisi file train.csv.")
    sys.exit(1)

print_step(f"Membaca {TRAIN_FILE} …")
t0 = time.time()
df = pd.read_csv(TRAIN_FILE, low_memory=False)
print_step(f"Dataset dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom  ({time.time()-t0:.1f}s)")

# Tampilkan kolom yang tersedia
print_step(f"Kolom tersedia: {list(df.columns)}")

# Konversi kolom numerik yang mungkin dibaca sebagai object/string
for col in NUM_COLS + [TARGET_COL]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# Pastikan semua kolom yang dibutuhkan ada
missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing_cols:
    print(f"\n  ❌  Kolom tidak ditemukan dalam CSV: {missing_cols}")
    print("  Periksa kembali nama kolom di file train.csv.")
    sys.exit(1)

# Ambil hanya baris dengan target yang valid
df_model = df.dropna(subset=[TARGET_COL]).copy()
print_step(f"Baris dengan target valid (pm25_concentration): {df_model.shape[0]:,}")

# Hapus baris target bernilai negatif (tidak fisis)
df_model = df_model[df_model[TARGET_COL] > 0]
print_step(f"Setelah membuang target ≤ 0: {df_model.shape[0]:,} baris")

# Tampilkan missing values pada fitur
print_step("Missing values pada fitur yang digunakan:")
for col in FEATURE_COLS:
    n_miss = df_model[col].isna().sum()
    pct    = 100.0 * n_miss / len(df_model)
    if n_miss > 0:
        print(f"       {col}: {n_miss:,} ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────
# 2. Split Data
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 2 — Split Data Train / Test")

X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print_step(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")


# ─────────────────────────────────────────────────────────────────
# 3. Membangun Pipeline
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 3 — Membangun Pipeline ML")

# Sub-pipeline untuk kolom numerik: imputasi median → standarisasi
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Sub-pipeline untuk kolom kategorikal: imputasi konstanta → OHE
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

# ColumnTransformer menggabungkan keduanya
preprocessor = ColumnTransformer([
    ("num", num_pipeline, NUM_COLS),
    ("cat", cat_pipeline, CAT_COLS),
])

# Pipeline akhir: pra-pemrosesan + Random Forest
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators  = 300,
        max_depth     = None,
        min_samples_split = 4,
        min_samples_leaf  = 2,
        max_features  = "sqrt",
        n_jobs        = -1,
        random_state  = RANDOM_STATE,
    )),
])

print_step("Pipeline berhasil dibuat:")
print_step("  • Numerik: MedianImputer → StandardScaler")
print_step("  • Kategorikal: ConstantImputer → OneHotEncoder")
print_step("  • Regressor: RandomForestRegressor (n_estimators=300)")


# ─────────────────────────────────────────────────────────────────
# 4. Melatih Model
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 4 — Melatih Model")

print_step("Melatih RandomForestRegressor … (ini mungkin memakan waktu beberapa menit)")
t0 = time.time()
pipeline.fit(X_train, y_train)
elapsed = time.time() - t0
print_step(f"Pelatihan selesai dalam {elapsed:.1f} detik")


# ─────────────────────────────────────────────────────────────────
# 5. Evaluasi Model
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 5 — Evaluasi Model")

y_pred_train = pipeline.predict(X_train)
y_pred_test  = pipeline.predict(X_test)

mae_train  = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train   = r2_score(y_train, y_pred_train)

mae_test   = mean_absolute_error(y_test, y_pred_test)
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test    = r2_score(y_test, y_pred_test)

print(f"  {'Metrik':<30} {'Train':>12} {'Test':>12}")
print(f"  {'-'*54}")
print(f"  {'MAE (µg/m³)':<30} {mae_train:>12.4f} {mae_test:>12.4f}")
print(f"  {'RMSE (µg/m³)':<30} {rmse_train:>12.4f} {rmse_test:>12.4f}")
print(f"  {'R² Score':<30} {r2_train:>12.4f} {r2_test:>12.4f}")


# ─────────────────────────────────────────────────────────────────
# 6. Simpan Pipeline
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 6 — Menyimpan Pipeline")

print_step(f"Menyimpan ke: {PIPELINE_FILE}")
joblib.dump(pipeline, PIPELINE_FILE)
size_mb = os.path.getsize(PIPELINE_FILE) / (1024 * 1024)
print_step(f"Ukuran file: {size_mb:.1f} MB")
print_step("Pipeline berhasil disimpan! ✅")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              PIPELINE BERHASIL DIBUAT!                      ║
║                                                              ║
║  File   : {PIPELINE_FILE:<46} ║
║  MAE    : {mae_test:<.4f} µg/m³{' '*37} ║
║  RMSE   : {rmse_test:<.4f} µg/m³{' '*37} ║
║  R²     : {r2_test:<.4f}{' '*45} ║
║                                                              ║
║  Sekarang jalankan:  streamlit run app.py                   ║
╚══════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────
# 7. Membuat Visualisasi Evaluasi (Opsional)
# ─────────────────────────────────────────────────────────────────
print_header("TAHAP 7 — Membuat Visualisasi")

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0e1a')

    ax_scatter, ax_resid = axes

    for ax in axes:
        ax.set_facecolor('#0d1b2a')
        ax.spines['bottom'].set_color('#64d2ff')
        ax.spines['top'].set_color('#0d1b2a')
        ax.spines['left'].set_color('#64d2ff')
        ax.spines['right'].set_color('#0d1b2a')
        ax.tick_params(colors='#a0c8ff', labelsize=9)
        ax.xaxis.label.set_color('#a0c8ff')
        ax.yaxis.label.set_color('#a0c8ff')
        ax.title.set_color('#64d2ff')

    # Scatter: Aktual vs Prediksi
    ax_scatter.scatter(y_test, y_pred_test,
                       alpha=0.4, s=12, c='#64d2ff', edgecolors='none')
    lims = [0, max(y_test.max(), y_pred_test.max()) * 1.05]
    ax_scatter.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.7, label='Ideal')
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel('PM2.5 Aktual (µg/m³)')
    ax_scatter.set_ylabel('PM2.5 Prediksi (µg/m³)')
    ax_scatter.set_title(f'Aktual vs. Prediksi\nR² = {r2_test:.4f}')
    ax_scatter.legend(labelcolor='#a0c8ff', facecolor='#0d1b2a',
                      edgecolor='#64d2ff', fontsize=9)

    # Histogram Residual
    residuals = y_test.values - y_pred_test
    ax_resid.hist(residuals, bins=60, color='#bf5af2', alpha=0.7, edgecolor='none')
    ax_resid.axvline(0, color='#f87171', linewidth=1.8, linestyle='--', label='Residu = 0')
    ax_resid.set_xlabel('Residu (Aktual − Prediksi)')
    ax_resid.set_ylabel('Frekuensi')
    ax_resid.set_title(f'Distribusi Residual\nMAE = {mae_test:.4f}  |  RMSE = {rmse_test:.4f}')
    ax_resid.legend(labelcolor='#a0c8ff', facecolor='#0d1b2a',
                    edgecolor='#64d2ff', fontsize=9)

    plt.tight_layout(pad=3.0)
    out_file = "evaluasi_model_final.png"
    plt.savefig(out_file, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print_step(f"Visualisasi disimpan: {out_file} ✅")

except Exception as exc:
    print_step(f"Visualisasi gagal (tidak wajib): {exc}")
