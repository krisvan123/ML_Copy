# ===================================================================
# GENERATE_DARK_CHARTS.PY — Script Pembuat Visualisasi Tema Gelap (Windows safe)
# ===================================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def main():
    # ── Konfigurasi Desain Matplotlib ───────────────────────────────────
    DARK_BG = '#090d16'      # Latar belakang luar (stApp gradient dark blue)
    CARD_BG = '#111827'      # Latar belakang plot area (gray-900)
    TEXT_COLOR = '#f1f5f9'   # Teks utama (slate-100)
    MUTED_TEXT = '#94a3b8'   # Teks sekunder (slate-400)
    BORDER_COLOR = '#334155' # Garis pembatas/spines (slate-700)
    GRID_COLOR = '#1e293b'   # Garis kisi (slate-800)

    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': CARD_BG,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': MUTED_TEXT,
        'ytick.color': MUTED_TEXT,
        'axes.edgecolor': BORDER_COLOR,
        'grid.color': GRID_COLOR,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'legend.facecolor': CARD_BG,
        'legend.edgecolor': BORDER_COLOR,
        'figure.autolayout': False
    })

    def apply_dark_theme_axes(ax, title=""):
        ax.set_title(title, color='#64d2ff', pad=12)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(BORDER_COLOR)
        ax.spines['bottom'].set_color(BORDER_COLOR)

    # ── Memuat & Menyiapkan Data ───────────────────────────────────────
    print("[INFO] Memuat dataset...")
    TRAIN_FILE = "action2024/train.csv"
    if not os.path.exists(TRAIN_FILE):
        print(f"[ERROR] File tidak ditemukan: {TRAIN_FILE}")
        sys.exit(1)

    df_raw = pd.read_csv(TRAIN_FILE, low_memory=False)

    # Konversi kolom numerik
    COLS_NUMERIC = [
        'pm25_concentration', 'pm10_concentration', 'no2_concentration',
        'pm10_tempcov', 'pm25_tempcov', 'no2_tempcov', 'population'
    ]
    for col in COLS_NUMERIC:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')


    # ===================================================================
    # CHART 1: missing_values_chart.png
    # ===================================================================
    print("[INFO] Membuat missing_values_chart.png...")
    data_missing = {
        "Feature Name": [
            "pm25_tempcov", "pm25_concentration", "pm10_tempcov", 
            "no2_tempcov", "population", "no2_concentration", "pm10_concentration"
        ],
        "Percentage (%)": [63.0, 49.9, 45.8, 43.4, 39.7, 33.9, 26.9]
    }
    missing_df = pd.DataFrame(data_missing)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    bars = sns.barplot(x="Feature Name", y="Percentage (%)", data=missing_df, palette="coolwarm", edgecolor=BORDER_COLOR, ax=ax)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color='#a180ff')

    apply_dark_theme_axes(ax, "Persentase Data Kosong per Fitur (Missing Values)")
    ax.set_ylabel('Persentase (%)')
    ax.set_xlabel('Nama Fitur')
    plt.xticks(rotation=30, ha='right')
    ax.set_ylim(0, 75)
    plt.tight_layout()
    plt.savefig('missing_values_chart.png', dpi=150, facecolor=DARK_BG)
    plt.close()


    # ===================================================================
    # CHART 2: class_distribution.png
    # ===================================================================
    print("[INFO] Membuat class_distribution.png...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    categories = df_raw['Air_quality_category'].dropna().value_counts()
    colors_cd = ['#10b981', '#f43f5e'] # Green vs Rose-pink
    bars = ax.bar(categories.index, categories.values, color=colors_cd, edgecolor=BORDER_COLOR, width=0.6)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontweight='bold', 
                    fontsize=10,
                    color='#ffffff')

    apply_dark_theme_axes(ax, "Distribusi Kategori Kualitas Udara")
    ax.set_xlabel('Kategori (Sehat vs Bahaya)')
    ax.set_ylabel('Jumlah Data')
    ax.set_ylim(0, max(categories.values) * 1.15)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, facecolor=DARK_BG)
    plt.close()


    # ===================================================================
    # CHART 3: eda_visualisasi.png
    # ===================================================================
    print("[INFO] Membuat eda_visualisasi.png (6 Subplot)...")
    REGION_LABEL = {
        '1_Afr': 'Afrika', '2_Amr': 'Amerika', '3_Sear': 'Asia Tenggara',
        '4_Eur': 'Eropa',  '5_Emr': 'Mediterania', '6_Wpr': 'Pasifik Barat',
        '7_NonMS': 'Non-MS'
    }
    df_raw['region_label'] = df_raw['who_region'].map(REGION_LABEL)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Plot 1: Distribusi PM2.5
    ax1 = fig.add_subplot(gs[0, 0])
    pm25_v = df_raw['pm25_concentration'].dropna()
    ax1.hist(pm25_v[pm25_v < 150], bins=50, color='#64d2ff', edgecolor=CARD_BG, alpha=0.9)
    ax1.axvline(pm25_v.median(), color='#f43f5e', ls='--', lw=1.8, label=f'Median: {pm25_v.median():.1f}')
    ax1.axvline(35.4, color='#f59e0b', ls=':', lw=2, label='Batas Aman (35.4)')
    apply_dark_theme_axes(ax1, 'Distribusi PM2.5')
    ax1.set_xlabel('PM2.5 (µg/m³)'); ax1.set_ylabel('Frekuensi')
    ax1.legend(fontsize=8, loc='upper right')

    # Plot 2: Distribusi PM10
    ax2 = fig.add_subplot(gs[0, 1])
    pm10_v = df_raw['pm10_concentration'].dropna()
    ax2.hist(pm10_v[pm10_v < 200], bins=50, color='#a180ff', edgecolor=CARD_BG, alpha=0.9)
    ax2.axvline(pm10_v.median(), color='#f43f5e', ls='--', lw=1.8, label=f'Median: {pm10_v.median():.1f}')
    apply_dark_theme_axes(ax2, 'Distribusi PM10')
    ax2.set_xlabel('PM10 (µg/m³)'); ax2.set_ylabel('Frekuensi')
    ax2.legend(fontsize=8, loc='upper right')

    # Plot 3: Scatter PM10 vs PM2.5
    ax3 = fig.add_subplot(gs[0, 2])
    sc = df_raw.dropna(subset=['pm10_concentration', 'pm25_concentration'])
    sc = sc[(sc['pm10_concentration'] < 200) & (sc['pm25_concentration'] < 150)]
    corr_val = sc['pm10_concentration'].corr(sc['pm25_concentration'])
    ax3.scatter(sc['pm10_concentration'], sc['pm25_concentration'],
                alpha=0.25, s=4, color='#38bdf8', edgecolors='none')
    apply_dark_theme_axes(ax3, f'PM10 vs PM2.5\n(Korelasi: {corr_val:.3f})')
    ax3.set_xlabel('PM10 (µg/m³)'); ax3.set_ylabel('PM2.5 (µg/m³)')

    # Plot 4: Boxplot PM2.5 per Wilayah WHO
    ax4 = fig.add_subplot(gs[1, :2])
    plot_df = df_raw.dropna(subset=['pm25_concentration', 'region_label'])
    plot_df = plot_df[plot_df['pm25_concentration'] < 150]
    order = plot_df.groupby('region_label')['pm25_concentration'].median().sort_values().index.tolist()
    bp_data = [plot_df[plot_df['region_label'] == r]['pm25_concentration'].values for r in order]

    bp = ax4.boxplot(bp_data, labels=order, patch_artist=True, showfliers=False,
                     medianprops=dict(color='#f43f5e', linewidth=2.5))
    bp_colors = sns.color_palette("plasma", len(order))
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor(BORDER_COLOR)

    ax4.axhline(35.4, color='#10b981', ls='--', lw=1.5, alpha=0.8, label='Batas Aman (35.4)')
    apply_dark_theme_axes(ax4, 'Distribusi PM2.5 per Wilayah WHO')
    ax4.set_xlabel('Wilayah WHO'); ax4.set_ylabel('PM2.5 (µg/m³)')
    ax4.tick_params(axis='x', rotation=15)
    ax4.legend(fontsize=8, loc='upper left')

    # Plot 5: Tren PM2.5 per Tahun
    ax5 = fig.add_subplot(gs[1, 2])
    yr = df_raw.dropna(subset=['pm25_concentration']).groupby('year')['pm25_concentration'].agg(['mean', 'median'])
    ax5.plot(yr.index, yr['mean'],   'o-', color='#38bdf8',  lw=2.5, ms=6, label='Rata-rata')
    ax5.plot(yr.index, yr['median'], 's--', color='#fb923c', lw=2, ms=5, label='Median')
    apply_dark_theme_axes(ax5, 'Tren PM2.5 per Tahun')
    ax5.set_xlabel('Tahun'); ax5.set_ylabel('PM2.5 (µg/m³)')
    ax5.legend(fontsize=8, loc='upper right')

    # Plot 6: Heatmap Korelasi
    ax6 = fig.add_subplot(gs[2, :])
    corr_cols = ['pm25_concentration', 'pm10_concentration', 'no2_concentration',
                 'year', 'population', 'latitude', 'longitude', 'number_of_stations']
    corr_m = df_raw[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_m, dtype=bool))
    sns.heatmap(corr_m, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax6, square=False,
                annot_kws={'size': 9, 'fontweight': 'bold'}, linewidths=0.5, linecolor=DARK_BG, cbar=True)
    ax6.set_title('Matriks Korelasi Fitur Numerik', color='#64d2ff', pad=12)
    ax6.set_facecolor(DARK_BG)
    ax6.tick_params(axis='x', rotation=15)
    ax6.tick_params(axis='y', rotation=0)

    plt.suptitle('Exploratory Data Analysis - Kualitas Udara Global WHO',
                 fontsize=15, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout(pad=3.0)
    plt.savefig('eda_visualisasi.png', dpi=150, facecolor=DARK_BG)
    plt.close()


    # ===================================================================
    # CHART 4: perbandingan_model.png
    # ===================================================================
    print("[INFO] Membuat perbandingan_model.png...")
    TARGET = 'pm25_concentration'
    FITUR_NUMERIK = ['year', 'latitude', 'longitude', 'pm10_concentration', 'no2_concentration', 'number_of_stations', 'who_ms', 'population']
    FITUR_KATEGORIKAL = ['who_region']
    SEMUA_FITUR = FITUR_NUMERIK + FITUR_KATEGORIKAL

    df_model = df_raw.dropna(subset=[TARGET]).copy()
    df_model = df_model[df_model[TARGET] > 0]
    df_model = df_model[SEMUA_FITUR + [TARGET]].copy()

    # Subset data agar proses CV sangat instan
    df_sub = df_model.sample(n=min(5000, len(df_model)), random_state=42)
    X_sub = df_sub[SEMUA_FITUR]
    y_sub = df_sub[TARGET]

    import sklearn
    if sklearn.__version__ >= '1.2':
        ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), FITUR_NUMERIK),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', ohe_transformer)
        ]), FITUR_KATEGORIKAL),
    ])

    MODELS = {
        'Linear Regression'   : LinearRegression(),
        'Ridge Regression'    : Ridge(alpha=1.0),
        'Decision Tree'       : DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest'       : RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting'   : GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=42),
    }

    hasil_cv = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, mdl in MODELS.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', mdl)])
        # Menggunakan execution single-threaded (tanpa n_jobs=-1)
        r2_scores = cross_val_score(pipe, X_sub, y_sub, cv=kf, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(pipe, X_sub, y_sub, cv=kf, scoring='neg_mean_squared_error'))
        hasil_cv[name] = {'r2': max(0.0, r2_scores.mean()), 'r2_std': r2_scores.std(), 'rmse': rmse_scores.mean()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(CARD_BG)
    ax2.set_facecolor(CARD_BG)

    names = list(hasil_cv.keys())
    r2_means = [hasil_cv[m]['r2'] for m in names]
    r2_stds = [hasil_cv[m]['r2_std'] for m in names]
    rmse_means = [hasil_cv[m]['rmse'] for m in names]

    best_r2_idx = np.argmax(r2_means)
    best_rmse_idx = np.argmin(rmse_means)
    colors_r2 = ['#a180ff' if i == best_r2_idx else '#334155' for i in range(len(names))]
    colors_rmse = ['#a180ff' if i == best_rmse_idx else '#334155' for i in range(len(names))]

    bars1 = ax1.barh(names, r2_means, xerr=r2_stds, color=colors_r2, edgecolor=BORDER_COLOR, alpha=0.9, capsize=4)
    apply_dark_theme_axes(ax1, 'Koefisien Determinasi (R2 Score)')
    ax1.set_xlabel('R2 Score (Lebih Tinggi = Lebih Baik)')
    ax1.set_xlim(0, 1.05)
    for bar, val in zip(bars1, r2_means):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontweight='bold', color='#ffffff')

    bars2 = ax2.barh(names, rmse_means, color=colors_rmse, edgecolor=BORDER_COLOR, alpha=0.9)
    apply_dark_theme_axes(ax2, 'Tingkat Kesalahan (RMSE)')
    ax2.set_xlabel('RMSE ug/m3 (Lebih Rendah = Lebih Baik)')
    ax2.set_xlim(0, max(rmse_means) * 1.15)
    for bar, val in zip(bars2, rmse_means):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontweight='bold', color='#ffffff')

    plt.suptitle('Perbandingan Performa Model - 5-Fold Cross Validation', fontsize=14, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout(pad=2.5)
    plt.savefig('perbandingan_model.png', dpi=150, facecolor=DARK_BG)
    plt.close()


    # ===================================================================
    # CHART 5: evaluasi_model_final.png
    # ===================================================================
    print("[INFO] Membuat evaluasi_model_final.png...")
    X_all = df_model[SEMUA_FITUR]
    y_all = df_model[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    final_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=80, random_state=42))
    ])
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)
    residuals = y_test.values - y_pred

    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    # 1. Scatter
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_scatter.set_facecolor(CARD_BG)
    ax_scatter.scatter(y_test, y_pred, alpha=0.25, s=6, color='#64d2ff', edgecolors='none')
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax_scatter.plot(lims, lims, 'r--', linewidth=2, label='Ideal')
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel('PM2.5 Aktual (ug/m3)')
    ax_scatter.set_ylabel('PM2.5 Prediksi (ug/m3)')
    apply_dark_theme_axes(ax_scatter, f'Aktual vs. Prediksi (R2 = {r2_score(y_test, y_pred):.4f})')
    ax_scatter.legend(loc='upper left', fontsize=8)

    # 2. Histogram Residual
    ax_resid = fig.add_subplot(gs[0, 1])
    ax_resid.set_facecolor(CARD_BG)
    ax_resid.hist(residuals, bins=50, color='#a180ff', alpha=0.85, edgecolor=BORDER_COLOR)
    ax_resid.axvline(0, color='#f43f5e', linewidth=2, linestyle='--', label='Residu = 0')
    ax_resid.set_xlabel('Residu (Aktual - Prediksi)')
    ax_resid.set_ylabel('Frekuensi')
    apply_dark_theme_axes(ax_resid, f'Distribusi Residual (MAE = {mean_absolute_error(y_test, y_pred):.2f})')
    ax_resid.legend(loc='upper right', fontsize=8)

    # 3. Feature Importance
    ax_feat = fig.add_subplot(gs[0, 2])
    ax_feat.set_facecolor(CARD_BG)
    # Mengambil nama fitur setelah preprocessing One-Hot Encoding
    ohe_step = final_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    if hasattr(ohe_step, 'get_feature_names_out'):
        ohe_feature_names = list(ohe_step.get_feature_names_out(FITUR_KATEGORIKAL))
    else:
        ohe_feature_names = list(ohe_step.get_feature_names(FITUR_KATEGORIKAL))
    all_feat_names = FITUR_NUMERIK + ohe_feature_names
    importances = final_pipe.named_steps['model'].feature_importances_

    indices = np.argsort(importances)[::-1][:8]
    top_names = [all_feat_names[i] for i in indices]
    top_importances = importances[indices]

    bars_feat = ax_feat.barh(top_names[::-1], top_importances[::-1], color='#10b981', edgecolor=BORDER_COLOR, alpha=0.9)
    ax_feat.set_xlabel('Tingkat Kepentingan Fitur')
    apply_dark_theme_axes(ax_feat, '8 Fitur Paling Berpengaruh')
    for bar, val in zip(bars_feat, top_importances[::-1]):
        ax_feat.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2%}', va='center', fontweight='bold', fontsize=8, color='#ffffff')

    plt.suptitle('Evaluasi Performa & Struktur Model Final', fontsize=14, fontweight='bold', color='#ffffff', y=1.02)
    plt.tight_layout()
    plt.savefig('evaluasi_model_final.png', dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()

    print("[SUCCESS] Semua file gambar berhasil dibuat ulang dengan tema gelap!")

if __name__ == '__main__':
    main()
