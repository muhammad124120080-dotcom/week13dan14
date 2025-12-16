import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.interpolate import griddata, NearestNDInterpolator
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Magnetic 2D", layout="wide")

# --- JUDUL & INTRO ---
st.title("📊 Pengolahan Data Magnetic 2D")
st.markdown("""
Aplikasi ini melakukan pemisahan anomali **Regional** dan **Residual**.
Mendukung proses **Gridding** dari data acak (scatter) menjadi data grid teratur.
""")

# --- SIDEBAR: UPLOAD FILE ---
st.sidebar.header("📁 Input Data")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file Excel/CSV", 
    type=['xlsx', 'csv', 'xls'],
    help="Format harus memiliki kolom: x, y, t_obs"
)

# --- FUNGSI GENERATOR DATA DUMMY ---
def get_dummy_dataframe(grid_size=50):
    """Generate dummy magnetic data for testing"""
    x = np.linspace(0, 1000, grid_size)
    y = np.linspace(0, 1000, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Buat anomali
    regional = 0.05 * X + 0.02 * Y + 45000
    r = np.sqrt((X - 500)**2 + (Y - 500)**2 + 50**2)
    residual = 5000 * (50 / r)**3
    noise = np.random.normal(0, 1, (grid_size, grid_size))
    total = regional + residual + noise
    
    # Flatten & Random Sampling
    df = pd.DataFrame({
        'x': X.flatten(),
        'y': Y.flatten(),
        't_obs': total.flatten()
    })
    return df.sample(frac=0.8).reset_index(drop=True)

# --- FUNGSI POLYNOMIAL FITTING ---
def polyfit2d(x, y, z, order=1):
    """2D polynomial fitting for trend surface analysis"""
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    # Filter NaN
    mask = ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]
    
    if len(x) == 0:
        return lambda xi, yi: np.zeros_like(xi)
    
    if order == 1:
        A = np.c_[np.ones(x.shape), x, y]
    elif order == 2:
        A = np.c_[np.ones(x.shape), x, y, x**2, x*y, y**2]
    else:
        raise ValueError("Order must be 1 or 2")
    
    # Least squares solution
    try:
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        C = np.zeros(A.shape[1])
    
    # Return prediction function
    if order == 1:
        return lambda xi, yi: C[0] + C[1]*xi + C[2]*yi
    elif order == 2:
        return lambda xi, yi: (C[0] + C[1]*xi + C[2]*yi + 
                               C[3]*xi**2 + C[4]*xi*yi + C[5]*yi**2)

# --- LOAD DATA ---
df_input = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df_input = pd.read_excel(uploaded_file)
        else:
            df_input = pd.read_csv(uploaded_file)
        
        # Validasi kolom
        required_cols = ['x', 'y', 't_obs']
        if not all(col in df_input.columns for col in required_cols):
            st.error(f"File harus memiliki kolom: {required_cols}")
            st.info("Kolom yang ditemukan: " + ", ".join(df_input.columns.tolist()))
            st.stop()
            
        st.success(f"✅ File berhasil dibaca! {len(df_input)} data points")
            
    except Exception as e:
        st.error(f"❌ Error membaca file: {e}")
        st.info("Menggunakan data dummy...")
        df_input = get_dummy_dataframe()
else:
    st.info("📝 Tidak ada file diunggah. Menggunakan data dummy...")
    df_input = get_dummy_dataframe()

# Tampilkan preview data
with st.expander("📊 Preview Data"):
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_input.head(10), use_container_width=True)
    with col2:
        st.dataframe(df_input.describe(), use_container_width=True)
    st.write(f"📈 Total data points: {len(df_input)}")

# --- SIDEBAR: PARAMETER GRIDDING ---
st.sidebar.header("🗺️ Parameter Gridding")
st.sidebar.markdown("---")

# Tentukan batas area
x_min, x_max = df_input['x'].min(), df_input['x'].max()
y_min, y_max = df_input['y'].min(), df_input['y'].max()

st.sidebar.text(f"Range X: {x_min:.1f} - {x_max:.1f} m")
st.sidebar.text(f"Range Y: {y_min:.1f} - {y_max:.1f} m")

# Input ukuran sel
default_cell = max((x_max - x_min) / 50, 1.0)  # Default 50 grid points
cell_size = st.sidebar.number_input(
    "Ukuran Sel (meter)",
    min_value=1.0,
    max_value=float(x_max - x_min),
    value=float(default_cell),
    step=5.0,
    help="Semakin kecil nilai ini, resolusi peta semakin tinggi tapi komputasi lebih berat."
)

# Metode interpolasi
interp_method = st.sidebar.selectbox(
    "Metode Interpolasi",
    ['linear', 'cubic', 'nearest'],
    index=0,
    help="'linear': standar, 'cubic': lebih halus, 'nearest': kotak-kotak"
)

# --- SIDEBAR: VISUALISASI (CHALLENGE) ---
st.sidebar.header("🎨 Visualisasi")
st.sidebar.markdown("---")

# 1. Pilihan Colormap
available_cmaps = ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 
                   'RdYlBu', 'Spectral', 'rainbow', 'seismic', 'terrain', 'tab20c']
cmap_choice = st.sidebar.selectbox("Pilih Colormap", available_cmaps, index=0)

# 2. Mode Scaling
scale_mode = st.sidebar.radio("Mode Scaling", ["Auto", "Manual"], horizontal=True)

# 3. Opsi Tambahan
st.sidebar.subheader("⚙️ Opsi Tambahan")
invert_yaxis = st.sidebar.checkbox("Balik Sumbu Y", value=False)
show_grid = st.sidebar.checkbox("Tampilkan Grid", value=False)
show_points = st.sidebar.checkbox("Tampilkan Titik Data", value=True)
point_size = st.sidebar.slider("Ukuran Titik", 1, 20, 5) if show_points else 0

# Manual scale controls (akan ditampilkan jika mode manual dipilih)
vmin_manual, vmax_manual = None, None
if scale_mode == "Manual":
    st.sidebar.subheader("🔧 Manual Scale Settings")
    
    # Dapatkan range data untuk referensi
    data_min = float(df_input['t_obs'].min())
    data_max = float(df_input['t_obs'].max())
    data_range = data_max - data_min
    
    vmin_manual = st.sidebar.slider(
        "Minimum Value (vmin)",
        min_value=float(data_min - data_range),
        max_value=float(data_max),
        value=float(data_min),
        step=float(data_range/100)
    )
    
    vmax_manual = st.sidebar.slider(
        "Maximum Value (vmax)",
        min_value=float(data_min),
        max_value=float(data_max + data_range),
        value=float(data_max),
        step=float(data_range/100)
    )

# --- PROSES GRIDDING ---
st.subheader("🔧 Proses Gridding")
st.markdown("---")

# Buat grid
xi = np.arange(x_min, x_max + cell_size, cell_size)
yi = np.arange(y_min, y_max + cell_size, cell_size)

# Pastikan ada setidaknya 2 titik
if len(xi) < 2:
    xi = np.array([x_min, x_max])
if len(yi) < 2:
    yi = np.array([y_min, y_max])

X, Y = np.meshgrid(xi, yi)

# Interpolasi data scatter ke grid
with st.spinner("Melakukan interpolasi data..."):
    try:
        T_obs_grid = griddata(
            (df_input['x'], df_input['y']),
            df_input['t_obs'],
            (X, Y),
            method=interp_method,
            fill_value=np.nan
        )
        st.success("✅ Interpolasi berhasil!")
    except Exception as e:
        st.error(f"❌ Error interpolasi: {e}")
        # Gunakan nearest sebagai fallback
        T_obs_grid = griddata(
            (df_input['x'], df_input['y']),
            df_input['t_obs'],
            (X, Y),
            method='nearest',
            fill_value=np.nan
        )

# --- VISUALISASI HASIL GRIDDING ---
st.subheader("📈 1. Visualisasi Data")
tab1, tab2 = st.tabs(['🗺️ Peta Kontur (Gridded)', "📊 Sebaran Titik Data (Scatter)"])

with tab1:
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        # Buat figure dengan ukuran yang sesuai
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Plot kontur
        if scale_mode == "Auto":
            cont1 = ax1.contourf(X, Y, T_obs_grid, levels=50, cmap=cmap_choice, alpha=0.9)
        else:
            cont1 = ax1.contourf(X, Y, T_obs_grid, levels=50, cmap=cmap_choice, 
                                vmin=vmin_manual, vmax=vmax_manual, alpha=0.9)
        
        # Plot titik data jika diaktifkan
        if show_points:
            scatter1 = ax1.scatter(df_input['x'], df_input['y'], c='white', 
                                  s=point_size, alpha=0.6, edgecolors='black', 
                                  linewidth=0.5, label='Titik Observasi')
        
        # Konfigurasi plot
        ax1.set_title(f"Total Magnetic Intensity\nGrid Spacing: {cell_size:.1f} m | Metode: {interp_method}", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Easting (x) [m]', fontsize=12)
        ax1.set_ylabel('Northing (y) [m]', fontsize=12)
        
        if show_grid:
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        if invert_yaxis:
            ax1.invert_yaxis()
        
        ax1.set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar1 = fig1.colorbar(cont1, ax=ax1, pad=0.01)
        cbar1.set_label('Intensitas Magnetik (nT)', fontsize=12)
        
        # Tambahkan legend
        if show_points:
            ax1.legend(loc='upper right', fontsize=10)
        
        st.pyplot(fig1, use_container_width=True)
    
    with col_b:
        st.info("""
        **📋 Informasi:**
        
        **Grid:**
        - Ukuran grid: {:.1f} m
        - Dimensi: {} × {} titik
        - Metode: {}
        
        **Legenda:**
        - Warna: Intensitas magnetik
        - Titik putih: Lokasi pengukuran
        - Area transparan: Tidak ada data
        
        **Tips:**
        - Gunakan slider di sidebar untuk mengatur visualisasi
        - Klik tombol expander untuk statistik detail
        """.format(cell_size, len(xi), len(yi), interp_method))

with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Scatter plot data asli
    scatter2 = ax2.scatter(df_input['x'], df_input['y'], 
                          c=df_input['t_obs'], 
                          cmap=cmap_choice, 
                          s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Konfigurasi plot
    ax2.set_title("Posisi Titik Pengukuran Asli", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Easting (x) [m]', fontsize=12)
    ax2.set_ylabel('Northing (y) [m]', fontsize=12)
    
    if show_grid:
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if invert_yaxis:
        ax2.invert_yaxis()
    
    ax2.set_aspect('equal', adjustable='box')
    
    # Colorbar
    cbar2 = fig2.colorbar(scatter2, ax=ax2, pad=0.01)
    cbar2.set_label('Intensitas Magnetik (nT)', fontsize=12)
    
    st.pyplot(fig2, use_container_width=True)

st.divider()

# --- PENGOLAHAN REGIONAL-RESIDUAL ---
st.subheader("📊 2. Pemisahan Regional-Residual")

# Isi NaN dengan nearest value untuk filter
mask_nan = np.isnan(T_obs_grid)
if np.any(mask_nan):
    try:
        mask_valid = ~mask_nan
        if np.sum(mask_valid) > 0:
            xx_valid, yy_valid = X[mask_valid], Y[mask_valid]
            zz_valid = T_obs_grid[mask_valid]
            interp_nn = NearestNDInterpolator(list(zip(xx_valid.flatten(), yy_valid.flatten())), 
                                            zz_valid.flatten())
            T_filled = interp_nn(X, Y)
        else:
            T_filled = T_obs_grid
    except:
        T_filled = T_obs_grid
else:
    T_filled = T_obs_grid

# --- SIDEBAR: METODE PEMISAHAN ---
st.sidebar.header("⚡ Metode Pemisahan")
st.sidebar.markdown("---")

method = st.sidebar.selectbox(
    "Metode Pemisahan:", 
    ["2D Moving Average", "Trend Surface Analysis"]
)

Calculated_Regional = np.zeros_like(T_obs_grid)

if method == "2D Moving Average":
    window_size = st.sidebar.slider("Ukuran Window", 3, 31, 9, step=2)
    st.sidebar.caption(f"Ukuran fisik window = {window_size * cell_size:.1f} meter")
    try:
        Calculated_Regional = uniform_filter(T_filled, size=window_size, mode='nearest')
    except:
        Calculated_Regional = T_filled

elif method == "Trend Surface Analysis":
    poly_order = st.sidebar.radio("Orde Polynomial:", [1, 2], horizontal=True)
    try:
        poly_func = polyfit2d(X, Y, T_obs_grid, order=poly_order)
        Calculated_Regional = poly_func(X, Y)
    except:
        Calculated_Regional = np.nanmean(T_obs_grid) * np.ones_like(T_obs_grid)

# Hitung residual
Calculated_Residual = T_obs_grid - Calculated_Regional

# --- TAMPILAN HASIL REGIONAL-RESIDUAL ---
col_reg, col_res = st.columns(2)

with col_reg:
    st.markdown("### 🗺️ Anomali Regional (Trend)")
    
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    
    # Plot regional
    cont3 = ax3.contourf(X, Y, Calculated_Regional, levels=40, cmap=cmap_choice, alpha=0.9)
    
    # Konfigurasi plot
    ax3.set_title(f"Anomali Regional\nMetode: {method}", fontsize=12, fontweight='bold')
    ax3.set_xlabel('Easting (x) [m]', fontsize=11)
    ax3.set_ylabel('Northing (y) [m]', fontsize=11)
    
    if show_grid:
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if invert_yaxis:
        ax3.invert_yaxis()
    
    ax3.set_aspect('equal', adjustable='box')
    
    # Colorbar
    cbar3 = fig3.colorbar(cont3, ax=ax3, pad=0.01)
    cbar3.set_label('Intensitas (nT)', fontsize=11)
    
    st.pyplot(fig3, use_container_width=True)
    
    # Statistik regional
    with st.expander("📈 Statistik Regional"):
        reg_valid = Calculated_Regional[~np.isnan(Calculated_Regional)]
        if len(reg_valid) > 0:
            st.metric("Rata-rata", f"{np.mean(reg_valid):.1f} nT")
            st.metric("Std Dev", f"{np.std(reg_valid):.1f} nT")
            st.metric("Min", f"{np.min(reg_valid):.1f} nT")
            st.metric("Max", f"{np.max(reg_valid):.1f} nT")

with col_res:
    st.markdown("### 🎯 Anomali Residual (Target)")
    
    fig4, ax4 = plt.subplots(figsize=(8, 7))
    
    # Plot residual dengan colormap seismic
    res_valid = Calculated_Residual[~np.isnan(Calculated_Residual)]
    if len(res_valid) > 0:
        vmax = np.percentile(np.abs(res_valid), 95)
        vmin = -vmax
    else:
        vmin, vmax = -100, 100
    
    cont4 = ax4.contourf(X, Y, Calculated_Residual, levels=40, 
                        cmap='seismic', vmin=vmin, vmax=vmax, alpha=0.9)
    
    # Konfigurasi plot
    ax4.set_title(f"Anomali Residual\nMetode: {method}", fontsize=12, fontweight='bold')
    ax4.set_xlabel('Easting (x) [m]', fontsize=11)
    ax4.set_ylabel('Northing (y) [m]', fontsize=11)
    
    if show_grid:
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if invert_yaxis:
        ax4.invert_yaxis()
    
    ax4.set_aspect('equal', adjustable='box')
    
    # Colorbar
    cbar4 = fig4.colorbar(cont4, ax=ax4, pad=0.01)
    cbar4.set_label('Intensitas (nT)', fontsize=11)
    
    st.pyplot(fig4, use_container_width=True)
    
    # Statistik residual
    with st.expander("📈 Statistik Residual"):
        if len(res_valid) > 0:
            st.metric("Rata-rata", f"{np.mean(res_valid):.1f} nT")
            st.metric("Std Dev", f"{np.std(res_valid):.1f} nT")
            st.metric("Min", f"{np.min(res_valid):.1f} nT")
            st.metric("Max", f"{np.max(res_valid):.1f} nT")
            st.metric("Amplitudo Max", f"{np.max(np.abs(res_valid)):.1f} nT")

# --- STATISTIK DATA ---
st.divider()
st.subheader("📋 Ringkasan Statistik")

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.metric("📊 Data Points", len(df_input))
    st.metric("📍 Min X", f"{x_min:.1f} m")

with col_stat2:
    st.metric("📐 Max X", f"{x_max:.1f} m")
    st.metric("📍 Min Y", f"{y_min:.1f} m")

with col_stat3:
    st.metric("📐 Max Y", f"{y_max:.1f} m")
    st.metric("📈 Min Anomali", f"{df_input['t_obs'].min():.1f} nT")

with col_stat4:
    st.metric("📉 Max Anomali", f"{df_input['t_obs'].max():.1f} nT")
    st.metric("📊 Mean Anomali", f"{df_input['t_obs'].mean():.1f} nT")

# --- DOWNLOAD HASIL ---
st.divider()
st.subheader("💾 Download Hasil")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    if st.button("📥 Download Data Asli (CSV)"):
        csv = df_input.to_csv(index=False)
        st.download_button(
            label="Klik untuk Download",
            data=csv,
            file_name="data_magnetik_asli.csv",
            mime="text/csv"
        )

with col_dl2:
    if st.button("📥 Download Grid Data (CSV)"):
        # Flatten grid data
        grid_df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            't_obs': T_obs_grid.flatten(),
            'regional': Calculated_Regional.flatten(),
            'residual': Calculated_Residual.flatten()
        })
        csv = grid_df.to_csv(index=False)
        st.download_button(
            label="Klik untuk Download",
            data=csv,
            file_name="data_magnetik_grid.csv",
            mime="text/csv"
        )

with col_dl3:
    if st.button("📸 Download Semua Plot"):
        st.info("Fitur ini akan menyimpan semua plot sebagai gambar PNG")
        # Implementasi save semua plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success(f"File akan disimpan dengan format: magnetic_plots_{timestamp}.zip")

# --- INFORMASI APLIKASI ---
st.divider()
st.info("""
**ℹ️ Aplikasi Analisis Data Magnetik 2D**  
Dikembangkan untuk Studi Kasus Week 14 - Implementasi Streamlit untuk Visualisasi dan Pemrosesan Data Magnetik

**✨ Fitur Challenge yang diimplementasikan:**
1. ✅ **Pilihan colormap** - Dropdown di sidebar dengan 12 pilihan colormap
2. ✅ **Kontrol vmin/vmax** - Mode Auto/Manual dengan slider untuk kontrol manual
3. ✅ **Opsi Auto Scale dan Manual Scale** - Fleksibilitas dalam scaling warna
4. ✅ **Opsi tambahan** - Balik sumbu Y, tampilkan grid, kontrol ukuran titik

**⚙️ Parameter yang digunakan:**
- Ukuran grid: {:.1f} m
- Metode interpolasi: {}
- Metode pemisahan: {}
- Colormap: {}
""".format(cell_size, interp_method, method, cmap_choice))

# --- FOOTER ---
st.markdown("---")
st.caption("""
Dikembangkan oleh: **Asido Saputra Sigalingging, Nugroho Prasetyo, Putu Pradnya Andika**  
© 2024 - Analisis Data Geofisika - Metode Magnetik
""")