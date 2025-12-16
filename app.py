import streamlit as st
import segyio
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import tempfile
import os
import sys

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="SEG-Y Seismic Data Viewer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== INISIALISASI SESSION STATE ====================
if 'seismic_data' not in st.session_state:
    st.session_state.seismic_data = None
if 'seismic_info' not in st.session_state:
    st.session_state.seismic_info = None
if 'trace_headers' not in st.session_state:
    st.session_state.trace_headers = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# ==================== FUNGSI UTAMA ====================
def load_segy_file(file_path):
    """Baca file SEG-Y dengan error handling yang baik"""
    try:
        st.write("🔍 Mencoba membaca file SEG-Y...")
        
        # Buka file SEG-Y
        with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
            # Dapatkan info dasar
            n_traces = segyfile.tracecount
            n_samples = segyfile.bin[segyio.BinField.Samples]
            sample_interval = segyfile.bin.get(segyio.BinField.Interval, 0)
            format_code = segyfile.bin.get(segyio.BinField.Format, 1)
            
            st.write(f"✅ File valid! Traces: {n_traces}, Samples: {n_samples}")
            
            # Konversi format code ke string
            format_map = {
                1: "IBM Float (32-bit)",
                2: "Integer (32-bit)",
                3: "Integer (16-bit)",
                5: "IEEE Float (32-bit)",
                8: "Integer (8-bit)"
            }
            format_str = format_map.get(format_code, f"Unknown ({format_code})")
            
            # Baca data - batasi untuk performa
            max_traces = min(1000, n_traces)  # Max 1000 traces untuk demo
            st.write(f"📥 Membaca {max_traces} dari {n_traces} traces...")
            
            # Buat array untuk data
            data = np.zeros((n_samples, max_traces), dtype=np.float32)
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Baca traces
            for i in range(max_traces):
                data[:, i] = segyfile.trace[i]
                if i % 100 == 0:
                    progress_bar.progress((i + 1) / max_traces)
            
            progress_bar.empty()
            
            # Baca header teks
            text_header = segyfile.text[0]
            
            # Potong text header jika terlalu panjang
            if isinstance(text_header, (bytes, bytearray)):
                try:
                    text_header_str = text_header.decode('ascii', errors='ignore')
                except:
                    text_header_str = str(text_header)[:500]
            else:
                text_header_str = str(text_header)[:500]
            
            # Info lengkap
            info = {
                'n_traces': n_traces,
                'n_samples': n_samples,
                'sample_interval': sample_interval,
                'format': format_str,
                'format_code': format_code,
                'text_header': text_header_str,
                'file_size': os.path.getsize(file_path)
            }
            
            st.success(f"✅ Data berhasil dimuat! Shape: {data.shape}")
            return data, info, None
            
    except Exception as e:
        st.error(f"❌ Gagal membaca file SEG-Y: {str(e)}")
        st.write("Detail error:", sys.exc_info())
        return None, None, None

def create_dummy_data():
    """Buat data dummy untuk testing"""
    traces = 400
    samples = 1200
    
    # Buat data acak
    data = np.random.randn(samples, traces).astype(np.float32)
    
    # Tambahkan beberapa event sintetik untuk tampilan lebih realistis
    for i in range(traces):
        # Event 1
        event1_start = 300
        event1_end = 400
        data[event1_start:event1_end, i] += np.sin(i * 0.05) * 3.0
        
        # Event 2
        event2_start = 600
        event2_end = 750
        data[event2_start:event2_end, i] += np.cos(i * 0.03) * 2.0
        
        # Event 3
        event3_start = 900
        event3_end = 1000
        data[event3_start:event3_end, i] += np.sin(i * 0.08) * 1.5
    
    info = {
        'n_traces': traces,
        'n_samples': samples,
        'sample_interval': 2000,
        'format': 'IEEE Float (32-bit)',
        'format_code': 5,
        'text_header': 'DUMMY DATA - Generated for testing purposes\n' +
                      'This is synthetic seismic data for demonstration.',
        'file_size': data.nbytes
    }
    
    return data, info

# ==================== HEADER APLIKASI ====================
st.title("🌋 SEG-Y SEISMIC DATA VIEWER")
st.markdown("""
**Visualisasi data seismik format SEG-Y**  
Upload file `.sgy` atau `.segy` Anda untuk memulai analisis.
""")
st.markdown("---")

# ==================== SIDEBAR - INPUT DATA ====================
st.sidebar.header("📁 INPUT DATA")

# Dropdown untuk pilih metode
method = st.sidebar.selectbox(
    "Pilih metode input:",
    ["📂 Load File Lokal", "📤 Upload File Baru", "🎲 Gunakan Data Dummy"],
    index=0
)

# Variabel untuk menyimpan file yang akan diproses
file_to_process = None

# Logika berdasarkan pilihan
if method == "📂 Load File Lokal":
    st.sidebar.subheader("File dari Folder 'data'")
    
    # Cek apakah folder data ada
    if not os.path.exists("data"):
        os.makedirs("data")
        st.sidebar.info("📁 Folder 'data' dibuat. Taruh file .sgy Anda di sini.")
    else:
        # Cari file SEG-Y di folder data
        segy_files = []
        for file in os.listdir("data"):
            if file.lower().endswith(('.sgy', '.segy', '.ggy')):
                segy_files.append(file)
        
        if segy_files:
            selected_file = st.sidebar.selectbox(
                "Pilih file SEG-Y:",
                segy_files
            )
            
            file_path = os.path.join("data", selected_file)
            st.sidebar.info(f"📄 **File:** {selected_file}")
            st.sidebar.write(f"📍 **Path:** `{file_path}`")
            
            # Tombol load
            if st.sidebar.button("🚀 LOAD FILE INI", type="primary", use_container_width=True):
                file_to_process = file_path
        else:
            st.sidebar.warning("❌ Tidak ada file SEG-Y di folder 'data'")

elif method == "📤 Upload File Baru":
    st.sidebar.subheader("Upload File SEG-Y")
    
    uploaded_file = st.sidebar.file_uploader(
        "Pilih file SEG-Y:",
        type=['sgy', 'segy', 'ggy'],
        help="Pilih file data seismik format SEG-Y"
    )
    
    if uploaded_file is not None:
        st.sidebar.success(f"✅ **File:** {uploaded_file.name}")
        st.sidebar.write(f"📏 **Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        
        if st.sidebar.button("🚀 PROCESS & LOAD", type="primary", use_container_width=True):
            # Simpan ke file temporary
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sgy') as tmp:
                tmp.write(uploaded_file.getvalue())
                file_to_process = tmp.name

elif method == "🎲 Gunakan Data Dummy":
    st.sidebar.subheader("Data Sintetis untuk Testing")
    st.sidebar.info("""
    Data dummy akan dibuat otomatis.
    Berguna untuk testing fitur visualisasi.
    """)
    
    if st.sidebar.button("🎲 GENERATE DUMMY DATA", type="primary", use_container_width=True):
        # Buat data dummy
        data, info = create_dummy_data()
        
        # Simpan ke session state
        st.session_state.seismic_data = data
        st.session_state.seismic_info = info
        st.session_state.file_loaded = True
        
        st.sidebar.success("✅ Data dummy berhasil dibuat!")
        st.rerun()

# ==================== PROSES FILE YANG DIPILIH ====================
if file_to_process:
    with st.spinner(f"🔄 Memproses file..."):
        # Baca file SEG-Y
        data, info, _ = load_segy_file(file_to_process)
        
        if data is not None:
            # Simpan ke session state
            st.session_state.seismic_data = data
            st.session_state.seismic_info = info
            st.session_state.file_loaded = True
            
            # Hapus file temporary jika dari upload
            if method == "📤 Upload File Baru":
                try:
                    os.unlink(file_to_process)
                except:
                    pass
            
            st.success("✅ Data berhasil dimuat ke aplikasi!")
            st.rerun()
        else:
            st.error("❌ Gagal memuat data dari file.")

# ==================== SIDEBAR - KONTROL VISUALISASI ====================
st.sidebar.markdown("---")
st.sidebar.header("🎨 VISUALIZATION CONTROLS")

# Hanya tampilkan kontrol jika data sudah dimuat
if st.session_state.file_loaded and st.session_state.seismic_data is not None:
    data = st.session_state.seismic_data
    
    # 1. Pilihan Colormap
    st.sidebar.subheader("Color Map")
    colormap_options = {
        "Seismic (Blue-White-Red)": "RdBu_r",
        "Seismic (Red-White-Blue)": "RdBu",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Rainbow": "rainbow",
        "Jet": "jet",
        "Hot": "hot",
        "Gray": "gray"
    }
    
    selected_cmap = st.sidebar.selectbox(
        "Pilih colormap:",
        list(colormap_options.keys()),
        index=0
    )
    
    # 2. Kontrol Skala
    st.sidebar.subheader("Scale Control")
    scale_mode = st.sidebar.radio(
        "Mode:",
        ["Auto Scale", "Manual Scale"],
        index=0,
        horizontal=True
    )
    
    if scale_mode == "Manual Scale":
        data_min = float(np.nanmin(data))
        data_max = float(np.nanmax(data))
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            vmin = st.number_input("Vmin", value=float(data_min))
        with col2:
            vmax = st.number_input("Vmax", value=float(data_max))
    else:
        vmin = vmax = None
    
    # 3. Opsi Tambahan
    st.sidebar.subheader("Additional Options")
    
    reverse_time = st.sidebar.checkbox("Reverse Time Axis", value=True)
    show_wiggle = st.sidebar.checkbox("Show Wiggle Trace", value=False)
    
    # Pilih rentang trace
    use_trace_range = st.sidebar.checkbox("Custom Trace Range", value=False)
    if use_trace_range:
        n_traces = data.shape[1]
        trace_range = st.sidebar.slider(
            "Select traces:",
            0, n_traces-1,
            (0, min(200, n_traces-1))
        )
    else:
        trace_range = None
    
    # Tombol reset
    if st.sidebar.button("🔄 Reset Data", type="secondary"):
        st.session_state.seismic_data = None
        st.session_state.seismic_info = None
        st.session_state.file_loaded = False
        st.rerun()
    
    # Statistik cepat
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.metric("Traces", data.shape[1])
    st.sidebar.metric("Samples", data.shape[0])
    st.sidebar.metric("Data Size", f"{data.nbytes / (1024*1024):.2f} MB")

# ==================== TAMPILAN UTAMA ====================

# JIKA DATA BELUM DIMUAT
if not st.session_state.file_loaded or st.session_state.seismic_data is None:
    st.info("""
    ## 📋 **SELAMAT DATANG DI SEG-Y VIEWER**
    
    ### 🚀 **CARA MENGGUNAKAN:**
    
    1. **Di sidebar kiri**, pilih metode input data:
       - **📂 Load File Lokal**: Pilih file dari folder `data/`
       - **📤 Upload File Baru**: Upload file SEG-Y Anda
       - **🎲 Gunakan Data Dummy**: Data sintetis untuk testing
    
    2. **Klik tombol LOAD/PROCESS** untuk memuat data
    
    3. **Atur visualisasi** dengan kontrol di sidebar
    
    ### 📂 **PERSIAPAN FILE:**
    
    Untuk menggunakan **Load File Lokal**:
    ```
    E:\tekpro2\seismic-visualization\
    ├── app.py
    ├── data/                    ← BUAT FOLDER INI
    │   └── Test_Post_Stack.sgy  ← TARUH FILE ANDA DI SINI
    └── requirements.txt
    ```
    """)
    
    # Tampilkan struktur folder saat ini
    with st.expander("🔍 Lihat struktur folder saat ini", expanded=False):
        import pathlib
        current_dir = pathlib.Path(".")
        st.write("**Current directory:**", current_dir.absolute())
        
        tree = []
        for path in current_dir.rglob("*"):
            if path.is_relative_to(current_dir):
                depth = len(path.relative_to(current_dir).parts)
                indent = "  " * depth
                tree.append(f"{indent}📁 {path.name}" if path.is_dir() else f"{indent}📄 {path.name}")
        
        if tree:
            st.text("\n".join(tree[:50]))  # Batasi output
        else:
            st.write("Tidak ada file/folder")
    
    # Debug info
    with st.expander("🐛 Debug Information", expanded=False):
        st.write("**Python Version:**", sys.version)
        st.write("**Current Directory:**", os.getcwd())
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        
        # Cek apakah segyio bisa diimport
        try:
            import segyio
            st.success(f"✅ segyio version: {segyio.__version__}")
        except ImportError as e:
            st.error(f"❌ segyio import error: {e}")

# JIKA DATA SUDAH DIMUAT
else:
    data = st.session_state.seismic_data
    info = st.session_state.seismic_info
    
    # ==================== TAB VISUALISASI ====================
    tab1, tab2, tab3 = st.tabs(["📊 SEISMIC SECTION", "📈 STATISTICS", "ℹ️ FILE INFO"])
    
    with tab1:
        # Header info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Traces", info['n_traces'])
        with col2:
            st.metric("Samples", info['n_samples'])
        with col3:
            st.metric("Interval", f"{info['sample_interval']} μs")
        with col4:
            st.metric("Format", info['format'])
        
        # Siapkan data untuk ditampilkan
        if trace_range:
            display_data = data[:, trace_range[0]:trace_range[1]]
            trace_info = f"Traces {trace_range[0]}-{trace_range[1]}"
        else:
            display_data = data
            trace_info = f"All {data.shape[1]} traces"
        
        # Buat plot dengan Plotly
        fig = go.Figure(data=go.Heatmap(
            z=display_data,
            colorscale=colormap_options[selected_cmap],
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title="Amplitude",
                titleside="right"
            ),
            hoverinfo='x+y+z',
            hovertemplate='Trace: %{x}<br>Sample: %{y}<br>Amplitude: %{z:.6f}<extra></extra>'
        ))
        
        # Konfigurasi layout
        fig.update_layout(
            title=f"Seismic Section - {trace_info}",
            xaxis_title="Trace Number",
            yaxis_title="Time Sample",
            height=700,
            template="plotly_dark"
        )
        
        # Balik sumbu waktu jika dipilih
        if reverse_time:
            fig.update_yaxes(autorange="reversed")
        
        # Tampilkan plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Kontrol di bawah plot
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            # Trace individual
            st.subheader("Individual Trace View")
            selected_trace = st.slider(
                "Select trace to display:",
                0, data.shape[1]-1,
                min(50, data.shape[1]-1)
            )
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=data[:, selected_trace],
                mode='lines',
                name=f'Trace {selected_trace}',
                line=dict(color='cyan', width=2)
            ))
            
            fig2.update_layout(
                title=f"Trace {selected_trace}",
                xaxis_title="Sample",
                yaxis_title="Amplitude",
                height=300,
                template="plotly_dark"
            )
            
            if reverse_time:
                fig2.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_right:
            # Export options
            st.subheader("💾 Export")
            
            # Export as PNG
            if st.button("Save as PNG"):
                img_bytes = fig.to_image(format="png", scale=2)
                st.download_button(
                    label="⬇️ Download PNG",
                    data=img_bytes,
                    file_name="seismic_section.png",
                    mime="image/png"
                )
            
            # Export data subset
            if st.button("Export Data (CSV)"):
                # Export subset 100x100
                subset = data[:100, :100]
                csv = pd.DataFrame(subset).to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="seismic_subset.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.header("Data Statistics")
        
        # Hitung statistik
        stats = pd.DataFrame({
            "Statistic": ["Minimum", "Maximum", "Mean", "Median", 
                         "Standard Deviation", "Variance", "25th Percentile", 
                         "75th Percentile"],
            "Value": [
                np.nanmin(data),
                np.nanmax(data),
                np.nanmean(data),
                np.nanmedian(data),
                np.nanstd(data),
                np.nanvar(data),
                np.nanpercentile(data, 25),
                np.nanpercentile(data, 75)
            ]
        })
        
        stats["Value"] = stats["Value"].apply(lambda x: f"{x:.6f}")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.dataframe(stats, use_container_width=True)
        
        with col_stat2:
            # Histogram
            st.subheader("Amplitude Distribution")
            
            # Sample data untuk histogram (jika data terlalu besar)
            if data.size > 100000:
                sample_size = min(100000, data.size)
                flat_data = np.random.choice(data.flatten(), size=sample_size)
            else:
                flat_data = data.flatten()
            
            fig_hist = px.histogram(
                x=flat_data,
                nbins=100,
                title="Amplitude Histogram"
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.header("File Information")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("Basic Info")
            info_table = pd.DataFrame({
                "Property": ["File Size", "Data Type", "Dimensions", 
                            "Total Samples", "Memory Usage"],
                "Value": [
                    f"{info['file_size'] / (1024*1024):.2f} MB",
                    str(data.dtype),
                    f"{data.shape[1]} × {data.shape[0]}",
                    f"{data.size:,}",
                    f"{data.nbytes / (1024*1024):.2f} MB"
                ]
            })
            st.table(info_table)
        
        with col_info2:
            st.subheader("SEG-Y Header")
            with st.expander("View Text Header", expanded=True):
                st.text(info['text_header'])

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><b>SEG-Y Seismic Viewer v2.0</b> • Made for Post-Stack Data Analysis</p>
    <p>Powered by Streamlit, segyio, and Plotly</p>
</div>
""", unsafe_allow_html=True)

# ==================== DEBUG INFO (HIDDEN) ====================
if st.sidebar.checkbox("🔧 Show Debug Info", value=False):
    st.sidebar.markdown("---")
    st.sidebar.header("Debug Information")
    
    st.sidebar.write("**Session State:**")
    for key in st.session_state.keys():
        value = st.session_state[key]
        if hasattr(value, 'shape'):
            st.sidebar.write(f"- {key}: {value.shape}")
        else:
            st.sidebar.write(f"- {key}: {type(value)}")
    
    st.sidebar.write("**Python Info:**")
    st.sidebar.write(f"- Python: {sys.version.split()[0]}")
    st.sidebar.write(f"- segyio: {segyio.__version__}")
    st.sidebar.write(f"- numpy: {np.__version__}")