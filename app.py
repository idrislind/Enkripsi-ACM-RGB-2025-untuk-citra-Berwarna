"""
Web Application untuk Enkripsi Citra Digital Berbasis Arnold Cat Map
Menggunakan Streamlit untuk Interface

Installation:
pip install streamlit opencv-python pillow numpy pandas matplotlib seaborn scipy

Run:
streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import base64
from typing import Tuple

# ==================== IMPORT DARI SISTEM ACM ====================

class ArnoldCatMap:
    """Implementasi Arnold Cat Map untuk enkripsi citra"""
    
    def __init__(self, b: int = 2, c: int = 2):
        self.b = b
        self.c = c
        
    def encrypt(self, image: np.ndarray, iterations: int = 5) -> np.ndarray:
        if len(image.shape) == 2:
            return self._encrypt_channel(image, iterations)
        else:
            encrypted = np.zeros_like(image)
            for ch in range(image.shape[2]):
                encrypted[:, :, ch] = self._encrypt_channel(image[:, :, ch], iterations)
            return encrypted
    
    def _encrypt_channel(self, channel: np.ndarray, iterations: int) -> np.ndarray:
        N = channel.shape[0]
        encrypted = channel.copy().astype(np.float32)
        
        # Permutasi (ACM)
        for _ in range(iterations):
            temp = np.zeros_like(encrypted)
            for x in range(N):
                for y in range(N):
                    x_new = (x + self.b * y) % N
                    y_new = (self.c * x + (self.b * self.c + 1) * y) % N
                    temp[x_new, y_new] = encrypted[x, y]
            encrypted = temp
        
        # Feedback Diffusion Layer - Sangat sensitif terhadap perubahan
        # Setiap pixel di-ADD dengan kombinasi:
        # 1. Key dari parameter (position-based)
        # 2. Previous pixel (feedback) untuk spread perubahan
        key_base = (self.b * self.c * 123 + self.b * 45 + self.c * 67) % 256
        
        # Store encrypted values sebagai uint8 untuk consistency
        encrypted_uint8 = np.clip(encrypted, 0, 255).astype(np.uint8)
        
        for x in range(N):
            for y in range(N):
                # Hitung key yang unik untuk setiap posisi
                pos_key = (key_base + x * 17 + y * 31) % 256
                
                # Feedback dari pixel sebelumnya
                if x == 0 and y == 0:
                    prev_val = key_base
                elif y == 0:
                    prev_val = int(encrypted_uint8[x-1, N-1])
                else:
                    prev_val = int(encrypted_uint8[x, y-1])
                
                # Diffusion: ADD (modulo 256) dengan combination dari pos_key dan prev_val
                feedback_key = (pos_key ^ prev_val) % 256
                encrypted_uint8[x, y] = (int(encrypted_uint8[x, y]) + feedback_key) % 256
        
        return encrypted_uint8
    
    def decrypt(self, cipher: np.ndarray, iterations: int = 5) -> np.ndarray:
        if len(cipher.shape) == 2:
            return self._decrypt_channel(cipher, iterations)
        else:
            decrypted = np.zeros_like(cipher)
            for ch in range(cipher.shape[2]):
                decrypted[:, :, ch] = self._decrypt_channel(cipher[:, :, ch], iterations)
            return decrypted
    
    def _decrypt_channel(self, channel: np.ndarray, iterations: int) -> np.ndarray:
        N = channel.shape[0]
        # Gunakan uint8 untuk presisi arithmetic
        decrypted = channel.astype(np.uint8).copy()
        
        # Reverse Feedback Diffusion Layer
        key_base = (self.b * self.c * 123 + self.b * 45 + self.c * 67) % 256
        
        for x in range(N):
            for y in range(N):
                # Hitung key yang sama seperti encryption
                pos_key = (key_base + x * 17 + y * 31) % 256
                
                # Feedback dari cipher asli (BUKAN dari decrypted yang sudah partial)
                if x == 0 and y == 0:
                    prev_val = key_base
                elif y == 0:
                    prev_val = int(channel[x-1, N-1])
                else:
                    prev_val = int(channel[x, y-1])
                
                # Reverse diffusion: kurangi dengan combination (modulo 256)
                feedback_key = (pos_key ^ prev_val) % 256
                decrypted[x, y] = (int(decrypted[x, y]) - feedback_key) % 256
        
        # Reverse Permutasi (ACM inverse)
        for _ in range(iterations):
            temp = np.zeros((N, N), dtype=np.uint8)
            for x in range(N):
                for y in range(N):
                    x_orig = ((self.b * self.c + 1) * x - self.b * y) % N
                    y_orig = (-self.c * x + y) % N
                    temp[x_orig, y_orig] = decrypted[x, y]
            decrypted = temp
        
        return decrypted


class ImageAnalyzer:
    """Tools untuk analisis keamanan enkripsi"""
    
    @staticmethod
    def calculate_correlation(image: np.ndarray, direction: str = 'horizontal', 
                            num_samples: int = 1000) -> Tuple[float, np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        x_samples = []
        y_samples = []
        
        for _ in range(num_samples):
            if direction == 'horizontal':
                i = np.random.randint(0, h)
                j = np.random.randint(0, w - 1)
                x_samples.append(image[i, j])
                y_samples.append(image[i, j + 1])
            elif direction == 'vertical':
                i = np.random.randint(0, h - 1)
                j = np.random.randint(0, w)
                x_samples.append(image[i, j])
                y_samples.append(image[i + 1, j])
            elif direction == 'diagonal':
                i = np.random.randint(0, h - 1)
                j = np.random.randint(0, w - 1)
                x_samples.append(image[i, j])
                y_samples.append(image[i + 1, j + 1])
        
        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        correlation = np.corrcoef(x_samples, y_samples)[0, 1]
        
        return correlation, x_samples, y_samples
    
    @staticmethod
    def calculate_npcr_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> Tuple[float, float]:
        # Gunakan channel pertama saja jika RGB
        if len(cipher1.shape) == 3:
            cipher1 = cipher1[:, :, 0]
            cipher2 = cipher2[:, :, 0]
        
        # Hitung perbedaan
        diff_matrix = (cipher1 != cipher2).astype(int)
        total_pixels = diff_matrix.size
        changed_pixels = np.sum(diff_matrix)
        
        # NPCR: Persentase pixel yang berubah
        npcr = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        
        # UACI: Rata-rata perubahan intensitas
        if changed_pixels > 0:
            intensity_diff = np.sum(np.abs(cipher1.astype(float) - cipher2.astype(float)))
            uaci = (intensity_diff / (total_pixels * 255)) * 100
        else:
            uaci = 0.0
        
        return npcr, uaci
    
    @staticmethod
    def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((255.0 ** 2) / mse)


# ==================== HELPER FUNCTIONS ====================

def make_square(image: np.ndarray) -> np.ndarray:
    """Crop citra menjadi persegi"""
    h, w = image.shape[:2]
    size = min(h, w)
    if len(image.shape) == 2:
        return image[:size, :size]
    else:
        return image[:size, :size, :]

def resize_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Resize citra ke ukuran target"""
    return cv2.resize(image, (target_size, target_size))

def get_image_download_link(img_array: np.ndarray, filename: str = "encrypted_image.png") -> str:
    """Generate link download untuk citra"""
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">📥 Download {filename}</a>'
    return href


# ==================== STREAMLIT APP ====================

def main():
    # Page Config
    st.set_page_config(
        page_title="ACM Image Encryption",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            padding: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #28a745;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Sistem Enkripsi Citra Digital</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Berbasis Arnold Cat Map (ACM) - ITERA 2025</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Pilih Mode:",
        ["🎨 Enkripsi/Dekripsi", "📊 Analisis Keamanan", "📚 Tentang"]
    )
    
    st.sidebar.markdown("---")
    
    # Parameter ACM
    st.sidebar.subheader("Parameter Kunci ACM")
    param_b = st.sidebar.number_input("Parameter b:", min_value=1, max_value=100, value=2)
    param_c = st.sidebar.number_input("Parameter c:", min_value=1, max_value=100, value=2)
    iterations = st.sidebar.slider("Jumlah Iterasi (m):", min_value=1, max_value=50, value=5)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Parameter Kunci:**
    - **b, c**: Integer positif (kunci rahasia)
    - **m**: Jumlah iterasi (semakin banyak = lebih acak)
    
    ⚠️ Simpan parameter ini untuk dekripsi!
    """)
    
    # Initialize ACM
    acm = ArnoldCatMap(b=param_b, c=param_c)
    
    # ==================== MODE 1: ENKRIPSI/DEKRIPSI ====================
    
    if mode == "🎨 Enkripsi/Dekripsi":
        st.header("🎨 Enkripsi dan Dekripsi Citra")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Citra")
            uploaded_file = st.file_uploader(
                "Pilih file citra (JPG, PNG, TIFF)", 
                type=['jpg', 'jpeg', 'png', 'tiff']
            )
            
            if uploaded_file is not None:
                # Load image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display original
                st.image(image, caption="Citra Original", use_container_width=True)
                
                # Image info
                st.info(f"📐 Ukuran: {image.shape[1]}×{image.shape[0]} | Tipe: {'RGB' if len(image.shape)==3 else 'Grayscale'}")
                
                # Preprocessing
                st.subheader("🔧 Preprocessing")
                target_size = st.selectbox("Resize ke:", [256, 512, 1024], index=1)
                
                if st.button("⚙️ Proses Gambar", type="primary"):
                    with st.spinner("Memproses citra..."):
                        # Make square and resize
                        image_square = make_square(image)
                        image_processed = resize_image(image_square, target_size)
                        
                        st.session_state['image_processed'] = image_processed
                        st.success(f"✅ Citra diproses ke {target_size}×{target_size}")
        
        with col2:
            if 'image_processed' in st.session_state:
                st.subheader("🔐 Enkripsi")
                
                image_processed = st.session_state['image_processed']
                
                if st.button("🔒 Enkripsi Citra", type="primary"):
                    with st.spinner("Mengenkripsi citra..."):
                        start_time = time.time()
                        
                        # Encrypt
                        cipher = acm.encrypt(image_processed, iterations)
                        
                        encrypt_time = (time.time() - start_time) * 1000
                        
                        st.session_state['cipher'] = cipher
                        st.session_state['encrypt_time'] = encrypt_time
                        
                        # Display cipher
                        st.image(cipher, caption="Citra Terenkripsi", use_container_width=True)
                        
                        st.success(f"✅ Enkripsi selesai dalam {encrypt_time:.2f} ms")
                        
                        # Download link
                        st.markdown(get_image_download_link(cipher, "encrypted_image.png"), 
                                  unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("🔓 Dekripsi")
                
                if 'cipher' in st.session_state:
                    cipher = st.session_state['cipher']
                    
                    if st.button("🔓 Dekripsi Citra", type="secondary"):
                        with st.spinner("Mendekripsi citra..."):
                            start_time = time.time()
                            
                            # Decrypt
                            decrypted = acm.decrypt(cipher, iterations)
                            
                            decrypt_time = (time.time() - start_time) * 1000
                            
                            # Display decrypted
                            st.image(decrypted, caption="Citra Terdekripsi", use_container_width=True)
                            
                            # Validate
                            identical = np.array_equal(image_processed, decrypted)
                            
                            if identical:
                                st.success(f"✅ Dekripsi sempurna! ({decrypt_time:.2f} ms)")
                                st.balloons()
                            else:
                                st.error("❌ Dekripsi tidak sempurna!")
                            
                            # Download link
                            st.markdown(get_image_download_link(decrypted, "decrypted_image.png"), 
                                      unsafe_allow_html=True)
    
    # ==================== MODE 2: ANALISIS KEAMANAN ====================
    
    elif mode == "📊 Analisis Keamanan":
        st.header("📊 Analisis Keamanan Enkripsi")
        
        uploaded_file = st.file_uploader(
            "Upload citra untuk dianalisis", 
            type=['jpg', 'jpeg', 'png', 'tiff']
        )
        
        if uploaded_file is not None:
            # Load and process
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image_square = make_square(image)
            image_processed = resize_image(image_square, 512)
            
            # Encrypt
            with st.spinner("Mengenkripsi citra untuk analisis..."):
                cipher = acm.encrypt(image_processed, iterations)
                decrypted = acm.decrypt(cipher, iterations)
            
            st.success("✅ Enkripsi selesai, memulai analisis...")
            
            # Display images
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image_processed, caption="Plain Image", use_container_width=True)
            with col2:
                st.image(cipher, caption="Cipher Image", use_container_width=True)
            with col3:
                st.image(decrypted, caption="Decrypted Image", use_container_width=True)
            
            st.markdown("---")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Histogram", "🔗 Korelasi", "🛡️ NPCR & UACI", "📐 PSNR", "📈 Ringkasan"
            ])
            
            # TAB 1: HISTOGRAM
            with tab1:
                st.subheader("📊 Analisis Histogram")
                
                with st.spinner("Menghitung histogram..."):
                    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                    
                    colors = ['red', 'green', 'blue']
                    color_names = ['Red', 'Green', 'Blue']
                    
                    # Prepare histogram data for CSV export
                    histogram_data = {'Intensity': list(range(256))}
                    
                    for i, (color, name) in enumerate(zip(colors, color_names)):
                        # Plain histogram
                        hist_plain, _ = np.histogram(image_processed[:, :, i].flatten(), 
                                                     bins=256, range=[0, 256])
                        axes[0, i].hist(image_processed[:, :, i].flatten(), 
                                       bins=256, range=[0, 256], 
                                       color=color, alpha=0.7, edgecolor='black')
                        axes[0, i].set_title(f'Plain - {name} Channel')
                        axes[0, i].set_ylabel('Frequency')
                        axes[0, i].grid(alpha=0.3)
                        
                        # Cipher histogram
                        hist_cipher, _ = np.histogram(cipher[:, :, i].flatten(), 
                                                      bins=256, range=[0, 256])
                        axes[1, i].hist(cipher[:, :, i].flatten(), 
                                       bins=256, range=[0, 256], 
                                       color=color, alpha=0.7, edgecolor='black')
                        axes[1, i].set_title(f'Cipher - {name} Channel')
                        axes[1, i].set_ylabel('Frequency')
                        axes[1, i].grid(alpha=0.3)
                        
                        # Decrypted histogram
                        hist_decrypted, _ = np.histogram(decrypted[:, :, i].flatten(), 
                                                         bins=256, range=[0, 256])
                        axes[2, i].hist(decrypted[:, :, i].flatten(), 
                                       bins=256, range=[0, 256], 
                                       color=color, alpha=0.7, edgecolor='black')
                        axes[2, i].set_title(f'Decrypted - {name} Channel')
                        axes[2, i].set_xlabel('Pixel Intensity')
                        axes[2, i].set_ylabel('Frequency')
                        axes[2, i].grid(alpha=0.3)
                        
                        # Store histogram data for CSV
                        histogram_data[f'Plain_{name}'] = hist_plain.tolist()
                        histogram_data[f'Cipher_{name}'] = hist_cipher.tolist()
                        histogram_data[f'Decrypted_{name}'] = hist_decrypted.tolist()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # CSV Export
                st.markdown("---")
                st.subheader("📥 Export Histogram Data")
                
                df_histogram = pd.DataFrame(histogram_data)
                
                # Display table preview
                st.caption("Preview Data (menampilkan 10 baris pertama):")
                st.dataframe(df_histogram.head(10), use_container_width=True)
                
                # Download CSV
                csv_buffer = io.StringIO()
                df_histogram.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Histogram Data (CSV)",
                    data=csv_data,
                    file_name="histogram_comparison.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # Calculate similarity between Plain and Decrypted
                st.markdown("---")
                st.subheader("🔍 Analisis Kesamaan Plain vs Decrypted")
                
                similarity_results = []
                
                for i, name in enumerate(color_names):
                    # Get histogram data
                    hist_plain, _ = np.histogram(image_processed[:, :, i].flatten(), 
                                                bins=256, range=[0, 256])
                    hist_decrypted, _ = np.histogram(decrypted[:, :, i].flatten(), 
                                                    bins=256, range=[0, 256])
                    
                    # Calculate similarity using correlation coefficient
                    correlation = np.corrcoef(hist_plain, hist_decrypted)[0, 1]
                    
                    # Convert correlation to percentage (normalize from [-1, 1] to [0, 100])
                    # correlation of 1.0 = 100% identical
                    similarity_percent = ((correlation + 1) / 2) * 100
                    
                    # Alternative: Calculate Mean Absolute Percentage Error
                    # MAPE untuk histogram comparison
                    diff = np.abs(hist_plain - hist_decrypted)
                    mape = (np.sum(diff) / np.sum(hist_plain)) * 100 if np.sum(hist_plain) > 0 else 0
                    # Similarity dari MAPE (100% - error)
                    similarity_mape = max(0, 100 - mape)
                    
                    # Use MAPE-based similarity as it's more intuitive
                    similarity_results.append({
                        'Channel': name,
                        'Similarity': similarity_mape,
                        'Correlation': correlation
                    })
                
                # Calculate overall similarity
                all_plain_flat = image_processed.flatten().astype(float)
                all_decrypted_flat = decrypted.flatten().astype(float)
                overall_correlation = np.corrcoef(all_plain_flat, all_decrypted_flat)[0, 1]
                overall_diff = np.abs(all_plain_flat - all_decrypted_flat)
                overall_mape = (np.sum(overall_diff) / np.sum(all_plain_flat)) * 100
                overall_similarity = max(0, 100 - overall_mape)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                for idx, (col, result) in enumerate(zip([col1, col2, col3], similarity_results)):
                    with col:
                        st.metric(
                            label=f"Kesamaan {result['Channel']} Channel",
                            value=f"{result['Similarity']:.2f}%",
                            delta="Target: 100%"
                        )
                        if result['Similarity'] >= 99.9:
                            st.success("✅ SEMPURNA")
                        elif result['Similarity'] >= 95:
                            st.success("✅ SANGAT BAIK")
                        else:
                            st.warning("⚠️ PERLU DIKECEK")
                
                # Overall similarity
                st.metric(
                    label="📊 Kesamaan Overall (Plain vs Decrypted)",
                    value=f"{overall_similarity:.2f}%",
                    delta="Target: 100%"
                )
                
                if overall_similarity >= 99.9:
                    st.success("🎉 DEKRIPSI SEMPURNA! Citra terdekripsikan dengan identik ke original")
                elif overall_similarity >= 95:
                    st.info("✅ Dekripsi sangat baik dengan tingkat kesamaan tinggi")
                else:
                    st.error("❌ Ada perbedaan signifikan antara Plain dan Decrypted")
                
                # Detailed table
                st.subheader("📋 Detail Perhitungan Kesamaan")
                
                similarity_table = []
                for result in similarity_results:
                    similarity_table.append({
                        'Channel': result['Channel'],
                        'Similarity (%)': f"{result['Similarity']:.4f}%",
                        'Correlation': f"{result['Correlation']:.6f}",
                        'Status': "✅ SEMPURNA" if result['Similarity'] >= 99.9 else "✅ BAIK" if result['Similarity'] >= 95 else "⚠️ PERLU DIKECEK"
                    })
                
                similarity_table.append({
                    'Channel': 'OVERALL',
                    'Similarity (%)': f"{overall_similarity:.4f}%",
                    'Correlation': f"{overall_correlation:.6f}",
                    'Status': "✅ SEMPURNA" if overall_similarity >= 99.9 else "✅ BAIK" if overall_similarity >= 95 else "❌ BURUK"
                })
                
                st.dataframe(pd.DataFrame(similarity_table), use_container_width=True)
                
                st.info("""
                **Penjelasan:**
                - **Histogram Plain**: Menunjukkan distribusi tidak merata (puncak-puncak)
                - **Histogram Cipher**: Menunjukkan distribusi uniform (flat) - enkripsi baik
                - **Histogram Decrypted**: Harus identik dengan Plain histogram - validasi dekripsi
                
                **Kesamaan (Similarity):**
                - Dihitung dari perbandingan histogram Plain vs Decrypted
                - 100% = Citra terdekripsikan dengan sempurna (identik)
                - ≥95% = Dekripsi berkualitas baik
                - <95% = Ada kesalahan dalam proses dekripsi
                
                **CSV Data:**
                - Kolom 'Intensity': Nilai intensitas pixel (0-255)
                - Kolom 'Plain_*', 'Cipher_*', 'Decrypted_*': Frekuensi untuk setiap channel
                - Gunakan data ini untuk analisis lebih lanjut di Excel/Python
                """)
            
            # TAB 2: KORELASI
            with tab2:
                st.subheader("🔗 Analisis Korelasi Piksel")
                
                analyzer = ImageAnalyzer()
                
                with st.spinner("Menghitung korelasi..."):
                    directions = ['horizontal', 'vertical', 'diagonal']
                    
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    corr_results = []
                    
                    for idx, direction in enumerate(directions):
                        # Plain correlation
                        corr_plain, x_plain, y_plain = analyzer.calculate_correlation(
                            image_processed, direction, 1000
                        )
                        
                        # Cipher correlation
                        corr_cipher, x_cipher, y_cipher = analyzer.calculate_correlation(
                            cipher, direction, 1000
                        )
                        
                        corr_results.append({
                            'Direction': direction.capitalize(),
                            'Plain': f"{corr_plain:.4f}",
                            'Cipher': f"{corr_cipher:.4f}",
                            'Reduction': f"{abs((corr_cipher-corr_plain)/corr_plain*100):.2f}%"
                        })
                        
                        # Plot Plain
                        axes[0, idx].scatter(x_plain, y_plain, alpha=0.3, s=5, color='blue')
                        axes[0, idx].set_title(f'Plain - {direction.capitalize()}\nr = {corr_plain:.4f}')
                        axes[0, idx].set_xlim([0, 255])
                        axes[0, idx].set_ylim([0, 255])
                        axes[0, idx].grid(alpha=0.3)
                        
                        # Plot Cipher
                        axes[1, idx].scatter(x_cipher, y_cipher, alpha=0.3, s=5, color='red')
                        axes[1, idx].set_title(f'Cipher - {direction.capitalize()}\nr = {corr_cipher:.4f}')
                        axes[1, idx].set_xlim([0, 255])
                        axes[1, idx].set_ylim([0, 255])
                        axes[1, idx].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Table
                df_corr = pd.DataFrame(corr_results)
                st.table(df_corr)
                
                # Validation
                max_corr = max([abs(float(r['Cipher'])) for r in corr_results])
                if max_corr < 0.1:
                    st.success(f"✅ PASS: Korelasi maksimal = {max_corr:.4f} < 0.1")
                else:
                    st.warning(f"⚠️ MARGINAL: Korelasi maksimal = {max_corr:.4f}")
            
            # TAB 3: NPCR & UACI
            with tab3:
                st.subheader("🛡️ Analisis Differential Attack (NPCR & UACI)")
                
                with st.spinner("Menghitung NPCR & UACI..."):
                    # Modify one pixel
                    modified = image_processed.copy().astype(np.uint16)  # Convert to uint16 first
                    modified[0, 0, 0] = (modified[0, 0, 0] + 1) % 256
                    modified = modified.astype(np.uint8)  # Convert back to uint8
                    cipher2 = acm.encrypt(modified, iterations)
                    
                    npcr, uaci = analyzer.calculate_npcr_uaci(cipher, cipher2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="NPCR (Number of Pixels Change Rate)",
                        value=f"{npcr:.4f}%",
                        delta="Target: ≥99%"
                    )
                    if npcr >= 99.0:
                        st.success("✅ PASS")
                    else:
                        st.error("❌ FAIL")
                
                with col2:
                    st.metric(
                        label="UACI (Unified Average Changing Intensity)",
                        value=f"{uaci:.4f}%",
                        delta="Target: ~33.46%"
                    )
                    if 31.0 <= uaci <= 36.0:
                        st.success("✅ PASS")
                    else:
                        st.warning("⚠️ MARGINAL")
                
                st.info("""
                **Interpretasi:**
                - **NPCR**: Persentase piksel yang berubah ketika 1 piksel plain image diubah
                - **UACI**: Rata-rata intensitas perubahan nilai piksel
                - ✅ Nilai tinggi mengindikasikan tahan terhadap differential attack
                """)
            
            # TAB 4: PSNR
            with tab4:
                st.subheader("📐 Analisis PSNR & MSE (Peak Signal-to-Noise Ratio & Mean Squared Error)")
                
                # Calculate PSNR and MSE
                psnr_plain_cipher = analyzer.calculate_psnr(image_processed, cipher)
                psnr_plain_decrypt = analyzer.calculate_psnr(image_processed, decrypted)
                
                # Calculate MSE
                mse_plain_cipher = np.mean((image_processed.astype(float) - cipher.astype(float)) ** 2)
                mse_plain_decrypt = np.mean((image_processed.astype(float) - decrypted.astype(float)) ** 2)
                
                # Display PSNR metrics
                st.subheader("📊 PSNR (Peak Signal-to-Noise Ratio)")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="PSNR (Plain vs Cipher)",
                        value=f"{psnr_plain_cipher:.2f} dB",
                        delta="Target: <10 dB"
                    )
                    if psnr_plain_cipher < 10:
                        st.success("✅ Enkripsi kuat")
                    else:
                        st.warning("⚠️ Enkripsi lemah")
                
                with col2:
                    if psnr_plain_decrypt == float('inf'):
                        st.metric(
                            label="PSNR (Plain vs Decrypt)",
                            value="∞ dB",
                            delta="Perfect!"
                        )
                        st.success("✅ Dekripsi sempurna")
                    else:
                        st.metric(
                            label="PSNR (Plain vs Decrypt)",
                            value=f"{psnr_plain_decrypt:.2f} dB"
                        )
                        st.error("❌ Dekripsi tidak sempurna")
                
                # Display MSE metrics
                st.markdown("---")
                st.subheader("📉 MSE (Mean Squared Error)")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric(
                        label="MSE (Plain vs Cipher)",
                        value=f"{mse_plain_cipher:.4f}",
                        delta="Target: Tinggi (>1000)"
                    )
                    if mse_plain_cipher > 1000:
                        st.success("✅ Enkripsi sangat berbeda")
                    elif mse_plain_cipher > 100:
                        st.info("ℹ️ Enkripsi berbeda")
                    else:
                        st.warning("⚠️ Enkripsi kurang efektif")
                
                with col4:
                    st.metric(
                        label="MSE (Plain vs Decrypt)",
                        value=f"{mse_plain_decrypt:.4f}",
                        delta="Target: 0"
                    )
                    if mse_plain_decrypt == 0:
                        st.success("✅ Dekripsi sempurna (identik)")
                    elif mse_plain_decrypt < 0.01:
                        st.success("✅ Dekripsi sangat baik")
                    else:
                        st.error("❌ Ada perbedaan dalam dekripsi")
                
                # Summary table
                st.markdown("---")
                st.subheader("📋 Tabel Ringkasan PSNR & MSE")
                
                summary_psnr_mse = pd.DataFrame({
                    'Perbandingan': ['Plain vs Cipher', 'Plain vs Decrypt'],
                    'PSNR (dB)': [
                        f"{psnr_plain_cipher:.2f}",
                        "∞" if psnr_plain_decrypt == float('inf') else f"{psnr_plain_decrypt:.2f}"
                    ],
                    'MSE': [
                        f"{mse_plain_cipher:.4f}",
                        f"{mse_plain_decrypt:.4f}"
                    ],
                    'Interpretasi': [
                        "Enkripsi kuat" if psnr_plain_cipher < 10 else "Enkripsi lemah",
                        "Dekripsi sempurna" if mse_plain_decrypt == 0 else f"Perbedaan terdeteksi"
                    ]
                })
                
                st.dataframe(summary_psnr_mse, use_container_width=True)
                
                st.info("""
                **Interpretasi:**
                
                **PSNR (Peak Signal-to-Noise Ratio):**
                - **Plain vs Cipher**: Nilai rendah (<10 dB) = citra sangat berbeda (enkripsi baik)
                - **Plain vs Decrypt**: Nilai ∞ = dekripsi sempurna (bit-perfect)
                
                **MSE (Mean Squared Error):**
                - **Plain vs Cipher**: Nilai tinggi (>1000) = enkripsi sangat efektif
                - **Plain vs Decrypt**: Nilai 0 = dekripsi identik dengan original, Nilai >0 = ada perbedaan
                """)
            
            # TAB 5: RINGKASAN
            with tab5:
                st.subheader("📈 Ringkasan Hasil Analisis")
                
                # Summary metrics
                summary_data = {
                    'Metrik': [
                        'NPCR', 'UACI', 
                        'Korelasi Horizontal', 'Korelasi Vertikal', 'Korelasi Diagonal',
                        'PSNR Plain-Cipher', 'PSNR Plain-Decrypt'
                    ],
                    'Nilai': [
                        f"{npcr:.4f}%", f"{uaci:.4f}%",
                        f"{corr_results[0]['Cipher']}", 
                        f"{corr_results[1]['Cipher']}", 
                        f"{corr_results[2]['Cipher']}",
                        f"{psnr_plain_cipher:.2f} dB",
                        "∞ dB" if psnr_plain_decrypt == float('inf') else f"{psnr_plain_decrypt:.2f} dB"
                    ],
                    'Target': [
                        '≥99%', '~33.46%',
                        '<0.1', '<0.1', '<0.1',
                        '<10 dB', '∞ dB'
                    ],
                    'Status': [
                        '✅ PASS' if npcr >= 99 else '❌ FAIL',
                        '✅ PASS' if 31 <= uaci <= 36 else '⚠️ MARGINAL',
                        '✅ PASS' if abs(float(corr_results[0]['Cipher'])) < 0.1 else '❌ FAIL',
                        '✅ PASS' if abs(float(corr_results[1]['Cipher'])) < 0.1 else '❌ FAIL',
                        '✅ PASS' if abs(float(corr_results[2]['Cipher'])) < 0.1 else '❌ FAIL',
                        '✅ PASS' if psnr_plain_cipher < 10 else '⚠️ MARGINAL',
                        '✅ PASS' if psnr_plain_decrypt == float('inf') else '❌ FAIL'
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                st.table(df_summary)
                
                # Overall assessment
                passed = df_summary['Status'].str.contains('✅').sum()
                total = len(df_summary)
                
                st.markdown("---")
                st.subheader("🎯 Penilaian Keseluruhan")
                
                if passed == total:
                    st.success(f"🎉 EXCELLENT! Semua metrik lolos ({passed}/{total})")
                    st.balloons()
                elif passed >= total * 0.7:
                    st.info(f"✅ GOOD! Sebagian besar metrik lolos ({passed}/{total})")
                else:
                    st.warning(f"⚠️ NEEDS IMPROVEMENT ({passed}/{total} lolos)")
    
    # ==================== MODE 3: TENTANG ====================
    
    elif mode == "📚 Tentang":
        st.header("📚 Tentang Sistem Ini")
        
        st.markdown("""
        ### 🔐 Arnold Cat Map (ACM)
        
        Arnold Cat Map adalah transformasi chaos yang digunakan untuk enkripsi citra digital melalui permutasi posisi piksel.
        
        **Formula Transformasi:**
        ```
        [x']   [1  b] [x]
        [y'] = [c bc+1] [y] mod N
        ```
        
        **Karakteristik:**
        - ✅ Deterministik dan reversible
        - ✅ Sensitif terhadap parameter kunci (b, c)
        - ✅ Menghasilkan pengacakan yang kompleks
        - ✅ Periode T < 3N
        
        ---
        
        ### 📊 Metrik Keamanan
        
        **1. Histogram Analysis**
        - Mengukur distribusi intensitas piksel
        - Target: Distribusi uniform pada cipher image
        
        **2. Correlation Coefficient**
        - Mengukur korelasi antar piksel bertetangga
        - Target: |r| < 0.1
        
        **3. NPCR (Number of Pixels Change Rate)**
        - Persentase piksel yang berubah saat 1 piksel input diubah
        - Target: ≥ 99%
        
        **4. UACI (Unified Average Changing Intensity)**
        - Rata-rata perubahan intensitas piksel
        - Target: ~33.46% (untuk 8-bit image)
        
        **5. PSNR (Peak Signal-to-Noise Ratio)**
        - Mengukur perbedaan antara dua citra
        - Target: <10 dB (plain-cipher), ∞ dB (plain-decrypt)
        
        ---

        
        ### 👨‍🔬 Tim Peneliti
        
        **Institut Teknologi Sumatera (ITERA)**
        - Program Studi: Informatika
        ---
    
    ### 📖 Referensi
    
    1. Rinaldi Munir (2012). "Algoritma Enkripsi Citra Digital Berbasis Chaos"
    2. Zhang et al. (2020). "Image encryption algorithm based on chaos and improved Arnold map"
    3. Ahmad et al. (2022). "Novel image encryption scheme based on Arnold cat map"
    
    ---
    
    ### 📞 Kontak
    
    Untuk pertanyaan atau kolaborasi, silakan hubungi tim peneliti.
    """)
    
    st.markdown("---")
    
    st.success("""
**Formula Transformasi:**
```
[x']   [1  b] [x]
[y'] = [c bc+1] [y] mod N
```
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🔐 <strong>Sistem Enkripsi Citra Digital - Arnold Cat Map</strong></p>
            <p>Institut Teknologi Sumatera (ITERA) © 2025</p>
            <p><small>Penelitian Kriptografi Berbasis Chaos Theory</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()