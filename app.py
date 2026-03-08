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

# ==================== ARNOLD CAT MAP ====================

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
        
        for _ in range(iterations):
            temp = np.zeros_like(encrypted)
            for x in range(N):
                for y in range(N):
                    x_new = (x + self.b * y) % N
                    y_new = (self.c * x + (self.b * self.c + 1) * y) % N
                    temp[x_new, y_new] = encrypted[x, y]
            encrypted = temp
        
        key_base = (self.b * self.c * 123 + self.b * 45 + self.c * 67) % 256
        encrypted_uint8 = np.clip(encrypted, 0, 255).astype(np.uint8)
        
        for x in range(N):
            for y in range(N):
                pos_key = (key_base + x * 17 + y * 31) % 256
                if x == 0 and y == 0:
                    prev_val = key_base
                elif y == 0:
                    prev_val = int(encrypted_uint8[x-1, N-1])
                else:
                    prev_val = int(encrypted_uint8[x, y-1])
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
        decrypted = channel.astype(np.uint8).copy()
        
        key_base = (self.b * self.c * 123 + self.b * 45 + self.c * 67) % 256
        
        for x in range(N):
            for y in range(N):
                pos_key = (key_base + x * 17 + y * 31) % 256
                if x == 0 and y == 0:
                    prev_val = key_base
                elif y == 0:
                    prev_val = int(channel[x-1, N-1])
                else:
                    prev_val = int(channel[x, y-1])
                feedback_key = (pos_key ^ prev_val) % 256
                decrypted[x, y] = (int(decrypted[x, y]) - feedback_key) % 256
        
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
        x_samples, y_samples = [], []
        
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
        if len(cipher1.shape) == 3:
            cipher1 = cipher1[:, :, 0]
            cipher2 = cipher2[:, :, 0]
        
        diff_matrix = (cipher1 != cipher2).astype(int)
        total_pixels = diff_matrix.size
        changed_pixels = np.sum(diff_matrix)
        npcr = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        
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
    h, w = image.shape[:2]
    size = min(h, w)
    return image[:size, :size] if len(image.shape) == 2 else image[:size, :size, :]

def resize_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    return cv2.resize(image, (target_size, target_size))

def get_image_download_link(img_array: np.ndarray, filename: str = "encrypted_image.png") -> str:
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="dl-btn">Download {filename}</a>'
    return href

def status_box(message: str, kind: str = "success") -> str:
    """kind: success | error | warning | info"""
    colors = {
        "success": ("#0d2e1a", "#16a34a", "#22c55e"),
        "error":   ("#2e0d0d", "#dc2626", "#ef4444"),
        "warning": ("#2e1f0d", "#d97706", "#f59e0b"),
        "info":    ("#0d1e2e", "#0369a1", "#38bdf8"),
    }
    bg, border, text = colors.get(kind, colors["info"])
    return (
        f'<div style="background:{bg};border:1px solid {border};border-left:4px solid {border};'
        f'padding:0.8rem 1rem;border-radius:6px;margin:0.5rem 0;">'
        f'<span style="color:{text};font-weight:500;font-family:\'IBM Plex Mono\',monospace;">{message}</span>'
        f'</div>'
    )

def popup_notification(message: str, kind: str = "info") -> None:
    """Show modern toast-style notifications with safe fallback."""
    if hasattr(st, "toast"):
        icons = {
            "success": ":material/check_circle:",
            "error": ":material/error:",
            "warning": ":material/warning:",
            "info": ":material/info:",
        }
        st.toast(message, icon=icons.get(kind, ":material/info:"))
    else:
        st.markdown(status_box(message, kind), unsafe_allow_html=True)

def section_header(title: str) -> str:
    return (
        f'<div style="border-bottom:1px solid #334155;padding-bottom:0.6rem;margin:1.5rem 0 1rem 0;">'
        f'<span style="color:#e2e8f0;font-size:1.1rem;font-weight:600;letter-spacing:0.04em;'
        f'font-family:\'IBM Plex Mono\',monospace;text-transform:uppercase;">{title}</span>'
        f'</div>'
    )


# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="ACM Image Encryption",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---- Google Fonts ----
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&'
        'family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">',
        unsafe_allow_html=True
    )

    # ---- Global Dark Theme CSS ----
    st.markdown("""
    <style>
    /* ===== BASE ===== */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"],
    .main {
        background-color: #0f172a !important;
        color: #cbd5e1 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* ===== ALL TEXT FORCE LIGHT ===== */
    p, span, li, label, div, h1, h2, h3, h4, h5, h6,
    .stMarkdown p, .stMarkdown span, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: #cbd5e1 !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #0a0f1e !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* ===== SIDEBAR RADIO ===== */
    [data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label {
        color: #38bdf8 !important;
    }

    /* ===== INPUTS ===== */
    .stNumberInput input,
    .stSlider .stSlider,
    .stSelectbox select,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Selectbox dropdown */
    [data-testid="stSelectbox"] > div > div {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }

    /* ===== SLIDER ===== */
    /* Slider container */
    [data-testid="stSlider"] {
        padding: 0.5rem 0 !important;
    }
    
    /* Slider label */
    [data-testid="stSlider"] label {
        color: #94a3b8 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Slider value display */
    [data-testid="stSlider"] p {
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Slider track background */
    [data-testid="stSlider"] [role="slider"] {
        background-color: transparent !important;
    }
    
    /* Slider track (unfilled part) */
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background-color: #1e293b !important;
        border-radius: 4px !important;
        height: 6px !important;
    }
    
    /* Slider track (filled part - active) */
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #0369a1 0%, #38bdf8 100%) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }
    
    /* Slider thumb (handle) */
    [data-testid="stSlider"] div[role="slider"] {
        background-color: #38bdf8 !important;
        border: 3px solid #0f172a !important;
        box-shadow: 0 0 0 2px #38bdf8, 0 2px 8px rgba(56, 189, 248, 0.4) !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
    }
    
    /* Min/Max labels */
    [data-testid="stSlider"] [data-testid="stTickBar"] {
        color: #64748b !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: #0f3460 !important;
        color: #e2e8f0 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        padding: 0.5rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #38bdf8 !important;
        color: #0a0f1e !important;
        border-color: #38bdf8 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #0369a1 !important;
        border-color: #38bdf8 !important;
        color: #f0f9ff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #38bdf8 !important;
        color: #0a0f1e !important;
    }

    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background-color: #134e4a !important;
        color: #ccfbf1 !important;
        border: 1px solid #2dd4bf !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stDownloadButton > button:hover {
        background-color: #2dd4bf !important;
        color: #0f2827 !important;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background-color: #1e293b !important;
        border: 1px dashed #334155 !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span {
        color: #94a3b8 !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0a0f1e !important;
        border-bottom: 1px solid #1e293b !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #64748b !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0f172a !important;
        padding-top: 1rem !important;
    }

    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.6rem !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #64748b !important;
        font-size: 0.78rem !important;
    }

    /* ===== DATAFRAME / TABLE ===== */
    .stDataFrame, [data-testid="stDataFrame"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    .stDataFrame thead th {
        background-color: #0f172a !important;
        color: #94a3b8 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    .stDataFrame tbody td {
        color: #cbd5e1 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        background-color: #1e293b !important;
    }
    .stDataFrame tbody tr:hover td {
        background-color: #263548 !important;
    }

    /* ===== ALERT / INFO ===== */
    [data-testid="stAlert"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
    }

    /* ===== SPINNER ===== */
    [data-testid="stSpinner"] {
        color: #38bdf8 !important;
    }

    /* ===== IMAGE CAPTION ===== */
    .stImage figcaption {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-family: 'IBM Plex Mono', monospace !important;
        text-align: center !important;
    }

    /* ===== DIVIDER ===== */
    hr {
        border-color: #1e293b !important;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f1e; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ===== CAPTION ===== */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* ===== DOWNLOAD LINK ===== */
    a.dl-btn {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        background-color: #134e4a;
        color: #ccfbf1 !important;
        text-decoration: none;
        border-radius: 4px;
        border: 1px solid #2dd4bf;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        margin-top: 0.5rem;
    }
    a.dl-btn:hover {
        background-color: #2dd4bf;
        color: #0f2827 !important;
    }

    /* ===== MATPLOTLIB FIGURE ===== */
    .stPlotlyChart, [data-testid="stImage"] {
        border: 1px solid #1e293b;
        border-radius: 6px;
    }

    /* ===== NUMBER INPUT ===== */
    [data-testid="stNumberInput"] button {
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
    <div style="padding:2rem 0 1.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#38bdf8;
                    letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;">
            ITERA &mdash; Institut Teknologi Sumatera &mdash; 2026
        </div>
        <h1 style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;
                   color:#e2e8f0;margin:0 0 0.3rem 0;letter-spacing:0.01em;">
            Image Encryption System
        </h1>
        <p style="color:#64748b;font-size:0.95rem;margin:0;font-family:'IBM Plex Sans',sans-serif;">
            Arnold Cat Map &mdash; Chaos-Based Digital Image Cryptography
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    st.sidebar.markdown("""
    <div style="padding:1rem 0 0.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#38bdf8;
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;">System Control</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:#e2e8f0;">
            ACM Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "Mode Operasi",
        ["Enkripsi / Dekripsi", "Analisis Keamanan", "Tentang Sistem"]
    )

    st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#38bdf8;
                letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;">Secret Key</div>
    """, unsafe_allow_html=True)

    param_b = st.sidebar.number_input("Parameter b", min_value=1, max_value=100, value=2)
    param_c = st.sidebar.number_input("Parameter c", min_value=1, max_value=100, value=2)
    iterations = st.sidebar.slider("Iterasi (m)", min_value=1, max_value=50, value=5)

    st.sidebar.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="background:#0f1e2e;border:1px solid #1e3a5f;border-radius:6px;padding:0.8rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#38bdf8;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;">Perhatian</div>
        <p style="color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.6;">
            Parameter b, c, dan m bersifat rahasia. Simpan sebelum memulai enkripsi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    acm = ArnoldCatMap(b=param_b, c=param_c)

    # ==================== MODE 1: ENKRIPSI/DEKRIPSI ====================

    if "Enkripsi" in mode:
        st.markdown(section_header("Enkripsi dan Dekripsi Citra"), unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:#38bdf8;'
                        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">Upload Citra</div>',
                        unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Pilih file citra (JPG, PNG, TIFF)",
                type=['jpg', 'jpeg', 'png', 'tiff'],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                st.image(image, caption="Citra Original", use_container_width=True)

                st.markdown(
                    f'<div style="background:#1e293b;border:1px solid #334155;border-radius:4px;'
                    f'padding:0.6rem 0.8rem;margin-top:0.5rem;">'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;color:#94a3b8;">'
                    f'Dimensi: <span style="color:#e2e8f0;">{image.shape[1]} x {image.shape[0]}</span>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;Mode: <span style="color:#e2e8f0;">{"RGB" if len(image.shape)==3 else "Grayscale"}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True
                )

                st.markdown('<div style="margin-top:1rem;font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;'
                            'color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;">'
                            'Preprocessing</div>', unsafe_allow_html=True)

                target_size = st.selectbox("Resize ke ukuran", [256, 512, 1024], index=1)

                if st.button("Proses Gambar", type="primary"):
                    with st.spinner("Memproses citra..."):
                        image_square = make_square(image)
                        image_processed = resize_image(image_square, target_size)
                        st.session_state['image_processed'] = image_processed
                    st.markdown(status_box(f"Citra berhasil diproses ke {target_size} x {target_size}", "success"),
                                unsafe_allow_html=True)

        with col2:
            if 'image_processed' in st.session_state:
                image_processed = st.session_state['image_processed']

                st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:#38bdf8;'
                            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">Enkripsi</div>',
                            unsafe_allow_html=True)

                if st.button("Enkripsi Citra", type="primary", key="encrypt_btn"):
                    with st.spinner("Mengenkripsi citra..."):
                        start_time = time.time()
                        cipher = acm.encrypt(image_processed, iterations)
                        encrypt_time = (time.time() - start_time) * 1000

                    st.session_state['cipher'] = cipher
                    st.session_state['encrypt_time'] = encrypt_time
                    st.image(cipher, caption="Citra Terenkripsi", use_container_width=True)
                    st.markdown(status_box(f"Enkripsi selesai dalam {encrypt_time:.2f} ms", "success"),
                                unsafe_allow_html=True)
                    st.markdown(get_image_download_link(cipher, "encrypted_image.png"), unsafe_allow_html=True)

                st.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>",
                            unsafe_allow_html=True)

                st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:#38bdf8;'
                            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">Dekripsi</div>',
                            unsafe_allow_html=True)

                if 'cipher' in st.session_state:
                    cipher = st.session_state['cipher']

                    if st.button("Dekripsi Citra", type="secondary", key="decrypt_btn"):
                        with st.spinner("Mendekripsi citra..."):
                            start_time = time.time()
                            decrypted = acm.decrypt(cipher, iterations)
                            decrypt_time = (time.time() - start_time) * 1000

                        st.image(decrypted, caption="Citra Terdekripsi", use_container_width=True)
                        identical = np.array_equal(image_processed, decrypted)

                        if identical:
                            st.markdown(status_box(f"Dekripsi sempurna. Waktu: {decrypt_time:.2f} ms", "success"),
                                        unsafe_allow_html=True)
                            popup_notification("Dekripsi berhasil dan hasil siap diunduh.", "success")
                        else:
                            st.markdown(status_box("Dekripsi tidak sempurna. Periksa parameter kunci.", "error"),
                                        unsafe_allow_html=True)

                        st.markdown(get_image_download_link(decrypted, "decrypted_image.png"),
                                    unsafe_allow_html=True)

    # ==================== MODE 2: ANALISIS KEAMANAN ====================

    elif "Analisis" in mode:
        st.markdown(section_header("Analisis Keamanan Enkripsi"), unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload citra untuk dianalisis",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_square = make_square(image)
            image_processed = resize_image(image_square, 512)

            with st.spinner("Mengenkripsi citra untuk analisis..."):
                cipher = acm.encrypt(image_processed, iterations)
                decrypted = acm.decrypt(cipher, iterations)

            st.markdown(status_box("Enkripsi selesai. Memulai analisis keamanan.", "success"),
                        unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                st.image(image_processed, caption="Plain Image", use_container_width=True)
            with col2:
                st.image(cipher, caption="Cipher Image", use_container_width=True)
            with col3:
                st.image(decrypted, caption="Decrypted Image", use_container_width=True)

            st.markdown("<div style='border-top:1px solid #1e293b;margin:1.5rem 0;'></div>",
                        unsafe_allow_html=True)

            analyzer = ImageAnalyzer()

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "HISTOGRAM", "KORELASI", "NPCR & UACI", "PSNR & MSE", "RINGKASAN"
            ])

            # --- Matplotlib dark style ---
            plt.rcParams.update({
                'figure.facecolor': '#0f172a',
                'axes.facecolor':   '#1e293b',
                'axes.edgecolor':   '#334155',
                'axes.labelcolor':  '#94a3b8',
                'xtick.color':      '#64748b',
                'ytick.color':      '#64748b',
                'text.color':       '#e2e8f0',
                'grid.color':       '#334155',
                'grid.alpha':       0.4,
                'axes.titlecolor':  '#e2e8f0',
                'font.family':      'monospace',
                'axes.titlesize':   9,
                'axes.labelsize':   8,
            })

            # TAB 1: HISTOGRAM
            with tab1:
                st.markdown(section_header("Analisis Histogram"), unsafe_allow_html=True)

                with st.spinner("Menghitung histogram..."):
                    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
                    fig.patch.set_facecolor('#0f172a')

                    colors_plot = ['#ef4444', '#22c55e', '#3b82f6']
                    color_names = ['Red', 'Green', 'Blue']
                    histogram_data = {'Intensity': list(range(256))}

                    for i, (cp, name) in enumerate(zip(colors_plot, color_names)):
                        for row_idx, (img_arr, label) in enumerate([
                            (image_processed, 'Plain'),
                            (cipher, 'Cipher'),
                            (decrypted, 'Decrypted')
                        ]):
                            hist_vals, _ = np.histogram(img_arr[:, :, i].flatten(), bins=256, range=[0, 256])
                            axes[row_idx, i].bar(range(256), hist_vals, color=cp, alpha=0.75, width=1.0)
                            axes[row_idx, i].set_title(f'{label} — {name}', fontsize=9, color='#94a3b8')
                            axes[row_idx, i].grid(alpha=0.3)
                            if row_idx == 2:
                                axes[row_idx, i].set_xlabel('Intensitas', fontsize=8, color='#64748b')
                            if i == 0:
                                axes[row_idx, i].set_ylabel('Frekuensi', fontsize=8, color='#64748b')
                            histogram_data[f'{label}_{name}'] = hist_vals.tolist()

                    plt.tight_layout(pad=1.5)
                    st.pyplot(fig)
                    plt.close()

                st.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>",
                            unsafe_allow_html=True)
                st.markdown(section_header("Export Data Histogram"), unsafe_allow_html=True)

                df_histogram = pd.DataFrame(histogram_data)
                st.caption("Preview — 10 baris pertama")
                st.dataframe(df_histogram.head(10), use_container_width=True)

                csv_buffer = io.StringIO()
                df_histogram.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Histogram Data (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="histogram_comparison.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

                st.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>",
                            unsafe_allow_html=True)
                st.markdown(section_header("Kesamaan Plain vs Decrypted"), unsafe_allow_html=True)

                similarity_results = []
                for i, name in enumerate(color_names):
                    hp, _ = np.histogram(image_processed[:, :, i].flatten(), bins=256, range=[0, 256])
                    hd, _ = np.histogram(decrypted[:, :, i].flatten(), bins=256, range=[0, 256])
                    corr_val = np.corrcoef(hp, hd)[0, 1]
                    diff = np.abs(hp - hd)
                    mape = (np.sum(diff) / np.sum(hp)) * 100 if np.sum(hp) > 0 else 0
                    sim = max(0, 100 - mape)
                    similarity_results.append({'Channel': name, 'Similarity': sim, 'Correlation': corr_val})

                all_pf = image_processed.flatten().astype(float)
                all_df_ = decrypted.flatten().astype(float)
                overall_corr = np.corrcoef(all_pf, all_df_)[0, 1]
                overall_diff = np.abs(all_pf - all_df_)
                overall_mape = (np.sum(overall_diff) / np.sum(all_pf)) * 100
                overall_sim = max(0, 100 - overall_mape)

                col_r, col_g, col_b = st.columns(3)
                for col_w, res in zip([col_r, col_g, col_b], similarity_results):
                    with col_w:
                        st.metric(
                            label=f"Kesamaan {res['Channel']}",
                            value=f"{res['Similarity']:.2f}%"
                        )
                        kind = "success" if res['Similarity'] >= 99.9 else "info" if res['Similarity'] >= 95 else "warning"
                        label_txt = "SEMPURNA" if res['Similarity'] >= 99.9 else "BAIK" if res['Similarity'] >= 95 else "PERLU DIKECEK"
                        st.markdown(status_box(label_txt, kind), unsafe_allow_html=True)

                st.metric(label="Kesamaan Overall", value=f"{overall_sim:.2f}%")

                sim_table_data = []
                for res in similarity_results:
                    sim_table_data.append({
                        'Channel': res['Channel'],
                        'Similarity (%)': f"{res['Similarity']:.4f}",
                        'Correlation': f"{res['Correlation']:.6f}",
                        'Status': "SEMPURNA" if res['Similarity'] >= 99.9 else "BAIK" if res['Similarity'] >= 95 else "PERLU DIKECEK"
                    })
                sim_table_data.append({
                    'Channel': 'OVERALL',
                    'Similarity (%)': f"{overall_sim:.4f}",
                    'Correlation': f"{overall_corr:.6f}",
                    'Status': "SEMPURNA" if overall_sim >= 99.9 else "BAIK" if overall_sim >= 95 else "GAGAL"
                })
                st.dataframe(pd.DataFrame(sim_table_data), use_container_width=True)

            # TAB 2: KORELASI
            with tab2:
                st.markdown(section_header("Analisis Korelasi Piksel"), unsafe_allow_html=True)

                with st.spinner("Menghitung korelasi..."):
                    directions = ['horizontal', 'vertical', 'diagonal']
                    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                    fig.patch.set_facecolor('#0f172a')
                    corr_results = []

                    for idx, direction in enumerate(directions):
                        corr_plain, xp, yp = analyzer.calculate_correlation(image_processed, direction, 1000)
                        corr_cipher, xc, yc = analyzer.calculate_correlation(cipher, direction, 1000)

                        corr_results.append({
                            'Direction': direction.capitalize(),
                            'Plain': f"{corr_plain:.4f}",
                            'Cipher': f"{corr_cipher:.4f}",
                        })

                        axes[0, idx].scatter(xp, yp, alpha=0.25, s=4, color='#38bdf8')
                        axes[0, idx].set_title(f'Plain — {direction.capitalize()}\nr = {corr_plain:.4f}')
                        axes[0, idx].set_xlim([0, 255]); axes[0, idx].set_ylim([0, 255])
                        axes[0, idx].grid(alpha=0.3)

                        axes[1, idx].scatter(xc, yc, alpha=0.25, s=4, color='#f97316')
                        axes[1, idx].set_title(f'Cipher — {direction.capitalize()}\nr = {corr_cipher:.4f}')
                        axes[1, idx].set_xlim([0, 255]); axes[1, idx].set_ylim([0, 255])
                        axes[1, idx].grid(alpha=0.3)

                    plt.tight_layout(pad=1.5)
                    st.pyplot(fig)
                    plt.close()

                st.dataframe(pd.DataFrame(corr_results), use_container_width=True)

                max_corr = max([abs(float(r['Cipher'])) for r in corr_results])
                if max_corr < 0.1:
                    st.markdown(status_box(f"PASS — Korelasi maksimal cipher: {max_corr:.4f} (target < 0.1)", "success"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(status_box(f"MARGINAL — Korelasi maksimal cipher: {max_corr:.4f}", "warning"),
                                unsafe_allow_html=True)

            # TAB 3: NPCR & UACI
            with tab3:
                st.markdown(section_header("Differential Attack — NPCR & UACI"), unsafe_allow_html=True)

                with st.spinner("Menghitung NPCR & UACI..."):
                    modified = image_processed.copy().astype(np.uint16)
                    modified[0, 0, 0] = (modified[0, 0, 0] + 1) % 256
                    modified = modified.astype(np.uint8)
                    cipher2 = acm.encrypt(modified, iterations)
                    npcr, uaci = analyzer.calculate_npcr_uaci(cipher, cipher2)

                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.metric(label="NPCR — Number of Pixels Change Rate", value=f"{npcr:.4f}%", delta="Target: >= 99%")
                    kind = "success" if npcr >= 99.0 else "error"
                    st.markdown(status_box("PASS" if npcr >= 99.0 else "FAIL", kind), unsafe_allow_html=True)
                with col2:
                    st.metric(label="UACI — Unified Average Changing Intensity", value=f"{uaci:.4f}%", delta="Target: ~33.46%")
                    kind = "success" if 31.0 <= uaci <= 36.0 else "warning"
                    st.markdown(status_box("PASS" if 31.0 <= uaci <= 36.0 else "MARGINAL", kind), unsafe_allow_html=True)

                st.markdown("""
                <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:1rem;margin-top:1rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#38bdf8;
                                letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Interpretasi</div>
                    <p style="color:#94a3b8;font-size:0.87rem;margin:0.3rem 0;line-height:1.6;">
                        NPCR: persentase piksel yang berubah ketika 1 piksel input dimodifikasi.
                    </p>
                    <p style="color:#94a3b8;font-size:0.87rem;margin:0.3rem 0;line-height:1.6;">
                        UACI: rata-rata perubahan intensitas piksel antara dua ciphertext.
                    </p>
                    <p style="color:#94a3b8;font-size:0.87rem;margin:0.3rem 0;line-height:1.6;">
                        Nilai tinggi mengindikasikan ketahanan terhadap differential attack.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # TAB 4: PSNR & MSE
            with tab4:
                st.markdown(section_header("PSNR & MSE"), unsafe_allow_html=True)

                psnr_pc = analyzer.calculate_psnr(image_processed, cipher)
                psnr_pd = analyzer.calculate_psnr(image_processed, decrypted)
                mse_pc  = np.mean((image_processed.astype(float) - cipher.astype(float)) ** 2)
                mse_pd  = np.mean((image_processed.astype(float) - decrypted.astype(float)) ** 2)

                st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#38bdf8;'
                            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">PSNR</div>',
                            unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.metric("PSNR Plain vs Cipher", f"{psnr_pc:.2f} dB", delta="Target: < 10 dB")
                    st.markdown(status_box("Enkripsi kuat" if psnr_pc < 10 else "Enkripsi lemah",
                                           "success" if psnr_pc < 10 else "warning"), unsafe_allow_html=True)
                with col2:
                    val = "Infinity" if psnr_pd == float('inf') else f"{psnr_pd:.2f} dB"
                    st.metric("PSNR Plain vs Decrypt", val, delta="Target: Infinity")
                    st.markdown(status_box("Dekripsi sempurna" if psnr_pd == float('inf') else "Terdapat perbedaan",
                                           "success" if psnr_pd == float('inf') else "error"), unsafe_allow_html=True)

                st.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>",
                            unsafe_allow_html=True)
                st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#38bdf8;'
                            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">MSE</div>',
                            unsafe_allow_html=True)
                col3, col4 = st.columns(2, gap="large")
                with col3:
                    st.metric("MSE Plain vs Cipher", f"{mse_pc:.4f}", delta="Target: > 1000")
                    kind = "success" if mse_pc > 1000 else "info" if mse_pc > 100 else "warning"
                    msg  = "Enkripsi sangat efektif" if mse_pc > 1000 else "Enkripsi berbeda" if mse_pc > 100 else "Enkripsi kurang efektif"
                    st.markdown(status_box(msg, kind), unsafe_allow_html=True)
                with col4:
                    st.metric("MSE Plain vs Decrypt", f"{mse_pd:.4f}", delta="Target: 0")
                    kind = "success" if mse_pd == 0 else "error"
                    msg  = "Identik dengan original" if mse_pd == 0 else f"Perbedaan terdeteksi"
                    st.markdown(status_box(msg, kind), unsafe_allow_html=True)

                st.markdown("<div style='border-top:1px solid #1e293b;margin:1rem 0;'></div>",
                            unsafe_allow_html=True)
                df_psnr = pd.DataFrame({
                    'Perbandingan':   ['Plain vs Cipher', 'Plain vs Decrypt'],
                    'PSNR (dB)':      [f"{psnr_pc:.2f}", "Infinity" if psnr_pd == float('inf') else f"{psnr_pd:.2f}"],
                    'MSE':            [f"{mse_pc:.4f}", f"{mse_pd:.4f}"],
                    'Interpretasi':   [
                        "Enkripsi kuat" if psnr_pc < 10 else "Enkripsi lemah",
                        "Dekripsi sempurna" if mse_pd == 0 else "Ada perbedaan"
                    ]
                })
                st.dataframe(df_psnr, use_container_width=True)

            # TAB 5: RINGKASAN
            with tab5:
                st.markdown(section_header("Ringkasan Hasil Analisis"), unsafe_allow_html=True)

                summary_data = {
                    'Metrik': [
                        'NPCR', 'UACI',
                        'Korelasi Horizontal', 'Korelasi Vertikal', 'Korelasi Diagonal',
                        'PSNR Plain–Cipher', 'PSNR Plain–Decrypt'
                    ],
                    'Nilai': [
                        f"{npcr:.4f}%", f"{uaci:.4f}%",
                        corr_results[0]['Cipher'],
                        corr_results[1]['Cipher'],
                        corr_results[2]['Cipher'],
                        f"{psnr_pc:.2f} dB",
                        "Infinity" if psnr_pd == float('inf') else f"{psnr_pd:.2f} dB"
                    ],
                    'Target': ['>= 99%', '~33.46%', '< 0.1', '< 0.1', '< 0.1', '< 10 dB', 'Infinity'],
                    'Status': [
                        'PASS' if npcr >= 99 else 'FAIL',
                        'PASS' if 31 <= uaci <= 36 else 'MARGINAL',
                        'PASS' if abs(float(corr_results[0]['Cipher'])) < 0.1 else 'FAIL',
                        'PASS' if abs(float(corr_results[1]['Cipher'])) < 0.1 else 'FAIL',
                        'PASS' if abs(float(corr_results[2]['Cipher'])) < 0.1 else 'FAIL',
                        'PASS' if psnr_pc < 10 else 'MARGINAL',
                        'PASS' if psnr_pd == float('inf') else 'FAIL'
                    ]
                }

                df_sum = pd.DataFrame(summary_data)
                st.dataframe(df_sum, use_container_width=True)

                passed = df_sum['Status'].str.contains('PASS').sum()
                total  = len(df_sum)
                ratio  = passed / total

                st.markdown("<div style='border-top:1px solid #1e293b;margin:1.5rem 0 1rem 0;'></div>",
                            unsafe_allow_html=True)
                st.markdown(section_header("Penilaian Keseluruhan"), unsafe_allow_html=True)

                if ratio == 1.0:
                    verdict, kind = f"EXCELLENT — Semua metrik lolos ({passed}/{total})", "success"
                elif ratio >= 0.7:
                    verdict, kind = f"GOOD — Sebagian besar metrik lolos ({passed}/{total})", "info"
                else:
                    verdict, kind = f"NEEDS IMPROVEMENT — {passed}/{total} metrik lolos", "warning"

                st.markdown(status_box(verdict, kind), unsafe_allow_html=True)

                if ratio == 1.0:
                    popup_notification("Analisis selesai, seluruh metrik utama terpenuhi.", "success")

    # ==================== MODE 3: TENTANG ====================

    elif "Tentang" in mode:
        st.markdown(section_header("Tentang Sistem"), unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:1.2rem;margin-bottom:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#38bdf8;
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">Arnold Cat Map (ACM)</div>
            <p style="color:#94a3b8;font-size:0.9rem;line-height:1.7;margin:0 0 0.8rem 0;">
                Arnold Cat Map adalah transformasi chaos deterministik yang digunakan untuk enkripsi citra digital
                melalui permutasi posisi piksel. Sistem ini menggabungkan ACM dengan Feedback Diffusion Layer
                untuk meningkatkan sensitivitas terhadap perubahan kunci.
            </p>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#64748b;
                        margin-bottom:0.4rem;">Karakteristik:</div>
            <ul style="color:#94a3b8;font-size:0.87rem;line-height:1.8;margin:0;padding-left:1.2rem;">
                <li>Deterministik dan reversible</li>
                <li>Sensitif terhadap parameter kunci (b, c)</li>
                <li>Menghasilkan pengacakan kompleks pada citra</li>
                <li>Periode T &lt; 3N untuk matriks N x N</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0f1e2e;border:1px solid #1e3a5f;border-radius:6px;padding:1rem;margin-bottom:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#38bdf8;
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;">Formula Transformasi</div>
        """, unsafe_allow_html=True)
        st.code("[x']   [ 1    b  ] [x]\n[y'] = [ c   bc+1 ] [y]  mod N", language=None)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(section_header("Metrik Keamanan"), unsafe_allow_html=True)

        metrics = [
            ("Histogram Analysis",              "Distribusi intensitas piksel",                          "Distribusi uniform pada cipher image"),
            ("Correlation Coefficient",          "Korelasi antar piksel bertetangga",                    "|r| < 0.1"),
            ("NPCR",                             "Perubahan piksel saat 1 piksel input dimodifikasi",    ">= 99%"),
            ("UACI",                             "Rata-rata perubahan intensitas piksel",                "~33.46%"),
            ("PSNR",                             "Perbedaan kuantitatif antara dua citra",               "< 10 dB (plain-cipher), Infinity (plain-decrypt)"),
        ]
        for title, desc, target in metrics:
            st.markdown(
                f'<div style="background:#1e293b;border:1px solid #334155;border-left:3px solid #38bdf8;'
                f'border-radius:0 4px 4px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;color:#e2e8f0;'
                f'font-weight:600;margin-bottom:0.2rem;">{title}</div>'
                f'<div style="color:#94a3b8;font-size:0.85rem;">{desc}</div>'
                f'<div style="color:#64748b;font-size:0.8rem;margin-top:0.2rem;">Target: {target}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(section_header("Referensi"), unsafe_allow_html=True)
        refs = [
            ("Rinaldi Munir (2012)", "Algoritma Enkripsi Citra Digital Berbasis Chaos"),
            ("Zhang et al. (2020)", "Image encryption algorithm based on chaos and improved Arnold map"),
            ("Ahmad et al. (2022)", "Novel image encryption scheme based on Arnold cat map"),
        ]
        for author, title in refs:
            st.markdown(
                f'<div style="background:#1e293b;border:1px solid #334155;border-radius:4px;'
                f'padding:0.7rem 1rem;margin-bottom:0.4rem;">'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:#38bdf8;">{author}</span>'
                f'<span style="color:#94a3b8;font-size:0.85rem;margin-left:0.8rem;">{title}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(section_header("Institusi"), unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#e2e8f0;font-weight:600;">
                Institut Teknologi Sumatera (ITERA)
            </div>
            <div style="color:#94a3b8;font-size:0.87rem;margin-top:0.3rem;">
                Program Studi Teknik Informatika &mdash; Penelitian Kriptografi Berbasis Chaos Theory &mdash; 2026
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Footer ----
    st.markdown("<div style='border-top:1px solid #1e293b;margin:2rem 0 0.5rem 0;'></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.8rem 0;">
        <div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#334155;">
                ACM Image Encryption System
            </span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#334155;
                          margin-left:1rem;">ITERA 2026</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#1e293b;">
            Chaos-Based Digital Image Cryptography
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
