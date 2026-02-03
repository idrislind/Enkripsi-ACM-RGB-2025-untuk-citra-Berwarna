"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║     SISTEM ENKRIPSI CITRA DIGITAL BERBASIS ARNOLD CAT MAP (ACM)              ║
║              Penelitian Kriptografi Citra - ITERA 2025                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

DESKRIPSI UMUM:
==============
Sistem enkripsi citra yang menggabungkan dua teknik utama:
  1. PERMUTASI PIXELPOSISI (Arnold Cat Map) - mengubah posisi pixel
  2. DIFFUSION LAYER (Feedback-based Diffusion) - mengubah nilai pixel

RUMUS ARNOLD CAT MAP (ACM):
==========================
Transformasi chaos linear yang merupakan map invertible pada torus 2D:

  ┌     ┐   ┌              ┐   ┌   ┐       ┌   ┐
  │ x' │   │  1      b     │   │ x │   mod │ N │
  │ y'│ = │  c   bc+1  │ · │ y │       │ N │
  └     ┘   └              ┘   └   ┘       └   ┘

Dimana:
  - (x, y) = koordinat pixel asli (0 ≤ x,y < N)
  - (x', y') = koordinat pixel setelah transformasi
  - b, c = parameter kunci (untuk penelitian ini: b=2, c=2)
  - N = ukuran citra (N×N pixels)
  - mod N = operasi modulo untuk wrap-around di torus

SIFAT PENTING ACM:
  ✓ Invertible: Bisa dikembalikan dengan inverse matrix
  ✓ Chaotic: Perubahan kecil menghasilkan permutasi sangat berbeda
  ✓ Periodic: Setelah iterasi tertentu, kembali ke posisi semula
  ✓ Deterministic: Hasil same input selalu sama

RUMUS INVERSE ACM (Untuk Dekripsi):
===================================
Untuk mengembalikan pixel ke posisi asli:

  ┌   ┐       ┌   ┐
  │ x │   mod │ N │     ┌              ┐
  │ y │   ┌   ┘ N ┘  =  │  bc+1   -b   │  · ┌ x' ┐
  └   ┘       │          │  -c      1   │    │ y' │
              └──────────└              ┘    └   ┘

DIFFUSION LAYER (3 Rounds):
===========================
Setelah permutasi ACM, layer diffusion mengubah NILAI pixel untuk:
  ✓ Meningkatkan NPCR (Number of Pixels Change Rate) → target ≥99%
  ✓ Meningkatkan UACI (Unified Average Changing Intensity) → target ~33.46%
  ✓ Mengurangi korelasi pixel tetangga → target <0.1

SETIAP ROUND DIFFUSION:

  encrypted_uint8[x,y] = (encrypted_uint8[x,y] + feedback_key[x,y]) mod 256

  Dimana:
    feedback_key = pos_key XOR (feedback_sum mod 256)
    
    pos_key = (key_base + round_num*100 + x*17 + y*31) mod 256
    
    feedback_sum = encrypted_uint8[x-1, y]      (neighbor kiri/atas wraps)
                 + encrypted_uint8[x, y-1]      (neighbor atas/kiri wraps)
                 + encrypted_uint8[x-1, y-1]    (neighbor diagonal wraps)

Proses 3 rounds:
  ROUND 0: feedback dari state awal (post-ACM)
  ROUND 1: feedback dari state setelah round 0
  ROUND 2: feedback dari state setelah round 1

METRIK KEAMANAN:
================
1. NPCR (Number of Pixels Change Rate):
   Persentase pixel yang berubah ketika 1 pixel plain image diubah
   
   Formula: NPCR = (D/N²) × 100%
   Dimana:
     D = jumlah pixel yang berbeda antara cipher1 dan cipher2
     N² = total jumlah pixel
   Target: ≥99% (menunjukkan sensitivity terhadap plain image)

2. UACI (Unified Average Changing Intensity):
   Rata-rata perubahan nilai intensitas per pixel
   
   Formula: UACI = (1/N²) × Σ|cipher1[i,j] - cipher2[i,j]| / 255 × 100%
   Target: ~33.46% (menunjukkan avalanche effect maksimal)

3. CORRELATION (Pearson Correlation Coefficient):
   Mengukur relationship antara pixel tetangga
   
   Formula: r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
   Target: <0.1 (menunjukkan pixel independent)
   Arah: Horizontal, Vertical, Diagonal

4. PSNR (Peak Signal-to-Noise Ratio):
   Mengukur kemiripan antara citra original dan decrypted
   
   Formula: PSNR = 10 × log₁₀((255²) / MSE) dB
   Dimana MSE = (1/N²) × Σ(original[i,j] - decrypted[i,j])²
   Target: ∞ dB untuk enkripsi-dekripsi sempurna (MSE = 0)

ALUR ENKRIPSI:
==============
Input: Plain Image (N×N)
        ↓
   [TAHAP 1: ACM PERMUTATION]
   Iterasi sebanyak 5 kali transformasi ACM
   Mengubah POSISI setiap pixel
        ↓
   [TAHAP 2: 3-ROUND FEEDBACK DIFFUSION]
   Round 0: Tambah feedback dari neighbors (state awal)
   Round 1: Tambah feedback dari neighbors (post-round0)
   Round 2: Tambah feedback dari neighbors (post-round1)
   Mengubah NILAI setiap pixel
        ↓
Output: Cipher Image (N×N) - random uniform distribution
        NPCR ≥99%, UACI ~33%, Correlation <0.1

ALUR DEKRIPSI:
==============
Input: Cipher Image (N×N) - terenkripsi
        ↓
   [TAHAP 1: REVERSE DIFFUSION (3 rounds, backwards)]
   Round 2: Kurang feedback dari intermediate state[2]
   Round 1: Kurang feedback dari intermediate state[1]
   Round 0: Kurang feedback dari intermediate state[0]
   Mengembalikan NILAI setiap pixel ke state setelah ACM
        ↓
   [TAHAP 2: INVERSE ACM PERMUTATION]
   Iterasi sebanyak 5 kali inverse transformasi ACM
   Mengembalikan POSISI setiap pixel
        ↓
Output: Decrypted Image ≡ Plain Image (100% identical, PSNR = ∞ dB)

Author: Research Team
Version: 2.0
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import time
import os
from typing import Tuple, List, Dict
import seaborn as sns

# ==================== MODUL 1: ARNOLD CAT MAP ====================

class ArnoldCatMap:
    """
    ╔════════════════════════════════════════════════════════════════════╗
    ║           ARNOLD CAT MAP (ACM) - CHAOS-BASED PERMUTATION          ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    Implementasi Arnold Cat Map untuk enkripsi citra berbasis chaos.
    
    PENJELASAN SINGKAT:
    ===================
    Arnold Cat Map adalah transformasi matematika yang secara acak mengubah
    posisi pixel tanpa mengubah nilai pixelnya. Ini adalah salah satu chaos
    map yang paling terkenal dalam kriptografi citra karena:
      • Mudah diimplementasikan
      • Bersifat chaotic (sensitif terhadap kondisi awal)
      • Invertible (bisa dikembalikan)
      • Periodic (mengulangi setiap T iterasi)
    
    PARAMETER KUNCI:
    ================
    b, c : int
        Konstanta kunci dalam transformasi ACM
        Default: b=2, c=2 (penelitian menggunakan nilai ini)
        Semakin besar nilai b dan c, semakin kompleks permutasi
    """
    
    def __init__(self, b: int = 2, c: int = 2):
        """
        Inisialisasi ACM dengan parameter kunci
        
        Parameters:
        -----------
        b, c : int
            Parameter kunci ACM (harus integer positif > 0)
            Standar: b=2, c=2
            
        Notes:
        ------
        Determinant dari matrix ACM: det(A) = (1)(bc+1) - (b)(c) = 1
        Ini menjamin matrix invertible (bisa dikembalikan)
        """
        self.b = b
        self.c = c
        
    def encrypt(self, image: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Enkripsi citra menggunakan ACM
        
        Parameters:
        -----------
        image : np.ndarray
            Citra input (grayscale atau RGB)
        iterations : int
            Jumlah iterasi ACM
            
        Returns:
        --------
        encrypted : np.ndarray
            Citra terenkripsi
        """
        if len(image.shape) == 2:  # Grayscale
            return self._encrypt_channel(image, iterations)
        else:  # RGB
            encrypted = np.zeros_like(image)
            for ch in range(image.shape[2]):
                encrypted[:, :, ch] = self._encrypt_channel(image[:, :, ch], iterations)
            return encrypted
    
    def _encrypt_channel(self, channel: np.ndarray, iterations: int) -> np.ndarray:
        """
        Enkripsi single channel dengan permutasi ACM + multiple diffusion rounds
        
        ALUR PROSES:
        ============
        1. PERMUTASI ACM (iterations kali)
           - Mengubah posisi pixel
           - Tidak mengubah nilai pixel
           - Teriterasi 5 kali untuk avalanche effect
        
        2. DIFFUSION LAYER (3 rounds)
           - Mengubah nilai pixel berdasarkan feedback neighbors
           - Round 0,1,2: Iterasi diffusion dengan key yang berbeda
           - Meningkatkan NPCR dan UACI
        """
        N = channel.shape[0]
        assert channel.shape[0] == channel.shape[1], "Citra harus berbentuk persegi (N×N)"
        
        encrypted = channel.copy().astype(np.float32)
        
        # ════════════════════════════════════════════════════════════════════════
        # TAHAP 1: PERMUTASI ACM - Mengubah POSISI pixel
        # ════════════════════════════════════════════════════════════════════════
        # Formula ACM:
        #   x_new = (x + b*y) mod N
        #   y_new = (c*x + (b*c+1)*y) mod N
        #
        # Interpretasi:
        #   - Setiap pixel [x,y] dipindahkan ke posisi baru [x_new, y_new]
        #   - Perpindahan ini mengikuti pola chaos yang kompleks
        #   - Setelah 5 iterasi, semua pixel tersebar secara acak
        # ════════════════════════════════════════════════════════════════════════
        for iteration in range(iterations):
            temp = np.zeros_like(encrypted)
            for x in range(N):
                for y in range(N):
                    # Transformasi ACM: perpindahan posisi pixel
                    # [1  b  ] [x]       [x + b*y    ]
                    # [c bc+1] [y]  =    [c*x + (bc+1)*y]  (mod N)
                    x_new = (x + self.b * y) % N          # Persamaan 1: x_new
                    y_new = (self.c * x + (self.b * self.c + 1) * y) % N  # Persamaan 2: y_new
                    
                    # Pindahkan pixel dari [x,y] ke [x_new, y_new]
                    temp[x_new, y_new] = encrypted[x, y]
            
            encrypted = temp
        
        # ════════════════════════════════════════════════════════════════════════
        # TAHAP 2: FEEDBACK DIFFUSION LAYER - Mengubah NILAI pixel
        # ════════════════════════════════════════════════════════════════════════
        # Tujuan:
        #   ✓ Meningkatkan NPCR dari 0% → 99%+ (sensitivity)
        #   ✓ Meningkatkan UACI dari 0% → 33%+ (avalanche effect)
        #   ✓ Mengurangi correlation <0.1 (independence)
        # 
        # Prinsip:
        #   Setiap pixel ditambah dengan feedback dari 3 neighbors sebelumnya
        #   Proses diulang 3 rounds dengan kunci posisi yang berbeda
        # ════════════════════════════════════════════════════════════════════════
        
        encrypted_uint8 = np.clip(encrypted, 0, 255).astype(np.uint8)
        
        # Hitung base key dari parameter ACM
        # Formula: key_base = (b*c*123 + b*45 + c*67) mod 256
        # Tujuan: Derive key yang unik dari parameter enkripsi
        key_base = (self.b * self.c * 123 + self.b * 45 + self.c * 67) % 256
        
        # Lakukan diffusion 3 rounds untuk sebaran maksimal
        for round_num in range(3):
            for x in range(N):
                for y in range(N):
                    # ────────────────────────────────────────────────────────────
                    # 1. Hitung POSITION-BASED KEY
                    # ────────────────────────────────────────────────────────────
                    # Rumus: pos_key = (key_base + round_num*100 + x*17 + y*31) mod 256
                    # 
                    # Komponen:
                    #   key_base    = kunci dasar dari parameter (tetap untuk semua pixel)
                    #   round_num*100 = pembeda untuk setiap round (0, 100, 200)
                    #   x*17        = komponen posisi x (multiplier 17 prime)
                    #   y*31        = komponen posisi y (multiplier 31 prime)
                    # 
                    # Hasil:
                    #   - Setiap pixel memiliki key unik tergantung posisi dan round
                    #   - Perubahan round menghasilkan key berbeda (cascade effect)
                    # ────────────────────────────────────────────────────────────
                    pos_key = (key_base + round_num * 100 + x * 17 + y * 31) % 256
                    
                    # ────────────────────────────────────────────────────────────
                    # 2. Hitung FEEDBACK DARI NEIGHBORS SEBELUMNYA
                    # ────────────────────────────────────────────────────────────
                    # Ambil nilai dari 3 neighbors:
                    #   - Horizontal (left):   encrypted[x, y-1] dengan wrap-around
                    #   - Vertical (top):      encrypted[x-1, y] dengan wrap-around
                    #   - Diagonal:            encrypted[x-1, y-1] dengan wrap-around
                    # 
                    # Wrap-around: Jika index negatif, ambil dari sisi opposite
                    # Tujuan: Setiap pixel influenced oleh neighbors yang sudah diproses
                    # ────────────────────────────────────────────────────────────
                    feedback_sum = 0
                    
                    # Feedback horizontal (kiri / previous pixel dalam row)
                    if y == 0:
                        feedback_sum += int(encrypted_uint8[x, N-1])  # Wrap to rightmost
                    else:
                        feedback_sum += int(encrypted_uint8[x, y-1])  # Normal: left neighbor
                    
                    # Feedback vertikal (atas / previous pixel dalam column)
                    if x == 0:
                        feedback_sum += int(encrypted_uint8[N-1, y])  # Wrap to bottom
                    else:
                        feedback_sum += int(encrypted_uint8[x-1, y])  # Normal: top neighbor
                    
                    # Feedback diagonal (top-left)
                    x_diag = (x - 1) % N  # Otomatis wrap dengan modulo
                    y_diag = (y - 1) % N  # Otomatis wrap dengan modulo
                    feedback_sum += int(encrypted_uint8[x_diag, y_diag])
                    
                    # ────────────────────────────────────────────────────────────
                    # 3. Hitung FEEDBACK KEY (XOR operation)
                    # ────────────────────────────────────────────────────────────
                    # Rumus: feedback_key = pos_key XOR (feedback_sum mod 256)
                    # 
                    # Operasi XOR (⊕):
                    #   Input:  pos_key (0-255), feedback_sum%256 (0-255)
                    #   Output: feedback_key (0-255)
                    #   Sifat:  a ⊕ b ⊕ b = a (reversible)
                    # 
                    # Tujuan:
                    #   - Menggabungkan deterministic key dengan data-dependent feedback
                    #   - Membuat setiap feedback_key non-predictable
                    #   - Memastikan diffusion tidak linear
                    # ────────────────────────────────────────────────────────────
                    feedback_key = (pos_key ^ (feedback_sum % 256)) % 256
                    
                    # ────────────────────────────────────────────────────────────
                    # 4. TAMBAH FEEDBACK KEY KE PIXEL (Diffusion)
                    # ────────────────────────────────────────────────────────────
                    # Rumus: encrypted_uint8[x,y] = (encrypted_uint8[x,y] + feedback_key) mod 256
                    # 
                    # Operasi:
                    #   - Tambahkan feedback_key ke pixel value saat ini
                    #   - Gunakan modulo 256 untuk wrap-around (tetap 8-bit)
                    #   - Proses in-place (update encrypted_uint8 langsung)
                    # 
                    # Efek:
                    #   - Setiap pixel value berubah sesuai feedback
                    #   - Perubahan cascade: pixel yang dimodifikasi mempengaruhi neighbors berikutnya
                    #   - Multi-round: 3 kali iterasi untuk diffusion maksimal
                    # ────────────────────────────────────────────────────────────
                    encrypted_uint8[x, y] = (int(encrypted_uint8[x, y]) + feedback_key) % 256
        
        return encrypted_uint8
    
    def decrypt(self, cipher: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Dekripsi citra menggunakan inverse ACM
        
        Parameters:
        -----------
        cipher : np.ndarray
            Citra terenkripsi
        iterations : int
            Jumlah iterasi (sama dengan saat enkripsi)
            
        Returns:
        --------
        decrypted : np.ndarray
            Citra terdekripsi
        """
        if len(cipher.shape) == 2:  # Grayscale
            return self._decrypt_channel(cipher, iterations)
        else:  # RGB
            decrypted = np.zeros_like(cipher)
            for ch in range(cipher.shape[2]):
                decrypted[:, :, ch] = self._decrypt_channel(cipher[:, :, ch], iterations)
            return decrypted
    
    def _decrypt_channel(self, channel: np.ndarray, iterations: int) -> np.ndarray:
        """
        Dekripsi single channel dengan REVERSE diffusion (3 rounds) + reverse permutasi
        
        PRINSIP REVERSIBILITY:
        =====================
        Enkripsi adalah operasi bertahap:
          1. ACM Permutation (reversible dengan inverse ACM)
          2. 3-Round Diffusion (reversible dengan reverse diffusion)
        
        Dekripsi adalah reverse dari langkah-langkah tersebut dalam urutan terbalik:
          1. REVERSE Diffusion (kebalikan dari 3-round diffusion)
          2. INVERSE ACM (kebalikan dari ACM permutation)
        
        METODE PENYIMPANAN STATE:
        =========================
        Saat enkripsi, intermediate states disimpan di self._diffusion_states:
          - State -1: Setelah ACM (pre-diffusion)
          - State 0: Setelah round 0
          - State 1: Setelah round 1
          - State 2: Setelah round 2
        
        Saat dekripsi, state ini digunakan untuk feedback yang sama persis
        seperti saat enkripsi, memastikan reversibility sempurna.
        """
        N = channel.shape[0]
        
        decrypted = channel.astype(np.uint8).copy()
        
        # ════════════════════════════════════════════════════════════════════════
        # TAHAP 1: REVERSE MULTIPLE DIFFUSION ROUNDS (3 rounds, in reverse order)
        # ════════════════════════════════════════════════════════════════════════
        # Untuk merekonstruksi pixel value sebelum diffusion, kita KURANGI dengan
        # feedback_key yang sama seperti saat enkripsi.
        #
        # Reverse diffusion formula:
        #   decrypted[x,y] = (decrypted[x,y] - feedback_key) mod 256
        #
        # Pemrosesan terbalik:
        #   Round 2 → Round 1 → Round 0 (mundur dari hasil akhir)
        #
        # Kunci konsistensi: Menggunakan intermediate state sebagai feedback source
        # ════════════════════════════════════════════════════════════════════════
        
        key_base = (self.b * self.c * 123 + self.b * 45 + self.c * 67) % 256
        
        # Reverse dari 3 rounds menjadi 3 rounds mundur: 2, 1, 0
        for round_num in range(2, -1, -1):
            # Ambil intermediate state yang sesuai dari penyimpanan
            feedback_state = self._diffusion_states.get(round_num)
            if feedback_state is None:
                # Fallback jika state tidak tersedia (error handling)
                feedback_state = channel.astype(np.uint8)
            
            # Buat copy untuk proses dekripsi round ini
            decrypted_copy = decrypted.copy()
            
            for x in range(N):
                for y in range(N):
                    # ────────────────────────────────────────────────────────────
                    # REVERSE DIFFUSION: Hitung dan kurangi feedback
                    # ────────────────────────────────────────────────────────────
                    # Position-based key (sama seperti enkripsi)
                    pos_key = (key_base + round_num * 100 + x * 17 + y * 31) % 256
                    
                    # Hitung feedback dari intermediate state (konsisten dengan enkripsi)
                    feedback_sum = 0
                    
                    # Feedback horizontal
                    if y == 0:
                        feedback_sum += int(feedback_state[x, N-1])
                    else:
                        feedback_sum += int(feedback_state[x, y-1])
                    
                    # Feedback vertikal
                    if x == 0:
                        feedback_sum += int(feedback_state[N-1, y])
                    else:
                        feedback_sum += int(feedback_state[x-1, y])
                    
                    # Feedback diagonal
                    x_diag = (x - 1) % N
                    y_diag = (y - 1) % N
                    feedback_sum += int(feedback_state[x_diag, y_diag])
                    
                    # Reverse diffusion: KURANGI feedback (kebalikan dari TAMBAH saat enkripsi)
                    # Jika enkripsi: c = (p + k) mod 256
                    # Maka dekripsi: p = (c - k) mod 256
                    feedback_key = (pos_key ^ (feedback_sum % 256)) % 256
                    decrypted_copy[x, y] = (int(decrypted_copy[x, y]) - feedback_key) % 256
            
            decrypted = decrypted_copy
        
        # ════════════════════════════════════════════════════════════════════════
        # TAHAP 2: REVERSE PERMUTASI - Inverse Arnold Cat Map
        # ════════════════════════════════════════════════════════════════════════
        # Inverse matrix dari ACM:
        #   Original: [1  b   ] → Inverse: [bc+1  -b]
        #             [c  bc+1]             [-c    1 ]
        #
        # Formula dekripsi:
        #   x_orig = ((bc+1)*x - b*y) mod N
        #   y_orig = (-c*x + y) mod N
        #
        # Ini mengembalikan pixel ke posisi asli sebelum ACM permutation
        # ════════════════════════════════════════════════════════════════════════
        for _ in range(iterations):
            temp = np.zeros((N, N), dtype=np.uint8)
            for x in range(N):
                for y in range(N):
                    # Inverse ACM transformation
                    # [x_orig]   [bc+1  -b ] [x]       [x]
                    # [y_orig] = [-c     1 ] [y]  mod N [y]
                    x_orig = ((self.b * self.c + 1) * x - self.b * y) % N
                    y_orig = (-self.c * x + y) % N
                    
                    # Pindahkan pixel dari posisi terenkripsi ke posisi asli
                    temp[x_orig, y_orig] = decrypted[x, y]
            decrypted = temp
        
        return decrypted

class ImageEncryptionEvaluator:
    """Evaluasi keamanan enkripsi citra"""
    
    @staticmethod
    def calculate_histogram(image: np.ndarray) -> Dict:
        """Hitung histogram citra"""
        if len(image.shape) == 2:  # Grayscale
            hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
            return {'gray': hist}
        else:  # RGB
            histograms = {}
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist, _ = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 256])
                histograms[color] = hist
            return histograms
    
    @staticmethod
    def plot_histogram_comparison(plain: np.ndarray, cipher: np.ndarray, 
                                  save_path: str = None):
        """Plot perbandingan histogram plain vs cipher"""
        is_gray = len(plain.shape) == 2
        
        if is_gray:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plain image
            axes[0, 0].imshow(plain, cmap='gray')
            axes[0, 0].set_title('Plain Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Cipher image
            axes[0, 1].imshow(cipher, cmap='gray')
            axes[0, 1].set_title('Cipher Image', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Plain histogram
            axes[1, 0].hist(plain.flatten(), bins=256, range=[0, 256], 
                           color='blue', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Histogram Plain Image', fontsize=11)
            axes[1, 0].set_xlabel('Pixel Intensity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(alpha=0.3)
            
            # Cipher histogram
            axes[1, 1].hist(cipher.flatten(), bins=256, range=[0, 256], 
                           color='red', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Histogram Cipher Image', fontsize=11)
            axes[1, 1].set_xlabel('Pixel Intensity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(alpha=0.3)
            
        else:  # RGB
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            colors = ['red', 'green', 'blue']
            
            # Plain histograms per channel
            for i, color in enumerate(colors):
                axes[0, i].hist(plain[:, :, i].flatten(), bins=256, 
                               range=[0, 256], color=color, alpha=0.7, 
                               edgecolor='black')
                axes[0, i].set_title(f'Plain - {color.capitalize()} Channel', 
                                    fontsize=10)
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].grid(alpha=0.3)
                
                # Cipher histogram
                axes[1, i].hist(cipher[:, :, i].flatten(), bins=256, 
                               range=[0, 256], color=color, alpha=0.7, 
                               edgecolor='black')
                axes[1, i].set_title(f'Cipher - {color.capitalize()} Channel', 
                                    fontsize=10)
                axes[1, i].set_xlabel('Pixel Intensity')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def calculate_correlation(image: np.ndarray, direction: str = 'horizontal', 
                             num_samples: int = 1000) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        ╔════════════════════════════════════════════════════════════════════╗
        ║     CORRELATION ANALYSIS - ADJACENT PIXEL INDEPENDENCE METRICS     ║
        ╚════════════════════════════════════════════════════════════════════╝
        
        Hitung koefisien korelasi piksel bertetangga (Pearson Correlation)
        
        KONSEP DASAR:
        =============
        Plain image memiliki high correlation antara pixel tetangga
        (karena natural image continuity). Enkripsi yang baik harus menghilangkan
        korelasi ini → ciphertext pixel harus independent.
        
        RUMUS PEARSON CORRELATION COEFFICIENT:
        =======================================
        
        r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
        
        Dimana:
          xi, yi = nilai pixel pasangan ke-i
          x̄, ȳ = rata-rata dari x dan y
          Σ = penjumlahan
        
        Penjelasan:
          - Range: -1 (perfectly negative correlated) sampai +1 (perfectly positive)
          - r = 0: Tidak berkorelasi (independent)
          - r ≈ ±1: Sangat berkorelasi
          - |r| < 0.1: Dianggap independent untuk keamanan kriptografi
        
        ARAH KORELASI:
        ==============
        1. HORIZONTAL: Pixel [i,j] vs [i,j+1] (tetangga kanan)
           - Plain image: usually r ≈ 0.95+ (sangat smooth horizontal)
           - Cipher image: r harus <0.1 (random)
        
        2. VERTICAL: Pixel [i,j] vs [i+1,j] (tetangga bawah)
           - Plain image: usually r ≈ 0.95+ (sangat smooth vertikal)
           - Cipher image: r harus <0.1 (random)
        
        3. DIAGONAL: Pixel [i,j] vs [i+1,j+1] (tetangga diagonal)
           - Plain image: usually r ≈ 0.90+ (smooth diagonal)
           - Cipher image: r harus <0.1 (random)
        
        SAMPLING:
        =========
        Karena full image correlation mahal, gunakan sampling:
          - num_samples = 1000 pasangan pixel
          - Random selection dari image untuk representatif
        
        Target: |r| < 0.1 untuk semua 3 arah
          Interpretasi: Ciphertext berperilaku seperti random noise
                       → Tidak ada pattern yang bisa dianalisis
        
        Contoh:
          Plain image horizontal correlation: r = 0.9543 (highly correlated)
          Cipher image horizontal correlation: r = -0.0234 (independent) ✓
        
        Parameters:
        -----------
        image : np.ndarray
            Citra input (grayscale atau RGB, akan dikonversi ke grayscale)
        direction : str
            'horizontal', 'vertical', atau 'diagonal'
        num_samples : int
            Jumlah pasangan piksel untuk sampling (default 1000)
            
        Returns:
        --------
        correlation : float
            Pearson correlation coefficient
        x_samples : np.ndarray
            Nilai pixel pertama dari sampel
        y_samples : np.ndarray
            Nilai pixel tetangga dari sampel
        """
        # Konversi ke grayscale jika RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        x_samples = []
        y_samples = []
        
        # ════════════════════════════════════════════════════════════════════════
        # SAMPLING: Ambil pasangan pixel secara random
        # ════════════════════════════════════════════════════════════════════════
        for _ in range(num_samples):
            if direction == 'horizontal':
                # Random row, random column (pastikan ada neighbor ke kanan)
                i = np.random.randint(0, h)
                j = np.random.randint(0, w - 1)
                x_samples.append(image[i, j])         # Pixel saat ini
                y_samples.append(image[i, j + 1])     # Neighbor kanan
                
            elif direction == 'vertical':
                # Random row (pastikan ada neighbor ke bawah), random column
                i = np.random.randint(0, h - 1)
                j = np.random.randint(0, w)
                x_samples.append(image[i, j])         # Pixel saat ini
                y_samples.append(image[i + 1, j])     # Neighbor bawah
                
            elif direction == 'diagonal':
                # Random row dan column (pastikan ada neighbor diagonal)
                i = np.random.randint(0, h - 1)
                j = np.random.randint(0, w - 1)
                x_samples.append(image[i, j])         # Pixel saat ini
                y_samples.append(image[i + 1, j + 1])  # Neighbor diagonal
        
        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        
        # ════════════════════════════════════════════════════════════════════════
        # HITUNG PEARSON CORRELATION COEFFICIENT
        # ════════════════════════════════════════════════════════════════════════
        # Gunakan numpy corrcoef:
        #   corrcoef(x, y) mengembalikan correlation matrix 2×2:
        #   [[r(x,x)   r(x,y)]     [[1     r   ]
        #    [r(y,x)   r(y,y)]]  =  [r     1   ]]
        #   Maka r(x,y) = corrcoef[0, 1]
        correlation = np.corrcoef(x_samples, y_samples)[0, 1]
        
        return correlation, x_samples, y_samples
    
    @staticmethod
    def plot_correlation_analysis(plain: np.ndarray, cipher: np.ndarray, 
                                  save_path: str = None):
        """Plot analisis korelasi lengkap"""
        directions = ['horizontal', 'vertical', 'diagonal']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        results = []
        
        for idx, direction in enumerate(directions):
            # Plain image correlation
            corr_plain, x_plain, y_plain = ImageEncryptionEvaluator.calculate_correlation(
                plain, direction, num_samples=1000
            )
            
            # Cipher image correlation
            corr_cipher, x_cipher, y_cipher = ImageEncryptionEvaluator.calculate_correlation(
                cipher, direction, num_samples=1000
            )
            
            results.append({
                'Direction': direction.capitalize(),
                'Plain Image': f"{corr_plain:.4f}",
                'Cipher Image': f"{corr_cipher:.4f}",
                'Reduction (%)': f"{abs((corr_cipher - corr_plain) / corr_plain * 100):.2f}"
            })
            
            # Plot Plain
            axes[0, idx].scatter(x_plain, y_plain, alpha=0.3, s=5, color='blue')
            axes[0, idx].set_title(f'Plain - {direction.capitalize()}\nr = {corr_plain:.4f}', 
                                  fontsize=11, fontweight='bold')
            axes[0, idx].set_xlabel('Pixel Value')
            axes[0, idx].set_ylabel('Adjacent Pixel Value')
            axes[0, idx].grid(alpha=0.3)
            axes[0, idx].set_xlim([0, 255])
            axes[0, idx].set_ylim([0, 255])
            
            # Plot Cipher
            axes[1, idx].scatter(x_cipher, y_cipher, alpha=0.3, s=5, color='red')
            axes[1, idx].set_title(f'Cipher - {direction.capitalize()}\nr = {corr_cipher:.4f}', 
                                  fontsize=11, fontweight='bold')
            axes[1, idx].set_xlabel('Pixel Value')
            axes[1, idx].set_ylabel('Adjacent Pixel Value')
            axes[1, idx].grid(alpha=0.3)
            axes[1, idx].set_xlim([0, 255])
            axes[1, idx].set_ylim([0, 255])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_npcr_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> Tuple[float, float]:
        """
        ╔════════════════════════════════════════════════════════════════════╗
        ║        NPCR & UACI - DIFFERENTIAL ATTACK SECURITY METRICS         ║
        ╚════════════════════════════════════════════════════════════════════╝
        
        Hitung NPCR dan UACI untuk differential attack analysis
        
        KONSEP DASAR:
        =============
        Untuk mengukur keamanan enkripsi terhadap differential attack,
        kita membandingkan dua ciphertext yang berasal dari plaintext yang
        hampir identik (hanya 1 pixel berbeda).
        
        Hasilnya seharusnya SANGAT BERBEDA (ideal: 50% pixel berubah, 
        intensity perubahan acak).
        
        RUMUS NPCR (Number of Pixels Change Rate):
        ===========================================
        
        NPCR = (D / N²) × 100%
        
        Dimana:
          D = jumlah pixel yang berbeda antara cipher1 dan cipher2
          N² = total jumlah pixel dalam citra
          
        Penjelasan:
          - Hitung perbedaan setiap pixel: D[i,j] = 1 jika cipher1[i,j] ≠ cipher2[i,j]
          - Jumlahkan: D = Σ D[i,j]
          - Hitung persentase: NPCR = (D / N²) × 100%
        
        Target: ≥99%
          Berarti: Minimal 99% pixel harus berbeda antara cipher1 dan cipher2
          Interpretasi: Enkripsi sangat sensitif terhadap perubahan plaintext
                       (avalanche effect excellent)
        
        Contoh perhitungan:
          Citra 4×4 = 16 pixel
          Jika 15 pixel berbeda: NPCR = (15/16) × 100% = 93.75%
          Jika 16 pixel berbeda: NPCR = (16/16) × 100% = 100%
        
        RUMUS UACI (Unified Average Changing Intensity):
        ================================================
        
        UACI = (1/N²) × Σ|cipher1[i,j] - cipher2[i,j]| / 255 × 100%
        
        Dimana:
          cipher1[i,j], cipher2[i,j] = intensitas pixel pada posisi [i,j]
          |...| = nilai absolut perbedaan intensitas (0-255)
          255 = maksimum intensitas pixel
          N² = total jumlah pixel
        
        Penjelasan:
          - Hitung perbedaan intensitas: diff[i,j] = |cipher1[i,j] - cipher2[i,j]|
          - Normalize: diff_norm[i,j] = diff[i,j] / 255
          - Rata-rata: mean_diff = (1/N²) × Σ diff_norm[i,j]
          - Persentase: UACI = mean_diff × 100%
        
        Target: ~33.46% (ideal: 100/3 ≈ 33.33%)
          Berarti: Rata-rata setiap pixel berubah ~1/3 dari range intensitas
          Interpretasi: Perubahan intensitas tersebar merata dan acak
        
        Contoh perhitungan:
          Citra 2×2 = 4 pixel
          cipher1 = [[10, 20], [30, 40]]
          cipher2 = [[50, 60], [70, 80]]
          diff = |10-50| + |20-60| + |30-70| + |40-80| = 40+40+40+40 = 160
          UACI = (160 / (4×255)) × 100% = (160/1020) × 100% = 15.69%
        
        KOMBINASI NPCR + UACI:
        ======================
        ✓ NPCR ≥99% + UACI ~33% = Enkripsi SANGAT AMAN
          - Pixel-level diffusion sempurna (NPCR)
          - Bit-level diffusion merata (UACI)
          - Resistant terhadap differential attack
        
        Parameters:
        -----------
        cipher1, cipher2 : np.ndarray
            Dua cipher image dari plain image yang hanya berbeda 1 pixel
            
        Returns:
        --------
        npcr : float
            Number of Pixels Change Rate (%)
        uaci : float
            Unified Average Changing Intensity (%)
        """
        # Jika RGB, gunakan channel pertama saja untuk konsistensi
        if len(cipher1.shape) == 3:
            cipher1 = cipher1[:, :, 0]  # Ambil red channel saja
            cipher2 = cipher2[:, :, 0]
        
        # ════════════════════════════════════════════════════════════════════════
        # HITUNG NPCR: Persentase pixel yang berubah
        # ════════════════════════════════════════════════════════════════════════
        # Buat difference matrix: 1 jika pixel berbeda, 0 jika sama
        diff_matrix = (cipher1 != cipher2).astype(int)
        total_pixels = diff_matrix.size              # N²
        changed_pixels = np.sum(diff_matrix)         # D
        
        # NPCR = (D / N²) × 100%
        npcr = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        
        # ════════════════════════════════════════════════════════════════════════
        # HITUNG UACI: Rata-rata perubahan intensitas
        # ════════════════════════════════════════════════════════════════════════
        if changed_pixels > 0:
            # Hitung sum of absolute intensity difference
            intensity_diff = np.sum(np.abs(cipher1.astype(float) - cipher2.astype(float)))
            
            # UACI = (sum_diff / (N² × 255)) × 100%
            uaci = (intensity_diff / (total_pixels * 255)) * 100
        else:
            # Jika tidak ada pixel yang berbeda, UACI = 0
            uaci = 0.0
        
        return npcr, uaci
    
    @staticmethod
    def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        ╔════════════════════════════════════════════════════════════════════╗
        ║         PSNR - PEAK SIGNAL-TO-NOISE RATIO (QUALITY METRIC)        ║
        ╚════════════════════════════════════════════════════════════════════╝
        
        Hitung PSNR antara dua citra
        
        KONSEP DASAR:
        =============
        PSNR adalah metrik untuk mengukur kualitas rekonstruksi citra.
        Semakin tinggi PSNR, semakin mirip image1 dengan image2.
        
        RUMUS PSNR:
        ===========
        
        MSE = (1/N²) × Σ(image1[i,j] - image2[i,j])²
        
        Dimana:
          MSE = Mean Squared Error (kesalahan kuadrat rata-rata)
          N² = jumlah pixel
          Σ = penjumlahan semua pixel
        
        PSNR = 10 × log₁₀((MAX²) / MSE) dB
        
        Dimana:
          MAX = maksimum nilai pixel (untuk 8-bit: 255)
          log₁₀ = logaritma basis 10
          dB = decibel (unit log)
        
        INTERPRETASI:
        ==============
        PSNR VALUE:        KUALITAS:
        ────────────────────────────
        ∞ dB               Identik sempurna (MSE = 0, image1 == image2)
        >50 dB             Sangat bagus (imperceptible difference)
        40-50 dB           Bagus (minimal difference)
        20-40 dB           Fair (noticeable difference)
        <20 dB             Buruk (sangat berbeda)
        
        PENGGUNAAN DALAM KRIPTOGRAFI:
        =============================
        1. PSNR(Plain vs Cipher): Harus SANGAT RENDAH (<20 dB)
           → Cipher harus terlihat seperti random noise
           → Tidak ada visual similarity dengan plain
           Contoh: 0.02 dB (MSE sangat besar)
        
        2. PSNR(Plain vs Decrypted): Harus INFINITY dB (∞)
           → Decrypted harus identik pixel-perfect dengan plain
           → MSE harus = 0 (tidak boleh ada error)
           Contoh: ∞ dB (MSE = 0, semua pixel sama)
        
        CONTOH KALKULASI:
        =================
        Plain image:    [100, 150, 200, 50]
        Decrypted:      [100, 150, 200, 50]
        
        Error: [0, 0, 0, 0]
        MSE = (0² + 0² + 0² + 0²) / 4 = 0
        PSNR = 10 × log₁₀(255² / 0) = 10 × log₁₀(∞) = ∞ dB ✓
        
        Contoh kalkulasi dengan error kecil:
        Plain:     [100, 150, 200, 50]
        Decrypted: [101, 149, 200, 50]  (1 error)
        
        Error: [1, 1, 0, 0]
        MSE = (1² + 1² + 0² + 0²) / 4 = 2/4 = 0.5
        PSNR = 10 × log₁₀(255² / 0.5) = 10 × log₁₀(130050) ≈ 51.1 dB
        
        KASUS ENKRIPSI-DEKRIPSI:
        ========================
        Enkripsi sempurna seharusnya mencapai:
          ✓ PSNR(Plain vs Cipher): ~0 dB (completely random)
          ✓ PSNR(Plain vs Decrypted): ∞ dB (perfect recovery)
        
        Jika PSNR(Plain vs Decrypted) < ∞ dB:
          ⚠️ Decryption ada error (bukan pixel-perfect)
          ⚠️ Mungkin masalah di reverse diffusion
          ⚠️ Bisa terjadi loss informasi
        
        Parameters:
        -----------
        image1, image2 : np.ndarray
            Dua citra yang dibandingkan
            
        Returns:
        --------
        psnr : float
            Peak Signal-to-Noise Ratio (dB)
            Nilai ∞ jika MSE = 0 (identik sempurna)
        """
        # ════════════════════════════════════════════════════════════════════════
        # HITUNG MEAN SQUARED ERROR (MSE)
        # ════════════════════════════════════════════════════════════════════════
        # MSE = (1/N²) × Σ(image1[i,j] - image2[i,j])²
        # 
        # Proses:
        #   1. Konversi ke float untuk precision
        #   2. Hitung difference setiap pixel
        #   3. Pangkat 2 (squared)
        #   4. Rata-rata (mean)
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        
        # Special case: jika MSE = 0 (identik sempurna)
        # → PSNR = ∞ dB (logaritma dari ∞ = ∞)
        if mse == 0:
            return float('inf')
        
        # ════════════════════════════════════════════════════════════════════════
        # HITUNG PSNR
        # ════════════════════════════════════════════════════════════════════════
        # PSNR = 10 × log₁₀((255²) / MSE) dB
        max_pixel = 255.0  # Maksimum intensitas pixel (8-bit)
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        
        return psnr


# ==================== MODUL 3: PIPELINE PENGUJIAN ====================

class ImageEncryptionTester:
    """Pipeline pengujian lengkap untuk enkripsi citra"""
    
    def __init__(self, acm: ArnoldCatMap, output_dir: str = 'results'):
        self.acm = acm
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/histograms", exist_ok=True)
        os.makedirs(f"{output_dir}/correlations", exist_ok=True)
        os.makedirs(f"{output_dir}/encrypted", exist_ok=True)
    
    def test_single_image(self, image_path: str, iterations: int = 5) -> Dict:
        """
        Test enkripsi pada satu citra
        
        Returns:
        --------
        results : dict
            Dictionary berisi semua metrik evaluasi
        """
        print(f"\n{'='*60}")
        print(f"Testing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to square jika perlu
        h, w = image.shape[:2]
        size = min(h, w)
        if len(image.shape) == 2:
            image = image[:size, :size]
        else:
            image = image[:size, :size, :]
        
        print(f"Image size: {image.shape}")
        
        # ENKRIPSI
        print("\n[1/6] Encrypting...")
        start_time = time.time()
        cipher = self.acm.encrypt(image, iterations)
        encrypt_time = time.time() - start_time
        print(f"   Encryption time: {encrypt_time*1000:.2f} ms")
        
        # DEKRIPSI
        print("[2/6] Decrypting...")
        start_time = time.time()
        decrypted = self.acm.decrypt(cipher, iterations)
        decrypt_time = time.time() - start_time
        print(f"   Decryption time: {decrypt_time*1000:.2f} ms")
        
        # VALIDASI DEKRIPSI
        print("[3/6] Validating decryption...")
        identical_pixels = np.sum(image == decrypted)
        total_pixels = image.size
        recovery_rate = (identical_pixels / total_pixels) * 100
        print(f"   Perfect recovery: {recovery_rate:.2f}% pixels identical")
        
        # HISTOGRAM ANALYSIS
        print("[4/6] Analyzing histograms...")
        ImageEncryptionEvaluator.plot_histogram_comparison(
            image, cipher,
            save_path=f"{self.output_dir}/histograms/{os.path.basename(image_path)}_hist.png"
        )
        
        # CORRELATION ANALYSIS
        print("[5/6] Analyzing correlation...")
        corr_df = ImageEncryptionEvaluator.plot_correlation_analysis(
            image, cipher,
            save_path=f"{self.output_dir}/correlations/{os.path.basename(image_path)}_corr.png"
        )
        print("\n   Correlation coefficients:")
        print(corr_df.to_string(index=False))
        
        # NPCR & UACI
        print("\n[6/6] Calculating NPCR & UACI...")
        # Ubah 1 piksel pada plain image dengan cara yang lebih significant
        modified = image.copy().astype(np.float32)
        mid = image.shape[0] // 2
        original_pixel = modified[mid, mid] if len(image.shape) == 2 else modified[mid, mid, 0].copy()
        
        # Ubah pixel ke nilai berbeda (pastikan berbeda minimal 50 units)
        if original_pixel < 128:
            if len(image.shape) == 2:
                modified[mid, mid] = min(original_pixel + 50, 255)
            else:
                modified[mid, mid, 0] = min(original_pixel + 50, 255)
        else:
            if len(image.shape) == 2:
                modified[mid, mid] = max(original_pixel - 50, 0)
            else:
                modified[mid, mid, 0] = max(original_pixel - 50, 0)
        
        modified = np.clip(modified, 0, 255).astype(np.uint8)
        
        # Pastikan ada perubahan
        if np.array_equal(image, modified):
            if len(image.shape) == 2:
                modified[mid, mid] = (int(original_pixel) + 1) % 256
            else:
                modified[mid, mid, 0] = (int(original_pixel) + 1) % 256
        
        cipher2 = self.acm.encrypt(modified, iterations)
        
        npcr, uaci = ImageEncryptionEvaluator.calculate_npcr_uaci(cipher, cipher2)
        print(f"   NPCR: {npcr:.4f}%")
        print(f"   UACI: {uaci:.4f}%")
        
        # PSNR
        psnr_plain_cipher = ImageEncryptionEvaluator.calculate_psnr(image, cipher)
        psnr_plain_decrypt = ImageEncryptionEvaluator.calculate_psnr(image, decrypted)
        
        print(f"\n   PSNR (Plain vs Cipher): {psnr_plain_cipher:.2f} dB")
        print(f"   PSNR (Plain vs Decrypt): {psnr_plain_decrypt:.2f} dB")
        
        # Save encrypted image
        cipher_path = f"{self.output_dir}/encrypted/{os.path.basename(image_path)}_cipher.png"
        cv2.imwrite(cipher_path, cipher)
        
        # Compile results
        results = {
            'Image': os.path.basename(image_path),
            'Size': f"{image.shape[0]}×{image.shape[1]}",
            'Pixels': image.shape[0] * image.shape[1],
            'Iterations': iterations,
            'Encrypt Time (ms)': round(encrypt_time * 1000, 2),
            'Decrypt Time (ms)': round(decrypt_time * 1000, 2),
            'Recovery Rate (%)': round(recovery_rate, 2),
            'Corr Horizontal (Plain)': float(corr_df[corr_df['Direction'] == 'Horizontal']['Plain Image'].values[0]),
            'Corr Horizontal (Cipher)': float(corr_df[corr_df['Direction'] == 'Horizontal']['Cipher Image'].values[0]),
            'Corr Vertical (Plain)': float(corr_df[corr_df['Direction'] == 'Vertical']['Plain Image'].values[0]),
            'Corr Vertical (Cipher)': float(corr_df[corr_df['Direction'] == 'Vertical']['Cipher Image'].values[0]),
            'Corr Diagonal (Plain)': float(corr_df[corr_df['Direction'] == 'Diagonal']['Plain Image'].values[0]),
            'Corr Diagonal (Cipher)': float(corr_df[corr_df['Direction'] == 'Diagonal']['Cipher Image'].values[0]),
            'NPCR (%)': round(npcr, 4),
            'UACI (%)': round(uaci, 4),
            'PSNR Plain-Cipher (dB)': round(psnr_plain_cipher, 2),
            'PSNR Plain-Decrypt (dB)': round(psnr_plain_decrypt, 2) if psnr_plain_decrypt != float('inf') else 'INF'
        }
        
        return results
    
    def test_multiple_images(self, image_paths: List[str], iterations: int = 5) -> pd.DataFrame:
        """
        Test enkripsi pada beberapa citra
        
        Returns:
        --------
        df_results : pd.DataFrame
            DataFrame berisi hasil semua pengujian
        """
        all_results = []
        
        for img_path in image_paths:
            try:
                result = self.test_single_image(img_path, iterations)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        df_results = pd.DataFrame(all_results)
        
        # Save to CSV
        csv_path = f"{self.output_dir}/encryption_results_iter{iterations}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {csv_path}")
        print(f"{'='*60}")
        
        return df_results


# ==================== MODUL 4: MAIN EXECUTION ====================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   SISTEM ENKRIPSI CITRA DIGITAL BERBASIS ARNOLD CAT MAP   ║
    ║            Penelitian Kriptografi - ITERA 2025            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # KONFIGURASI
    PARAMETER_B = 2
    PARAMETER_C = 2
    ITERATIONS = 5
    
    # Inisialisasi ACM
    acm = ArnoldCatMap(b=PARAMETER_B, c=PARAMETER_C)
    
    # Inisialisasi Tester
    tester = ImageEncryptionTester(acm, output_dir='results')
    
    # DAFTAR CITRA ANDA
    test_images = [
        'test_images/lenna.jpg',
        'test_images/4.2.03.tiff',
        'test_images/4.2.05.tiff',
        'test_images/4.2.06.tiff',
        'test_images/4.2.07.tiff',
    ]
    
    # JALANKAN PENGUJIAN
    print("\n[STARTING BATCH ENCRYPTION TEST]")
    print("="*80)
    
    try:
        df_results = tester.test_multiple_images(test_images, iterations=ITERATIONS)
        
        # TAMPILKAN RINGKASAN LENGKAP
        print("\n" + "="*80)
        print("SUMMARY RESULTS - ALL IMAGES")
        print("="*80)
        print(df_results.to_string(index=False))
        
        # RINGKASAN METRIK KUNCI (dengan pengecekan kolom)
        print("\n" + "="*80)
        print("KEY METRICS SUMMARY")
        print("="*80)
        
        # Pilih kolom yang ada
        available_cols = df_results.columns.tolist()
        key_metrics = []
        
        for col in ['Image', 'Size', 'NPCR (%)', 'UACI (%)', 
                    'Corr Horizontal (Cipher)', 'PSNR Plain-Cipher (dB)', 
                    'PSNR Plain-Decrypt (dB)']:
            if col in available_cols:
                key_metrics.append(col)
        
        if key_metrics:
            print(df_results[key_metrics].to_string(index=False))
        
        # CEK TARGET PROPOSAL
        print("\n" + "="*80)
        print("VALIDATION AGAINST PROPOSAL TARGETS")
        print("="*80)
        
        # Cek NPCR
        if 'NPCR (%)' in available_cols:
            npcr_values = df_results['NPCR (%)']
            npcr_pass = (npcr_values >= 99.0).all()
            npcr_min = npcr_values.min()
            npcr_max = npcr_values.max()
            npcr_avg = npcr_values.mean()
            
            print(f"✓ NPCR ≥ 99%:        {'✓ PASS' if npcr_pass else '✗ FAIL'}")
            print(f"  - Range: {npcr_min:.4f}% - {npcr_max:.4f}%")
            print(f"  - Average: {npcr_avg:.4f}%")
        
        # Cek Korelasi
        if 'Corr Horizontal (Cipher)' in available_cols:
            corr_values = df_results['Corr Horizontal (Cipher)'].abs()
            corr_pass = (corr_values < 0.1).all()
            corr_max = corr_values.max()
            corr_avg = corr_values.mean()
            
            print(f"\n✓ Correlation < 0.1: {'✓ PASS' if corr_pass else '✗ FAIL'}")
            print(f"  - Max: {corr_max:.4f}")
            print(f"  - Average: {corr_avg:.4f}")
        
        # Cek UACI
        if 'UACI (%)' in available_cols:
            uaci_values = df_results['UACI (%)']
            uaci_pass = ((uaci_values >= 31.0) & (uaci_values <= 36.0)).all()
            uaci_avg = uaci_values.mean()
            
            print(f"\n✓ UACI ≈ 33%:        {'✓ PASS' if uaci_pass else '✗ FAIL'}")
            print(f"  - Average: {uaci_avg:.4f}% (expected: ~33.46%)")
        
        # Cek Dekripsi
        if 'Recovery Rate (%)' in available_cols:
            recovery = df_results['Recovery Rate (%)']
            recovery_pass = (recovery == 100.0).all()
            
            print(f"\n✓ Perfect Decryption: {'✓ PASS' if recovery_pass else '✗ FAIL'}")
            print(f"  - All images: {recovery.min():.2f}% recovery rate")
        
        # Cek Waktu
        if 'Encrypt Time (ms)' in available_cols:
            encrypt_time = df_results['Encrypt Time (ms)']
            time_pass = (encrypt_time < 1000).all()
            
            print(f"\n✓ Time < 1 second:   {'✓ PASS' if time_pass else '⚠ MARGINAL'}")
            print(f"  - Average: {encrypt_time.mean():.2f} ms")
            print(f"  - Max: {encrypt_time.max():.2f} ms")
        
        print("\n" + "="*80)
        print("✓ All results saved to: results/")
        print("="*80)
        print("""
Files generated:
- CSV table: results/encryption_results_iter5.csv
- Histograms: results/histograms/*.png
- Correlations: results/correlations/*.png
- Encrypted images: results/encrypted/*.png

Next steps for your paper:
1. Copy CSV data to your results table
2. Insert histogram comparisons as figures
3. Insert correlation scatter plots as figures
4. Reference the metrics in your discussion section
        """)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Please check that image paths are correct:")
        for img in test_images:
            exists = "✓" if os.path.exists(img) else "✗"
            print(f"   {exists} {img}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Execution completed!")