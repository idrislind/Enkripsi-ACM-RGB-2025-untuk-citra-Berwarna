# 📖 Panduan Penggunaan Lengkap - ACM Image Encryption

Panduan step-by-step untuk menggunakan aplikasi enkripsi citra menggunakan Arnold Cat Map.

---

## 🎯 Daftar Isi

1. [Persiapan Awal](#-persiapan-awal)
2. [Mode Enkripsi & Dekripsi](#-mode-enkripsi--dekripsi)
3. [Mode Analisis Keamanan](#-mode-analisis-keamanan)
4. [Tips & Troubleshooting](#-tips--troubleshooting)

---

## 🚀 Persiapan Awal

### 1. Jalankan Aplikasi

Buka terminal/command prompt di folder project, lalu jalankan:

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

### 2. Siapkan Gambar

- Format yang didukung: **JPG, PNG, TIFF, BMP**
- Rekomendasi ukuran: 256x256 hingga 1024x1024 piksel
- Letakkan gambar test di folder `test_images/`

---

## 🔒 Mode Enkripsi & Dekripsi

### Langkah 1: Pilih Mode Operasi di Sidebar

<img src="docs/sidebar-mode.png" alt="Sidebar Mode" width="300">

1. Di **sidebar kiri**, cari bagian **"Mode Operasi"**
2. Pilih mode **"🔒 Enkripsi & Dekripsi"** dari dropdown

### Langkah 2: Atur Parameter Kunci Enkripsi

<img src="docs/sidebar-parameters.png" alt="Parameters" width="300">

Di sidebar, atur parameter kunci:
- **Parameter b**: Nilai integer (default: 2)
- **Parameter c**: Nilai integer (default: 2)

> ⚠️ **PENTING**: Simpan nilai b dan c ini! Anda memerlukan nilai yang sama untuk dekripsi.

**Contoh konfigurasi:**
```
Parameter b = 3
Parameter c = 5
```

### Langkah 3: Upload Gambar Original

1. Di bagian **kiri halaman utama**, klik area **"Upload Citra"**
2. Pilih file gambar dari komputer Anda
3. Gambar akan ditampilkan bersama informasinya:
   - Dimensi (lebar x tinggi)
   - Mode warna (RGB/Grayscale)

**Screenshot:**
```
┌─────────────────────────────┐
│  📤 Upload Citra            │
│  ┌───────────────────────┐  │
│  │  [Browse files...]    │  │
│  └───────────────────────┘  │
│                             │
│  🖼️ Gambar Preview          │
│  ┌───────────────────────┐  │
│  │    [Original Image]   │  │
│  └───────────────────────┘  │
│  Dimensi: 512 x 512         │
│  Mode: RGB                  │
└─────────────────────────────┘
```

### Langkah 4: Preprocessing Gambar

1. Pilih **ukuran target** dari dropdown:
   - 256 x 256 piksel
   - 512 x 512 piksel (recommended)
   - 1024 x 1024 piksel

2. Klik tombol **"Proses Gambar"**

3. Tunggu hingga muncul notifikasi:
   > ✅ Citra berhasil diproses ke 512 x 512

### Langkah 5: Enkripsi Gambar

1. Di bagian **kanan halaman**, akan muncul tombol **"🔐 Enkripsi Citra"**
2. Klik tombol tersebut
3. Progress bar akan muncul: *"Mengenkripsi citra..."*
4. Setelah selesai:
   - Gambar terenkripsi ditampilkan (tampak acak/noise)
   - Waktu enkripsi ditampilkan (contoh: `Enkripsi selesai dalam 245.32 ms`)
   - Tombol **"Download encrypted_image.png"** tersedia

**Hasil Enkripsi:**
```
Original Image          →          Encrypted Image
┌─────────────┐                   ┌─────────────┐
│   🏞️        │    [ENCRYPT]     │   📊 Noise  │
│  Clear      │    ═══════>       │   Random    │
│  Readable   │                   │   Pattern   │
└─────────────┘                   └─────────────┘
```

### Langkah 6: Download Gambar Terenkripsi

1. Klik link **"Download encrypted_image.png"**
2. Simpan file di folder yang mudah diakses
3. File ini akan digunakan untuk:
   - Testing dekripsi
   - Analisis keamanan

### Langkah 7: Dekripsi Gambar

1. Scroll ke bawah pada bagian kanan
2. Cari section **"Dekripsi"** (dibawah section Enkripsi)
3. Klik tombol **"Dekripsi Citra"**
4. Progress bar akan muncul: *"Mendekripsi citra..."*
5. Hasil dekripsi akan ditampilkan

**Verifikasi Hasil:**
- ✅ **Dekripsi Sempurna**: Gambar hasil dekripsi identik dengan original
  - Muncul notifikasi: `✅ Dekripsi sempurna. Waktu: 198.47 ms`
  - Popup: "Dekripsi berhasil dan hasil siap diunduh"

- ❌ **Dekripsi Gagal**: Gambar tidak sama dengan original
  - Muncul pesan: `❌ Dekripsi tidak sempurna. Periksa parameter kunci.`
  - Kemungkinan penyebab: Parameter b atau c salah

### Langkah 8: Download Gambar Terdekripsi

1. Klik link **"Download decrypted_image.png"**
2. Bandingkan secara visual dengan gambar original
3. Jika identik, proses enkripsi-dekripsi berhasil!

---

## 🔍 Mode Analisis Keamanan

Mode ini digunakan untuk mengevaluasi tingkat keamanan enkripsi dengan berbagai metrik kriptografi.

### Langkah 1: Pindah ke Mode Analisis

1. Di **sidebar kiri**, ubah **"Mode Operasi"**
2. Pilih **"🔍 Analisis Keamanan"**
3. Halaman akan reload dengan interface analisis

### Langkah 2: Upload Gambar untuk Dianalisis

1. Klik area **"Upload citra untuk dianalisis"**
2. Pilih **gambar original** (bukan yang terenkripsi)
3. Aplikasi akan **otomatis**:
   - Melakukan preprocessing (resize ke 512x512)
   - Mengenkripsi gambar
   - Mendekripsi untuk verifikasi
   - Menghitung semua metrik keamanan

**Loading:**
```
⏳ Mengenkripsi citra untuk analisis...
━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
```

### Langkah 3: Tab 📊 HISTOGRAM

Tab ini menampilkan distribusi frekuensi piksel untuk menganalisis keacakan.

**Yang Ditampilkan:**

1. **Histogram Citra Original**
   - 3 grafik bar (Red, Green, Blue channels)
   - Menunjukkan distribusi nilai piksel 0-255
   - Biasanya memiliki pola tertentu

2. **Histogram Citra Terenkripsi**
   - 3 grafik bar (Red, Green, Blue channels)
   - Harus terlihat **merata** (uniform distribution)
   - Menunjukkan enkripsi berhasil mengacak data

**Interpretasi:**
- ✅ **Baik**: Histogram encrypted sangat berbeda dari original, distribusi merata
- ❌ **Buruk**: Histogram encrypted masih memiliki pola seperti original

**Download:**
- Tombol **"Download histogram_original.png"** - histogram citra asli
- Tombol **"Download histogram_encrypted.png"** - histogram citra terenkripsi

### Langkah 4: Tab 🔗 KORELASI

Tab ini menganalisis korelasi antara piksel yang berdekatan.

**Yang Ditampilkan:**

1. **Scatter Plot Original**
   - 3 arah korelasi: Horizontal, Vertikal, Diagonal
   - 3 channels: Red, Green, Blue
   - Total 9 subplot (3x3 grid)
   - Biasanya menunjukkan korelasi kuat (garis diagonal)

2. **Scatter Plot Encrypted**
   - 3 arah korelasi: Horizontal, Vertikal, Diagonal
   - 3 channels: Red, Green, Blue
   - Harus menunjukkan **scatter acak** (tidak ada pola)

3. **Tabel Koefisien Korelasi**

| Direction | Original R | Original G | Original B | Encrypted R | Encrypted G | Encrypted B |
|-----------|------------|------------|------------|-------------|-------------|-------------|
| Horizontal| 0.9234     | 0.9456     | 0.9123     | 0.0012      | -0.0023     | 0.0034      |
| Vertical  | 0.9345     | 0.9567     | 0.9234     | -0.0015     | 0.0019      | -0.0027     |
| Diagonal  | 0.9123     | 0.9345     | 0.9012     | 0.0008      | -0.0031     | 0.0021      |

**Interpretasi:**
- ✅ **Sangat Baik**: Korelasi encrypted < 0.01 (mendekati 0)
  - Tampil: `✅ EXCELLENT - Korelasi maksimal cipher: 0.0034`
- ⚠️ **Marginal**: Korelasi encrypted antara 0.01 - 0.05
  - Tampil: `⚠️ MARGINAL - Korelasi maksimal cipher: 0.0234`

**Download:**
- **"Download correlation_original.png"** - scatter plot citra asli
- **"Download correlation_encrypted.png"** - scatter plot citra terenkripsi

### Langkah 5: Tab 🔄 NPCR & UACI

Tab ini menguji resistansi terhadap **differential attack** (serangan diferensial).

**Cara Kerja:**
1. Aplikasi membuat modifikasi kecil pada citra original (1 piksel diubah)
2. Mengenkripsi citra yang sudah dimodifikasi
3. Membandingkan hasil enkripsi original vs modified

**Metrik yang Ditampilkan:**

1. **NPCR (Number of Pixels Change Rate)**
   - Persentase piksel yang berubah
   - **Target: ≥ 99%**
   - Contoh hasil: `99.6234%`
   - Status:
     - ✅ PASS jika ≥ 99%
     - ❌ FAIL jika < 99%

2. **UACI (Unified Average Changing Intensity)**
   - Rata-rata intensitas perubahan
   - **Target: ~33.46%** (range: 31-36%)
   - Contoh hasil: `33.2145%`
   - Status:
     - ✅ PASS jika 31% ≤ UACI ≤ 36%
     - ⚠️ MARGINAL jika di luar range

**Screenshot:**
```
┌─────────────────────────────────────┐
│  NPCR - Number of Pixels Change Rate│
│  99.6234%                           │
│  Target: >= 99%                     │
│  ✅ PASS                            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  UACI - Unified Average Changing    │
│  Intensity                          │
│  33.2145%                           │
│  Target: ~33.46%                    │
│  ✅ PASS                            │
└─────────────────────────────────────┘
```

**Interpretasi:**
- ✅ Perubahan 1 piksel menyebabkan perubahan massive pada hasil enkripsi
- ❌ Perubahan kecil tidak cukup mempengaruhi hasil (enkripsi lemah)

### Langkah 6: Tab 📈 PSNR & MSE

Tab ini mengukur **kualitas degradasi** antara citra original dan encrypted.

**Metrik yang Ditampilkan:**

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Satuan: **dB (decibel)**
   - **Semakin rendah semakin baik** untuk enkripsi
   - Target: **< 10 dB**
   - Contoh hasil: `8.2345 dB`

2. **MSE (Mean Square Error)**
   - Rata-rata kuadrat error per piksel
   - **Semakin tinggi semakin baik** untuk enkripsi
   - Target: **> 5000**
   - Contoh hasil: `12345.67`

**Format Tampilan:**
```
┌─────────────────────────────────────┐
│  PSNR (Peak Signal-to-Noise Ratio) │
│  8.23 dB                            │
│  ✅ Excellent (< 10 dB)             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  MSE (Mean Square Error)            │
│  12345.67                           │
│  ✅ High difference detected        │
└─────────────────────────────────────┘
```

**Interpretasi:**
- ✅ **PSNR rendah** = citra encrypted sangat berbeda dari original
- ✅ **MSE tinggi** = error besar antara original dan encrypted
- Kedua metrik ini menunjukkan enkripsi efektif

### Langkah 7: Tab 📋 RINGKASAN

Tab ini menampilkan **semua metrik dalam satu tabel** untuk kemudahan perbandingan.

**Tabel Metrik Lengkap:**

| Metrik | Nilai | Status | Keterangan |
|--------|-------|--------|------------|
| **NPCR** | 99.6234% | ✅ PASS | Perubahan piksel sangat tinggi |
| **UACI** | 33.2145% | ✅ PASS | Intensitas perubahan ideal |
| **Correlation (H)** | 0.0023 | ✅ EXCELLENT | Korelasi horizontal rendah |
| **Correlation (V)** | -0.0015 | ✅ EXCELLENT | Korelasi vertikal rendah |
| **Correlation (D)** | 0.0034 | ✅ EXCELLENT | Korelasi diagonal rendah |
| **PSNR** | 8.23 dB | ✅ EXCELLENT | Kualitas degradasi tinggi |
| **MSE** | 12345.67 | ✅ HIGH | Error tinggi (baik) |
| **Waktu Enkripsi** | 245.32 ms | ℹ️ INFO | Kecepatan proses |
| **Waktu Dekripsi** | 198.47 ms | ℹ️ INFO | Kecepatan proses |

**Kesimpulan Otomatis:**

Aplikasi akan memberikan verdict keseluruhan:

```
╔═══════════════════════════════════════╗
║  🎯 KESIMPULAN ANALISIS KEAMANAN      ║
╠═══════════════════════════════════════╣
║  Status: ✅ AMAN                      ║
║                                       ║
║  Sistem enkripsi ini memenuhi semua   ║
║  standar keamanan kriptografi untuk   ║
║  enkripsi citra digital.              ║
║                                       ║
║  Metrik yang PASS: 7/7 (100%)         ║
╚═══════════════════════════════════════╝
```

**Download Ringkasan:**
- Tombol **"Export Summary as CSV"** - download tabel sebagai file CSV
- Dapat dibuka di Excel/Google Sheets untuk dokumentasi

### Langkah 8: Interpretasi Keseluruhan

**Enkripsi Berkualitas Tinggi:**
- ✅ NPCR ≥ 99%
- ✅ UACI ≈ 33%
- ✅ Korelasi < 0.01
- ✅ PSNR < 10 dB
- ✅ Histogram merata
- ✅ Dekripsi sempurna dengan kunci benar

**Enkripsi Berkualitas Rendah:**
- ❌ NPCR < 95%
- ❌ UACI jauh dari 33%
- ❌ Korelasi > 0.1
- ❌ PSNR > 20 dB
- ❌ Histogram masih berpola
- ❌ Masih terlihat outline gambar original

---

## 💡 Tips & Troubleshooting

### Tips Penggunaan

1. **Parameter Kunci**
   - Gunakan nilai b dan c yang berbeda untuk keamanan lebih tinggi
   - Simpan parameter di tempat aman (buku catatan/password manager)
   - Jangan gunakan nilai yang mudah ditebak (seperti 1, 1)

2. **Ukuran Gambar**
   - Untuk analisis cepat: gunakan 256x256
   - Untuk hasil optimal: gunakan 512x512
   - Untuk high-resolution: gunakan 1024x1024
   - Gambar akan di-crop menjadi persegi (aspect ratio 1:1)

3. **Format File**
   - Gunakan PNG untuk hasil terbaik (lossless)
   - JPG dapat digunakan tapi ada kompresi
   - Hindari format dengan transparansi untuk RGB analysis

4. **Performa**
   - Waktu enkripsi meningkat dengan ukuran gambar
   - 512x512: ~200-300ms
   - 1024x1024: ~800-1200ms
   - Tutup tab browser lain untuk performa maksimal

### Troubleshooting

**Problem 1: "Dekripsi tidak sempurna"**
- ✅ **Solusi**: Pastikan parameter b dan c sama dengan saat enkripsi
- ✅ **Solusi**: Jangan mengubah parameter di tengah proses

**Problem 2: "Import error saat running"**
- ✅ **Solusi**: Install dependencies: `pip install -r requirements.txt`
- ✅ **Solusi**: Pastikan Python 3.8+ terinstall

**Problem 3: "Gambar tidak muncul setelah upload"**
- ✅ **Solusi**: Pastikan format file didukung (JPG/PNG/TIFF)
- ✅ **Solusi**: Cek ukuran file tidak terlalu besar (< 10MB)
- ✅ **Solusi**: Refresh halaman browser

**Problem 4: "Analisis keamanan lambat"**
- ✅ **Solusi**: Gunakan gambar yang lebih kecil (256x256)
- ✅ **Solusi**: Tutup aplikasi lain yang berat
- ✅ **Solusi**: Tunggu hingga proses selesai (jangan refresh)

**Problem 5: "Correlation plot tidak muncul"**
- ✅ **Solusi**: Install matplotlib: `pip install matplotlib`
- ✅ **Solusi**: Pastikan tidak ada error di console browser (F12)

**Problem 6: "Port 8501 already in use"**
- ✅ **Solusi**: Matikan instance Streamlit lain yang running
- ✅ **Solusi**: Atau gunakan port berbeda: `streamlit run app.py --server.port 8502`

---

## 📊 Contoh Skenario Lengkap

### Skenario: Enkripsi & Analisis Foto Landscape

**1. Setup**
```bash
cd Enkripsi-ACM-RGB-2025-untuk-citra-Berwarna
streamlit run app.py
```

**2. Enkripsi**
- Mode: Enkripsi & Dekripsi
- Parameter b: 3
- Parameter c: 5
- Upload: `test_images/landscape.jpg`
- Resize: 512x512
- Proses → Enkripsi → Download

**3. Dekripsi**
- Gunakan parameter yang sama (b=3, c=5)
- Klik Dekripsi
- Verifikasi hasilnya identical
- Download hasil dekripsi

**4. Analisis**
- Pindah ke Mode: Analisis Keamanan
- Upload: `test_images/landscape.jpg` (original)
- Tunggu auto-processing

**5. Review Hasil**
- Tab Histogram: ✅ Distribusi merata
- Tab Korelasi: ✅ Korelasi < 0.005
- Tab NPCR/UACI: ✅ NPCR 99.61%, UACI 33.24%
- Tab PSNR/MSE: ✅ PSNR 8.23 dB
- Tab Ringkasan: ✅ Semua metrik PASS

**6. Dokumentasi**
- Download semua plot (histogram, correlation)
- Export summary sebagai CSV
- Simpan gambar encrypted untuk publikasi

---

## 📚 Referensi

- **Arnold Cat Map**: [Wikipedia](https://en.wikipedia.org/wiki/Arnold%27s_cat_map)
- **NPCR & UACI Standards**: IEEE Cryptography Standards
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Image Encryption Survey**: Chaos-based Cryptography Methods

---

## ❓ FAQ

**Q: Apakah enkripsi ini aman untuk data sensitif?**
A: Algoritma ini cocok untuk akademis dan penelitian. Untuk produksi, gunakan standard AES-256.

**Q: Bisakah saya mengenkripsi video?**
A: Tidak langsung. Anda perlu extract frame-by-frame terlebih dahulu.

**Q: Apakah parameter b dan c harus integer?**
A: Ya, harus integer positif (1-100).

**Q: Mengapa hasil enkripsi selalu berbeda?**
A: Jika parameter sama, hasil enkripsi harus identik. Jika berbeda, ada bug atau parameter tidak sama.

**Q: Bisakah saya mengembalikan gambar tanpa kunci?**
A: TIDAK. Tanpa parameter b dan c yang tepat, dekripsi tidak akan berhasil.

---

**📞 Bantuan Lebih Lanjut**: Buat issue di repository GitHub atau hubungi pengembang.

**⭐ Suka aplikasi ini?**: Star repository di GitHub dan share ke teman/kolega!

---

*Last Updated: March 8, 2026*
