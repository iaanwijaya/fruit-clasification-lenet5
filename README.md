# Klasifikasi Kematangan Buah (Fresh vs Rotten) — Modifikasi LeNet-5 (PyTorch)

Proyek *Deep Learning* berbasis **PyTorch** untuk mengklasifikasikan tingkat kesegaran buah secara biner (**Fresh** vs **Rotten**) menggunakan arsitektur **Modifikasi LeNet-5** (input citra RGB berukuran $64 \times 64$). Proyek ini mencakup *notebook* pelatihan dan evaluasi (`klasifikasi-buah.ipynb`), file bobot model hasil *checkpointing* (`lenet_buah_model.pth`), aplikasi antarmuka (`app.py`), serta konfigurasi *environment* (`requirements.txt` dan `.gitignore`).

## Tentang Proyek

Penilaian kesegaran buah secara manual sering kali memakan waktu dan bersifat subjektif. Proyek ini membangun *pipeline* *end-to-end* mulai dari pemuatan dataset lokal, *preprocessing* citra, perancangan arsitektur CNN modifikasi LeNet-5, pelatihan dengan *Early Stopping* dan *Model Checkpointing*, hingga evaluasi performa menggunakan *classification report* dan *confusion matrix*.

### Ringkasan Dataset & Preprocessing
- **Dataset:** 1.058 file citra awal (`684` kelas `fresh` dan `374` kelas `rotten`), dengan **546 sampel citra valid** yang diproses melalui *data loader* (mendukung format `.jpg`, `.jpeg`, `.png`, `.webp`, dan `.jfif`). Link dataset: https://www.kaggle.com/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images
- **Pembagian Data (*Data Splitting*):** **80% data latih (436 sampel)** dengan `shuffle=True` dan **20% data validasi (110 sampel)** dengan `shuffle=False`, menggunakan `batch_size = 32`.
- **Transformasi Citra:** *Resize* menjadi **64 × 64** piksel (3 kanal warna RGB) dan konversi ke *tensor* PyTorch.

### Arsitektur Modifikasi LeNet-5
- **Feature Extractor:**
  - `Conv1` (3 $\rightarrow$ 6 *feature maps*, kernel $5 \times 5$) + *Average Pooling* $2 \times 2$
  - `Conv2` (6 $\rightarrow$ 16 *feature maps*, kernel $5 \times 5$) + *Average Pooling* $2 \times 2$
- **Classifier:**
  - *Flatten* ($16 \times 13 \times 13 = 2704$) $\rightarrow$ `FC1` ($2704 \rightarrow 120$, ReLU) $\rightarrow$ `FC2` ($120 \rightarrow 84$, ReLU) $\rightarrow$ `FC3` ($84 \rightarrow 2$ kelas: `fresh` dan `rotten`).

### Pelatihan & Evaluasi Model
- **Konfigurasi Pelatihan:** Menggunakan **Adam Optimizer** (`lr = 0.001`) dan `CrossEntropyLoss`, dilengkapi **Early Stopping** (`patience = 4`, batas maksimal 30 *epoch*) serta *checkpointing* otomatis ke `lenet_buah_model.pth` berdasarkan *Validation Loss* terendah.
- **Ringkasan Hasil Pelatihan & Evaluasi:**
  - **Proses Training:** Pelatihan berjalan selama 17 *epoch* sebelum dihentikan otomatis oleh *Early Stopping*, dengan bobot terbaik tersimpan pada **Epoch ke-13** (*Val Loss* `0.2323`, *Val Acc* `92.73%`), puncak *Val Acc* `94.55%` di **Epoch ke-15**, dan *Train Acc* `99.31%` di akhir sesi.
  - **Classification Report (110 sampel validasi):** Mencapai akurasi keseluruhan **92.73%** (*macro F1-score* `0.9185`, *weighted F1-score* `0.9268`). Kelas `fresh` (72 sampel) memperoleh *precision* `0.9324`, *recall* `0.9583`, dan *F1-score* `0.9452`, sedangkan kelas `rotten` (38 sampel) memperoleh *precision* `0.9167`, *recall* `0.8684`, dan *F1-score* `0.8919`.
  - **Confusion Matrix:** Dari 110 citra validasi, terdapat **102 prediksi benar** (69/72 `fresh` dan 33/38 `rotten`) dan **8 kesalahan klasifikasi** (3 `fresh` terprediksi `rotten` dan 5 `rotten` terprediksi `fresh`).

## Struktur Proyek

    klasifikasi-buah/
    ├── .venv/                    # Virtual environment lokal (diabaikan oleh .gitignore)
    ├── dataset/                  # Folder dataset lokal (diabaikan oleh .gitignore)
    │   ├── fresh/
    │   └── rotten/
    ├── .gitignore                # Aturan pengabaian file/folder untuk Git
    ├── app.py                    # Aplikasi inferensi / antarmuka pengguna
    ├── klasifikasi-buah.ipynb    # Notebook preprocessing, model LeNet-5, training & evaluasi
    ├── lenet_buah_model.pth      # Bobot model terbaik hasil checkpointing & early stopping
    ├── requirements.txt          # Daftar dependensi Python proyek
    └── README.md                 # Dokumentasi proyek

## Cara Clone & Menjalankan Proyek

1. **Clone repository ke komputer lokal:**

       git clone https://github.com/iaanwijaya/fruit-clasification-lenet5.git
       cd klasifikasi-buah

   *(Sesuaikan URL repository dengan alamat repo GitHub milikmu.)*

2. **Buat dan aktifkan *virtual environment*:**
   - **Windows (PowerShell):**

         python -m venv .venv
         .venv\Scripts\Activate.ps1

   - **Windows (CMD):**

         python -m venv .venv
         .venv\Scripts\activate.bat

   - **Linux / macOS:**

         python3 -m venv .venv
         source .venv/bin/activate

3. **Install seluruh *library* dari `requirements.txt`:**

       pip install -r requirements.txt

4. **Siapkan dataset (wajib jika menjalankan ulang pelatihan dari awal):**
   Karena folder `dataset/` tidak di-push ke GitHub, buat struktur folder berikut di dalam direktori proyek lalu isi dengan citra buah segar dan busuk:

       dataset/
       ├── fresh/
       └── rotten/

5. **Jalankan *notebook* atau aplikasi:**
   - **Pelatihan & Evaluasi (`klasifikasi-buah.ipynb`):** Buka `klasifikasi-buah.ipynb` menggunakan VS Code atau Jupyter Notebook, pilih kernel `.venv`, lalu jalankan sel secara berurutan.
   - **Aplikasi (`app.py`):** Jalankan aplikasi menggunakan perintah yang sesuai dengan implementasi di `app.py`, misalnya `streamlit run app.py` atau `python app.py`.

## Catatan & Keterbatasan

- Folder `.venv/` dan `dataset/` dikecualikan melalui `.gitignore` agar repository tetap ringan dan terhindar dari isu ukuran file maupun distribusi dataset.
- Komputasi berjalan otomatis pada perangkat yang tersedia (`cuda` jika GPU NVIDIA/CUDA tersedia, atau `cpu` sebagai *fallback*).
- Karena arsitektur LeNet-5 tergolong ringkas dan dilatih pada dataset terbatas, performa terbaik dicapai pada citra yang selaras dengan karakteristik domain dataset pelatihan; pengujian pada gambar di luar domain dataset (*domain shift*) dapat memiliki akurasi yang lebih bervariasi.