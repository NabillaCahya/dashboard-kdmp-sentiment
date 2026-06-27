# 🌾 Dashboard Sentimen KDMP

Dashboard analisis sentimen komentar TikTok Koperasi Desa Merah Putih (KDMP).

## 📁 Struktur Folder

```
dashboard_kdmp/
├── app.py                     ← Entry point (halaman home)
├── requirements.txt
├── .env                       ← API key Gemini
├── pages/
│   ├── 1_Overview.py          ← Distribusi, tren, wordcloud
│   ├── 2_Prediksi.py          ← Prediksi teks baru
│   ├── 3_Perbandingan.py      ← Evaluasi semua model
│   └── 4_Chatbot.py           ← Chatbot berbasis data
├── utils/
│   ├── preprocessing.py
│   ├── predictor.py
│   └── chatbot.py
├── models/                    ← ⬅️ TARUH FILE INI
│   ├── tfidf_vectorizer.pkl
│   ├── model_naive_bayes.pkl
│   ├── model_svm.pkl
│   ├── model_logistic_regression.pkl
│   ├── model_decision_tree.pkl
│   └── bert_kdmp_final/       ← ⬅️ EXTRACT ZIP INDOBERT KE SINI
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files...
└── data/                      ← ⬅️ TARUH FILE INI
    └── data_with_sentiment.csv
```

## 🚀 Setup & Jalankan

### 1. Install dependencies
```bash
cd dashboard_kdmp
pip install -r requirements.txt
```

### 2. Siapkan file model
- Copy semua `.pkl` ke folder `models/`
- Extract `bert_kdmp_final.zip` ke folder `models/bert_kdmp_final/`
- Copy `data_with_sentiment.csv` ke folder `data/`

### 3. Set API Key Gemini (untuk chatbot)
Edit file `.env`:
```
GEMINI_API_KEY=AIza...your_key_here
```
Atau bisa diisi langsung di sidebar halaman Chatbot.

Dapetin API key gratis di: https://aistudio.google.com/app/apikey

### 4. Update angka performa model
Di file `pages/2_Prediksi.py` dan `pages/3_Perbandingan.py`,
update variabel `MODEL_METRICS` dengan angka aktual dari output Notebook 2 kamu.

### 5. Jalankan
```bash
streamlit run app.py
```

Buka browser: http://localhost:8501

## 📝 Catatan
- IndoBERT membutuhkan RAM minimal 4GB untuk load
- Kalau RAM terbatas, prediksi IndoBERT bisa dinonaktifkan
- Data CSV cukup besar (219K baris), loading pertama agak lama tapi sudah di-cache
