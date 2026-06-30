import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os

st.set_page_config(page_title="Dashboard Sentimen KDMP", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
[data-testid="stSidebar"] * { color: #f4f7fa !important; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 16px;
    padding: 20px;
}
.section-title {
    font-size: 1.15rem; font-weight: 700;
    color: #f4f7fa !important;
    background: rgba(10,22,39,0.85);
    border-left: 4px solid #3498db;
    padding: 12px 16px;
    margin: 28px 0 10px 0;
    border-radius: 8px;
}
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
}
.hero-banner h1 { color: white; margin: 0; font-size: 1.8rem; }
.hero-banner p  { color: #cbd5e1; margin: 8px 0 0 0; font-size: 0.95rem; }
.insight-text {
    background: rgba(52,152,219,0.08);
    border-left: 4px solid #3498db;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0 0 0;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
}
.model-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.model-card .name  { font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px; }
.model-card .score { font-size: 1.5rem; font-weight: 700; }
.best-badge {
    background: rgba(52,152,219,0.15);
    border: 1px solid #3498db;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #3498db;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data_with_sentiment.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['month'] = df['create_time'].dt.to_period('M').astype(str)
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"File data tidak ditemukan: `{DATA_PATH}`")
    st.stop()

df = load_data()

total = len(df)
counts = df['sentiment'].value_counts()
pos = counts.get('positive', 0)
neu = counts.get('neutral', 0)
neg = counts.get('negative', 0)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>Dashboard Analisis Sentimen KDMP</h1>
    <p>Analisis sentimen 219.746 komentar TikTok terhadap program Koperasi Desa Merah Putih (KDMP)
    menggunakan IndoBERT fine-tuned dan empat model Machine Learning klasikal.</p>
</div>
""", unsafe_allow_html=True)

# ── Metric Cards ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Komentar",   f"{total:,}",  "TikTok KDMP")
c2.metric("Sentimen Positif", f"{pos:,}",    f"{pos/total*100:.1f}% dari total")
c3.metric("Sentimen Netral",  f"{neu:,}",    f"{neu/total*100:.1f}% dari total")
c4.metric("Sentimen Negatif", f"{neg:,}",    f"{neg/total*100:.1f}% dari total", delta_color="inverse")

# ── Distribusi Sentimen ───────────────────────────────────────
st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig_pie = px.pie(
        pd.DataFrame({'Sentimen': ['Positif','Netral','Negatif'], 'Jumlah': [pos, neu, neg]}),
        names='Sentimen', values='Jumlah', hole=0.45,
        color='Sentimen',
        color_discrete_map={'Positif':'#2ecc71','Netral':'#3498db','Negatif':'#e74c3c'},
        title="Proporsi Sentimen"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
    fig_pie.update_layout(height=360, showlegend=True, legend=dict(orientation='h', y=-0.15),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#f4f7fa'))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        pd.DataFrame({
            'Sentimen': ['Positif','Netral','Negatif'],
            'Jumlah':   [pos, neu, neg],
            'Persen':   [f"{pos/total*100:.1f}%", f"{neu/total*100:.1f}%", f"{neg/total*100:.1f}%"]
        }),
        x='Sentimen', y='Jumlah', color='Sentimen', text='Persen',
        color_discrete_map={'Positif':'#2ecc71','Netral':'#3498db','Negatif':'#e74c3c'},
        title="Jumlah Komentar per Sentimen"
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(showlegend=False, height=360,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#f4f7fa'))
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("""
<div class="insight-text">
Distribusi sentimen dari 219.746 komentar TikTok menunjukkan dominasi sentimen <b>negatif sebesar 48.6%</b>,
yang mengindikasikan banyaknya kritik dan kekhawatiran masyarakat terhadap program KDMP.
Sentimen positif (25.9%) dan netral (25.6%) relatif berimbang, mencerminkan bahwa sebagian masyarakat
mendukung program namun masih banyak yang menunggu pembuktian nyata.
Tingginya sentimen negatif menjadi sinyal penting bagi pemangku kebijakan untuk meningkatkan
komunikasi publik dan transparansi implementasi program.
</div>
""", unsafe_allow_html=True)

# ── Tren Sentimen ─────────────────────────────────────────────
st.markdown('<div class="section-title">Tren Sentimen per Bulan</div>', unsafe_allow_html=True)

monthly = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
pivot   = monthly.pivot(index='month', columns='sentiment', values='count').fillna(0).reset_index()
pivot   = pivot[pivot['month'] >= '2025-01']

fig_trend = go.Figure()
for col, color, label in [('positive','#2ecc71','Positif'),('neutral','#3498db','Netral'),('negative','#e74c3c','Negatif')]:
    if col in pivot.columns:
        fig_trend.add_trace(go.Scatter(
            x=pivot['month'], y=pivot[col], name=label,
            line=dict(color=color, width=3), mode='lines+markers', marker=dict(size=8)
        ))

fig_trend.update_layout(
    xaxis_title="Bulan", yaxis_title="Jumlah Komentar", height=380,
    hovermode='x unified', legend=dict(orientation='h', y=-0.2),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f4f7fa')
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("""
<div class="insight-text">
Tren sentimen menunjukkan lonjakan signifikan komentar pada periode awal peluncuran program KDMP,
dengan sentimen negatif yang konsisten mendominasi sepanjang periode pengamatan.
Peningkatan komentar negatif umumnya berkorelasi dengan pemberitaan media dan pernyataan
pejabat publik terkait program, yang memicu respons kritis dari warganet.
Sentimen positif cenderung meningkat setelah adanya demonstrasi manfaat nyata di tingkat desa,
namun belum mampu mengimbangi dominasi sentimen negatif secara keseluruhan.
</div>
""", unsafe_allow_html=True)

# ── WordCloud ─────────────────────────────────────────────────
st.markdown('<div class="section-title">WordCloud per Sentimen</div>', unsafe_allow_html=True)

@st.cache_data
def gen_wc(sentiment):
    texts = df[df['sentiment'] == sentiment]['content_clean'].dropna().astype(str)
    if texts.empty:
        return None
    cmap = {'positive': 'Greens', 'neutral': 'Blues', 'negative': 'Reds'}
    wc = WordCloud(
        width=700, height=350, background_color='white',
        max_words=80, colormap=cmap[sentiment], collocations=False
    ).generate(' '.join(texts))
    return wc

wc1, wc2, wc3 = st.columns(3)
for wcol, sent, title in [(wc1,'positive','Positif'),(wc2,'neutral','Netral'),(wc3,'negative','Negatif')]:
    with wcol:
        st.markdown(f"**Sentimen {title}**")
        wc = gen_wc(sent)
        if wc:
            fig_wc, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc, use_container_width=True)
            plt.close()

st.markdown("""
<div class="insight-text">
WordCloud sentimen positif didominasi kata <b>bagus, mantap, bermanfaat, dukung, semoga</b>,
mencerminkan harapan dan dukungan masyarakat terhadap program pemberdayaan ekonomi desa.
Sentimen negatif menampilkan kata <b>gagal, bohong, korupsi, tipu, janji</b> yang kuat,
mengindikasikan ketidakpercayaan publik terkait transparansi dan realisasi program.
Sentimen netral cenderung berisi pertanyaan dan pernyataan faktual seperti
<b>kapan, dimana, gimana, informasi</b>, menunjukkan masyarakat yang masih mencari informasi.
</div>
""", unsafe_allow_html=True)

# ── Top 15 Kata ───────────────────────────────────────────────
st.markdown('<div class="section-title">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)

@st.cache_data
def get_top_words(sentiment, n=15):
    texts = df[df['sentiment'] == sentiment]['content_clean'].dropna().astype(str)
    words = ' '.join(texts).split()
    return Counter(words).most_common(n)

tk1, tk2, tk3 = st.columns(3)
for tcol, sent, color, title in [
    (tk1,'positive','#2ecc71','Positif'),
    (tk2,'neutral','#3498db','Netral'),
    (tk3,'negative','#e74c3c','Negatif')
]:
    with tcol:
        top  = get_top_words(sent)
        wdf  = pd.DataFrame(top, columns=['Kata','Frekuensi'])
        fig  = px.bar(wdf, x='Frekuensi', y='Kata', orientation='h',
                      color_discrete_sequence=[color], title=title)
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'}, height=420,
            margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f4f7fa')
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight-text">
Frekuensi kata mengonfirmasi temuan WordCloud secara lebih terukur.
Kata-kata negatif dengan frekuensi tertinggi mengindikasikan narasi yang berulang di kalangan pengkritik program,
sementara kata positif menunjukkan harapan konkret masyarakat terhadap manfaat ekonomi yang diharapkan.
Pola ini konsisten dengan dominasi sentimen negatif 48.6% dan menjadi panduan prioritas
komunikasi yang perlu diperkuat oleh pengelola program.
</div>
""", unsafe_allow_html=True)

# ── Perbandingan Model ────────────────────────────────────────
st.markdown('<div class="section-title">Perbandingan Performa Model</div>', unsafe_allow_html=True)

MODEL_METRICS = [
    {'Model': 'IndoBERT Fine-tuned', 'Accuracy': 91.18, 'F1-Score': 91.09, 'color': '#3498db', 'best': True},
    {'Model': 'SVM',                 'Accuracy': 85.08, 'F1-Score': 84.18, 'color': '#2ecc71', 'best': False},
    {'Model': 'Logistic Regression', 'Accuracy': 84.76, 'F1-Score': 83.00, 'color': '#9b59b6', 'best': False},
    {'Model': 'Naive Bayes',         'Accuracy': 81.61, 'F1-Score': 78.20, 'color': '#f39c12', 'best': False},
    {'Model': 'Decision Tree',       'Accuracy': 78.25, 'F1-Score': 73.38, 'color': '#e74c3c', 'best': False},
]

cols = st.columns(5)
for i, m in enumerate(MODEL_METRICS):
    with cols[i]:
        badge = '<span class="best-badge">Terbaik</span>' if m['best'] else ''
        st.markdown(f"""
<div class="model-card">
    <div class="name">{m['Model']}{badge}</div>
    <div class="score" style="color:{m['color']}">{m['Accuracy']}%</div>
    <div style="color:#94a3b8; font-size:0.8rem; margin-top:4px;">Accuracy</div>
    <div style="color:{m['color']}; font-size:1.1rem; font-weight:600; margin-top:8px;">{m['F1-Score']}%</div>
    <div style="color:#94a3b8; font-size:0.8rem;">F1-Score</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

df_metrics = pd.DataFrame(MODEL_METRICS)
fig_m = go.Figure()
for metric, color in [('Accuracy','#3498db'),('F1-Score','#2ecc71')]:
    fig_m.add_trace(go.Bar(
        name=metric, x=df_metrics['Model'], y=df_metrics[metric],
        marker_color=color,
        text=[f"{v:.1f}%" for v in df_metrics[metric]],
        textposition='outside'
    ))
fig_m.update_layout(
    barmode='group', height=380,
    yaxis=dict(range=[60,100], title='Score (%)'),
    legend=dict(orientation='h', y=-0.2),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f4f7fa'), margin=dict(t=20,b=20)
)
st.plotly_chart(fig_m, use_container_width=True)

st.markdown("""
<div class="insight-text">
IndoBERT fine-tuned mencapai akurasi tertinggi <b>91.18%</b> dengan F1-Score 91.09%,
unggul signifikan dibanding model Machine Learning klasikal karena kemampuannya
memahami konteks bahasa Indonesia secara mendalam melalui pre-training skala besar.
SVM menempati posisi kedua (85.08%) diikuti Logistic Regression (84.76%),
keduanya kompetitif dan jauh lebih efisien secara komputasi dibanding IndoBERT.
Decision Tree menunjukkan performa terendah (78.25%) yang disebabkan keterbatasan
dalam menangkap pola linguistik kompleks pada data teks berdimensi tinggi.
Pemilihan IndoBERT sebagai model utama pada halaman User Analysis didasarkan pada
keunggulan performa ini, terutama untuk menangani nuansa bahasa seperti sarkasme dan ironi.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Insight & Rekomendasi AI ──────────────────────────────────
st.markdown('<div class="section-title">Insight & Rekomendasi AI</div>', unsafe_allow_html=True)
st.markdown("Klik tombol di bawah untuk menghasilkan analisis mendalam dan rekomendasi dari Gemini berdasarkan keseluruhan data sentimen KDMP.")

@st.cache_data(show_spinner=False, ttl=3600)
def generate_insight_from_gemini(prompt_text):
    from dotenv import load_dotenv
    import google.generativeai as genai

    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt_text)
    return response.text

if st.button("Hasilkan Insight dengan AI", type="primary", use_container_width=True):
    prompt = f"""Kamu adalah analis data sentimen yang berpengalaman dalam menganalisis opini publik terhadap program pemerintah Indonesia.

Berikut adalah data hasil analisis sentimen komentar TikTok terhadap program Koperasi Desa Merah Putih (KDMP):

- Total komentar: {total:,}
- Sentimen Positif: {pos:,} ({pos/total*100:.1f}%)
- Sentimen Netral: {neu:,} ({neu/total*100:.1f}%)
- Sentimen Negatif: {neg:,} ({neg/total*100:.1f}%)

Model terbaik yang digunakan: IndoBERT Fine-tuned (Accuracy: 91.18%, F1-Score: 91.09%)

Berikan analisis dalam format JSON PERSIS berikut (tidak ada teks lain di luar JSON, jangan gunakan karakter ** atau ##):

{{
  "insight": "3-4 poin temuan utama dari distribusi sentimen ini, jelaskan apa yang bisa disimpulkan dari dominasi sentimen dan implikasinya. Tulis sebagai paragraf mengalir, bukan list bernomor.",
  "kesimpulan": "simpulkan secara keseluruhan kondisi persepsi publik terhadap program KDMP berdasarkan data sentimen ini dalam 3-4 kalimat",
  "rekomendasi": ["rekomendasi konkret 1", "rekomendasi konkret 2", "rekomendasi konkret 3", "rekomendasi konkret 4", "rekomendasi konkret 5"]
}}

Rekomendasi harus konkret dan spesifik, mencakup baik untuk pengelola program KDMP maupun untuk peneliti dan pemangku kebijakan. Bahasa Indonesia profesional dan akademis."""

    try:
        with st.spinner("Gemini sedang menganalisis data..."):
            raw = generate_insight_from_gemini(prompt).strip()
            import json, re
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'^```\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            st.session_state['home_insight'] = json.loads(raw)
    except Exception as e:
        st.error(f"Gagal menghubungi Gemini: {e}")

if 'home_insight' in st.session_state:
    result = st.session_state['home_insight']

    st.markdown(f"""
<div style="background:rgba(155,89,182,0.08); border:1px solid rgba(155,89,182,0.35);
            border-radius:12px; padding:24px 28px; color:#f4f7fa; line-height:1.8; margin-top:16px;">
    <div style="font-weight:700; color:#9b59b6; font-size:1.05rem; margin-bottom:10px;">Insight</div>
    {result.get('insight', '-')}
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="background:rgba(241,196,15,0.08); border:1px solid rgba(241,196,15,0.35);
            border-radius:12px; padding:24px 28px; color:#f4f7fa; line-height:1.8; margin-top:16px;">
    <div style="font-weight:700; color:#f1c40f; font-size:1.05rem; margin-bottom:10px;">Kesimpulan</div>
    {result.get('kesimpulan', '-')}
</div>
""", unsafe_allow_html=True)

    rekomendasi_list = result.get('rekomendasi', [])
    rekomendasi_html = "<ol style='margin:8px 0 0 0; padding-left:20px; line-height:1.9;'>"
    for r in rekomendasi_list:
        rekomendasi_html += f"<li>{r}</li>"
    rekomendasi_html += "</ol>"

    st.markdown(f"""
<div style="background:rgba(46,204,113,0.08); border:1px solid rgba(46,204,113,0.35);
            border-radius:12px; padding:24px 28px; color:#f4f7fa; margin-top:16px;">
    <div style="font-weight:700; color:#2ecc71; font-size:1.05rem;">Rekomendasi</div>
    {rekomendasi_html}
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Dashboard Analisis Sentimen KDMP • Tugas Akhir Teknik Informatika UNESA")