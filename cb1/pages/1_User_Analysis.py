import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
import io

from utils.chatbot import build_insight_payload, generate_insight, _get_gemini

st.set_page_config(page_title="User Analysis — KDMP", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
[data-testid="stSidebar"] * { color: #f4f7fa !important; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 16px; padding: 20px;
}
.section-title {
    font-size: 1.15rem; font-weight: 700;
    color: #f4f7fa !important;
    background: rgba(10,22,39,0.85);
    border-left: 4px solid #3498db;
    padding: 12px 16px; margin: 28px 0 10px 0; border-radius: 8px;
}
.insight-text {
    background: rgba(52,152,219,0.08);
    border-left: 4px solid #3498db;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin: 8px 0 0 0;
    color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;
}
.gemini-box {
    background: rgba(155,89,182,0.08);
    border: 1px solid rgba(155,89,182,0.35);
    border-radius: 12px; padding: 22px 26px;
    color: #f4f7fa; line-height: 1.8;
}
.result-box {
    border-radius: 12px; padding: 24px 28px;
    margin-top: 16px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

MODEL_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
LABEL_DECODE = {0: 'positive', 1: 'neutral', 2: 'negative'}
LABEL_INDO   = {'positive': 'Positif', 'neutral': 'Netral', 'negative': 'Negatif'}
COLOR_MAP    = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}

def normalize_sentiment(series):
    mapping = {
        'positif':'positive','positive':'positive',
        'netral':'neutral','neutral':'neutral',
        'negatif':'negative','negative':'negative',
    }
    return series.astype(str).str.lower().str.strip().map(mapping)

def find_col(cols, candidates):
    lower = [c.lower() for c in cols]
    for c in candidates:
        if c in lower:
            return lower.index(c)
    return 0

st.title("User Analysis")

tab_manual, tab_csv = st.tabs(["Prediksi Teks Manual", "Analisis Dataset CSV"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI TEKS MANUAL
# ══════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown("Ketik kalimat dan sistem akan memprediksi sentimennya menggunakan IndoBERT.")

    user_text = st.text_area(
        "Tulis kalimat di sini:",
        placeholder="Contoh: Program koperasi desa ini sangat membantu masyarakat sekitar...",
        height=120, key="manual_text_input"
    )

    if st.button("Prediksi Kalimat", type="primary", key="btn_manual"):
        if not user_text.strip():
            st.warning("Tulis kalimat terlebih dahulu.")
        else:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                BERT_DIR  = os.path.join(MODEL_DIR, 'bert_kdmp_final')
                tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
                bert_m    = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
                clf       = pipeline("text-classification", model=bert_m, tokenizer=tokenizer,
                                      truncation=True, max_length=128, device=-1)
                with st.spinner("Memprediksi..."):
                    res = clf([user_text])[0]

                lmap       = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
                pred       = lmap.get(res['label'], res['label'])
                confidence = res['score'] * 100
                color      = COLOR_MAP.get(pred, '#888')
                label      = LABEL_INDO.get(pred, pred)

                st.markdown(f"""
<div style="background:{color}22; border:2px solid {color}; border-radius:12px;
            padding:24px 28px; margin-top:16px; text-align:center;">
    <div style="font-size:1.6rem; font-weight:700; color:{color};">Sentimen: {label}</div>
    <div style="color:#94a3b8; margin-top:6px; font-size:0.95rem;">
        Confidence: <b style="color:{color};">{confidence:.1f}%</b>
    </div>
</div>
""", unsafe_allow_html=True)

                gemini = _get_gemini()
                if gemini:
                    with st.spinner("Menganalisis alasan..."):
                        try:
                            prompt = f"""Kamu adalah sistem analisis sentimen bahasa Indonesia.
Teks: "{user_text}"
Hasil prediksi IndoBERT: {label} (confidence {confidence:.1f}%)
Jelaskan 2-3 kalimat mengapa teks ini diklasifikasikan sebagai sentimen {label}.
Sebutkan kata atau frasa kunci yang menjadi indikator. Bahasa Indonesia, profesional."""
                            response = gemini.generate_content(prompt)
                            st.markdown(
                                f'<div class="insight-text"><b>Analisis:</b> {response.text}</div>',
                                unsafe_allow_html=True
                            )
                        except Exception:
                            pass

            except Exception as e:
                st.error(f"Gagal prediksi: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS DATASET CSV
# ══════════════════════════════════════════════════════════════
with tab_csv:
    st.markdown("Upload dataset CSV — sistem akan memprediksi sentimen menggunakan IndoBERT dan menganalisis hasilnya.")

    uploaded = st.file_uploader("Upload file CSV", type=['csv'], key="csv_uploader")

    if uploaded is None:
        st.info("Mohon upload data CSV terlebih dahulu untuk memulai analisis.")
        with st.expander("Format CSV yang didukung"):
            st.markdown("""
**Wajib:** Kolom teks/komentar (nama bebas: `text`, `comment`, `komentar`, `content`, dll)

**Opsional:**
- Kolom waktu untuk analisis tren (`date`, `create_time`, `tanggal`)
- Kolom sentimen yang sudah ada — sistem akan memprediksi ulang menggunakan IndoBERT dan menampilkan perbandingannya
""")
        st.stop()

    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)

    df_user = load_csv(uploaded)
    st.success(f"File berhasil diupload: **{len(df_user):,} baris**, **{len(df_user.columns)} kolom**")

    all_cols = list(df_user.columns)

    st.markdown('<div class="section-title">Konfigurasi Kolom</div>', unsafe_allow_html=True)
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        text_col = st.selectbox("Kolom Teks/Komentar *", all_cols,
            index=find_col(all_cols, ['comment','text','komentar','content','ulasan','teks']))
    with cp2:
        time_opts = ['— Tidak ada —'] + all_cols
        time_sel  = st.selectbox("Kolom Waktu (opsional)", time_opts)
        time_col  = None if time_sel == '— Tidak ada —' else time_sel
    with cp3:
        sent_opts = ['— Tidak ada —'] + all_cols
        sent_sel  = st.selectbox("Kolom Sentimen (jika ada)", sent_opts)
        sent_col  = None if sent_sel == '— Tidak ada —' else sent_sel

    st.markdown("---")

    if 'user_df_pred' not in st.session_state or st.session_state.get('last_file') != uploaded.name:
        if st.button("Jalankan Prediksi IndoBERT", type="primary", use_container_width=True):
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                BERT_DIR  = os.path.join(MODEL_DIR, 'bert_kdmp_final')
                tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
                bert_m    = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
                clf       = pipeline("text-classification", model=bert_m, tokenizer=tokenizer,
                                      truncation=True, max_length=128, device=-1)

                texts = df_user[text_col].fillna('').astype(str)
                lmap  = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}

                with st.spinner("IndoBERT sedang memprediksi... Mohon tunggu."):
                    results = clf(texts.tolist(), batch_size=32)

                df_user['sentimen_indobert'] = [lmap.get(r['label'], r['label']) for r in results]
                df_user['confidence']        = [round(r['score'] * 100, 1) for r in results]

                if sent_col:
                    df_user['sentimen_user'] = normalize_sentiment(df_user[sent_col])

                st.session_state['user_df_pred'] = df_user.copy()
                st.session_state['last_file']    = uploaded.name
                st.session_state['text_col']     = text_col
                st.session_state['time_col']     = time_col
                st.session_state['sent_col']     = sent_col
                st.rerun()

            except Exception as e:
                st.error(f"Gagal prediksi: {e}")
        st.stop()

    df_pred  = st.session_state['user_df_pred']
    text_col = st.session_state.get('text_col', text_col)
    time_col = st.session_state.get('time_col', time_col)
    sent_col = st.session_state.get('sent_col', sent_col)

    total  = len(df_pred)
    counts = df_pred['sentimen_indobert'].value_counts()
    pos    = counts.get('positive', 0)
    neu    = counts.get('neutral',  0)
    neg    = counts.get('negative', 0)

    st.markdown('<div class="section-title">Ringkasan Hasil Prediksi</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{total:,}")
    c2.metric("Positif",    f"{pos:,}",  f"{pos/total*100:.1f}%")
    c3.metric("Netral",     f"{neu:,}",  f"{neu/total*100:.1f}%")
    c4.metric("Negatif",    f"{neg:,}",  f"{neg/total*100:.1f}%", delta_color="inverse")

    # Perbandingan sentimen user vs IndoBERT
    if sent_col and 'sentimen_user' in df_pred.columns:
        st.markdown('<div class="section-title">Perbandingan Sentimen User vs IndoBERT</div>', unsafe_allow_html=True)
        df_valid = df_pred[df_pred['sentimen_user'].notna()].copy()
        match    = (df_valid['sentimen_user'] == df_valid['sentimen_indobert']).sum()
        pct      = round(match / len(df_valid) * 100, 1) if len(df_valid) > 0 else 0

        st.metric("Tingkat Kecocokan Label", f"{pct}%", f"{match:,} dari {len(df_valid):,} data")

        comp_data = []
        for s in ['positive','neutral','negative']:
            comp_data.append({
                'Sentimen':    LABEL_INDO[s],
                'Label User':  int((df_valid['sentimen_user']     == s).sum()),
                'IndoBERT':    int((df_valid['sentimen_indobert'] == s).sum()),
            })
        df_comp = pd.DataFrame(comp_data)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='Label User', x=df_comp['Sentimen'], y=df_comp['Label User'],
                                   marker_color='#94a3b8', text=df_comp['Label User'], textposition='outside'))
        fig_comp.add_trace(go.Bar(name='IndoBERT',   x=df_comp['Sentimen'], y=df_comp['IndoBERT'],
                                   marker_color='#3498db', text=df_comp['IndoBERT'],   textposition='outside'))
        fig_comp.update_layout(
            barmode='group', height=360,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f4f7fa'), legend=dict(orientation='h', y=-0.2)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown(f"""
<div class="insight-text">
Perbandingan antara label sentimen yang tersedia di data dengan hasil prediksi IndoBERT
menunjukkan tingkat kecocokan sebesar <b>{pct}%</b>.
Perbedaan yang ada mencerminkan kemampuan IndoBERT dalam menangkap nuansa bahasa
yang mungkin tidak tertangkap oleh metode pelabelan sebelumnya.
</div>
""", unsafe_allow_html=True)

    # Distribusi
    st.markdown('<div class="section-title">Distribusi Sentimen (Hasil IndoBERT)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            pd.DataFrame({'Sentimen':['Positif','Netral','Negatif'], 'Jumlah':[pos,neu,neg]}),
            names='Sentimen', values='Jumlah', hole=0.45, color='Sentimen',
            color_discrete_map={'Positif':'#2ecc71','Netral':'#3498db','Negatif':'#e74c3c'},
            title="Proporsi Sentimen"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
        fig_pie.update_layout(height=360, legend=dict(orientation='h', y=-0.15),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#f4f7fa'))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(
            pd.DataFrame({
                'Sentimen':['Positif','Netral','Negatif'], 'Jumlah':[pos,neu,neg],
                'Persen':[f"{pos/total*100:.1f}%", f"{neu/total*100:.1f}%", f"{neg/total*100:.1f}%"]
            }),
            x='Sentimen', y='Jumlah', color='Sentimen', text='Persen',
            color_discrete_map={'Positif':'#2ecc71','Netral':'#3498db','Negatif':'#e74c3c'},
            title="Jumlah Data per Sentimen"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, height=360,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#f4f7fa'))
        st.plotly_chart(fig_bar, use_container_width=True)

    dom_label = LABEL_INDO.get(
        max({'positive':pos,'neutral':neu,'negative':neg},
            key=lambda k: {'positive':pos,'neutral':neu,'negative':neg}[k]), '-'
    )
    dom_pct = max(pos, neu, neg) / total * 100
    st.markdown(f"""
<div class="insight-text">
Dari <b>{total:,} data</b> yang dianalisis menggunakan IndoBERT,
sentimen <b>{dom_label} mendominasi dengan {dom_pct:.1f}%</b>.
Distribusi ini mencerminkan opini keseluruhan yang terkandung dalam dataset yang diupload.
</div>
""", unsafe_allow_html=True)

    # Tren Waktu
    if time_col:
        st.markdown('<div class="section-title">Tren Sentimen per Waktu</div>', unsafe_allow_html=True)
        try:
            df_pred['_time']  = pd.to_datetime(df_pred[time_col], errors='coerce')
            df_pred['_month'] = df_pred['_time'].dt.to_period('M').astype(str)
            monthly = df_pred.groupby(['_month','sentimen_indobert']).size().reset_index(name='count')
            pivot   = monthly.pivot(index='_month', columns='sentimen_indobert', values='count').fillna(0).reset_index()
            fig_t   = go.Figure()
            for c, color, label in [('positive','#2ecc71','Positif'),('neutral','#3498db','Netral'),('negative','#e74c3c','Negatif')]:
                if c in pivot.columns:
                    fig_t.add_trace(go.Scatter(x=pivot['_month'], y=pivot[c], name=label,
                                                line=dict(color=color, width=2.5),
                                                mode='lines+markers', marker=dict(size=7)))
            fig_t.update_layout(
                height=380, hovermode='x unified', legend=dict(orientation='h', y=-0.2),
                xaxis_title="Bulan", yaxis_title="Jumlah Data",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f4f7fa')
            )
            st.plotly_chart(fig_t, use_container_width=True)
            st.markdown("""
<div class="insight-text">
Grafik tren menunjukkan perubahan distribusi sentimen dari waktu ke waktu pada data yang diupload.
Perhatikan periode dengan lonjakan sentimen tertentu yang dapat mengindikasikan
respons terhadap peristiwa atau informasi spesifik.
</div>
""", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Tren waktu tidak dapat ditampilkan: {e}")

    # WordCloud
    word_col = next(
        (c for c in df_pred.columns if c.lower() in ['content_clean','text_clean','clean_text']),
        text_col
    )
    st.markdown('<div class="section-title">WordCloud per Sentimen</div>', unsafe_allow_html=True)
    wc1, wc2, wc3 = st.columns(3)
    for wcol, sent, title, cmap in [
        (wc1,'positive','Positif','Greens'),
        (wc2,'neutral','Netral','Blues'),
        (wc3,'negative','Negatif','Reds')
    ]:
        with wcol:
            st.markdown(f"**Sentimen {title}**")
            texts = df_pred[df_pred['sentimen_indobert'] == sent][word_col].dropna().astype(str)
            if not texts.empty:
                wc = WordCloud(width=600, height=300, background_color='white',
                               max_words=60, colormap=cmap, collocations=False).generate(' '.join(texts))
                fig_wc, ax = plt.subplots(figsize=(5, 2.5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc, use_container_width=True)
                plt.close()
            else:
                st.info("Tidak ada data.")
    st.markdown("""
<div class="insight-text">
WordCloud menampilkan kata-kata yang paling sering muncul pada masing-masing kategori sentimen
berdasarkan data yang diupload. Ukuran kata mencerminkan frekuensi kemunculannya.
</div>
""", unsafe_allow_html=True)

    # Top 15 Kata
    st.markdown('<div class="section-title">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    tk1, tk2, tk3 = st.columns(3)
    for tcol, sent, color, title in [
        (tk1,'positive','#2ecc71','Positif'),
        (tk2,'neutral','#3498db','Netral'),
        (tk3,'negative','#e74c3c','Negatif')
    ]:
        with tcol:
            texts = df_pred[df_pred['sentimen_indobert'] == sent][word_col].dropna().astype(str)
            if not texts.empty:
                words = ' '.join(texts).split()
                top   = Counter(words).most_common(15)
                wdf   = pd.DataFrame(top, columns=['Kata','Frekuensi'])
                fig   = px.bar(wdf, x='Frekuensi', y='Kata', orientation='h',
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
Grafik frekuensi kata memberikan gambaran lebih terukur dibanding WordCloud —
memperlihatkan secara eksplisit kata mana yang paling sering mendorong
masing-masing kategori sentimen pada data yang dianalisis.
</div>
""", unsafe_allow_html=True)

    # Gemini Insight
    st.markdown('<div class="section-title">Insight & Rekomendasi AI</div>', unsafe_allow_html=True)
    st.markdown("Klik tombol di bawah untuk menghasilkan analisis mendalam dari Gemini berdasarkan data yang diupload.")

    if st.button("Generate Insight", type="primary", use_container_width=True, key="btn_insight"):
        with st.spinner("Gemini sedang menganalisis data..."):
            result, source = generate_insight(df_pred.rename(columns={'sentimen_indobert': '_sentiment'}))
        st.session_state['user_insight'] = result

    if 'user_insight' in st.session_state:
        result = st.session_state['user_insight']
        if isinstance(result, dict) and result.get('text'):
            st.markdown(f'<div class="gemini-box">{result["text"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
<div style="background:rgba(155,89,182,0.1);border:1px solid rgba(155,89,182,0.4);border-radius:10px;padding:18px;">
<b style="color:#9b59b6;">Insight</b><br><br>{result.get('insight','-')}
</div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
<div style="background:rgba(241,196,15,0.08);border:1px solid rgba(241,196,15,0.35);border-radius:10px;padding:18px;">
<b style="color:#f1c40f;">Kesimpulan</b><br><br>{result.get('conclusion','-')}
</div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
<div style="background:rgba(46,204,113,0.08);border:1px solid rgba(46,204,113,0.35);border-radius:10px;padding:18px;">
<b style="color:#2ecc71;">Rekomendasi</b><br><br>{result.get('recommendation','-')}
</div>""", unsafe_allow_html=True)

    # Download
    st.markdown('<div class="section-title">Unduh Hasil Prediksi</div>', unsafe_allow_html=True)
    df_out = df_pred.copy()
    for col in df_out.select_dtypes(include=['datetimetz']).columns:
        df_out[col] = df_out[col].dt.tz_localize(None)

    st.dataframe(df_out.head(20), use_container_width=True)

    csv_bytes    = df_out.to_csv(index=False).encode('utf-8')
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='Prediksi')
    excel_buffer.seek(0)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Unduh CSV",   csv_bytes, "sentiment_predictions.csv",
                           "text/csv", use_container_width=True)
    with dl2:
        st.download_button("Unduh Excel", excel_buffer.getvalue(), "sentiment_predictions.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

st.markdown("---")
st.caption("Dashboard Analisis Sentimen KDMP • Tugas Akhir Teknik Informatika UNESA")