import os
import pandas as pd
from collections import Counter
from dotenv import load_dotenv

# Muat file .env supaya GEMINI_API_KEY tersedia saat modul diimpor
load_dotenv()

# ── Coba import Gemini ───────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Setup Gemini (panggil sekali) ────────────────────────────
_gemini_model = None

def _get_gemini():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    if not GEMINI_AVAILABLE:
        return None
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        return _gemini_model
    except Exception:
        return None


# ════════════════════════════════════════════════════════════
# 1. BUILD INSIGHT PAYLOAD
# ════════════════════════════════════════════════════════════
def build_insight_payload(df: pd.DataFrame) -> dict:
    sent_col = None
    for c in ['_sentiment', 'sentiment', 'label', 'sentimen']:
        if c in df.columns:
            sent_col = c
            break
    if sent_col is None:
        return {
            'total': len(df), 'positive': 0, 'neutral': 0, 'negative': 0,
            'pct_pos': 0, 'pct_neu': 0, 'pct_neg': 0,
            'top_positive': [], 'top_neutral': [], 'top_negative': [],
            'dominant': 'unknown', 'tren': {}
        }

    label_norm = {
        'positif': 'positive', 'positive': 'positive',
        'netral':  'neutral',  'neutral':  'neutral',
        'negatif': 'negative', 'negative': 'negative',
    }
    df2 = df.copy()
    df2['_s'] = df2[sent_col].astype(str).str.lower().str.strip().map(label_norm)
    df2 = df2[df2['_s'].notna()]

    total = len(df2)
    if total == 0:
        return {'total': 0, 'positive': 0, 'neutral': 0, 'negative': 0,
                'pct_pos': 0, 'pct_neu': 0, 'pct_neg': 0,
                'top_positive': [], 'top_neutral': [], 'top_negative': [],
                'dominant': 'unknown', 'tren': {}}

    counts = df2['_s'].value_counts()
    pos    = int(counts.get('positive', 0))
    neu    = int(counts.get('neutral',  0))
    neg    = int(counts.get('negative', 0))

    word_col = None
    for c in ['content_clean', 'text_clean', 'clean_text',
              'comment', 'text', 'komentar', 'content']:
        if c in df2.columns:
            word_col = c
            break

    def top_words(sentiment, n=5):
        if word_col is None:
            return []
        texts = df2[df2['_s'] == sentiment][word_col].dropna().astype(str)
        if texts.empty:
            return []
        words = [w for w in ' '.join(texts).split() if len(w) > 3]
        return [w for w, _ in Counter(words).most_common(n)]

    tren = {}
    for c in ['create_time', 'date', 'created_at', 'tanggal', 'time']:
        if c in df2.columns:
            try:
                df2['_t'] = pd.to_datetime(df2[c], errors='coerce')
                df2['_m'] = df2['_t'].dt.to_period('M').astype(str)
                monthly   = df2.groupby(['_m', '_s']).size().unstack(fill_value=0)
                tren = monthly.tail(6).to_dict()
            except Exception:
                pass
            break

    dominant = max({'positive': pos, 'neutral': neu, 'negative': neg},
                   key=lambda k: {'positive': pos, 'neutral': neu, 'negative': neg}[k])

    return {
        'total'       : total,
        'positive'    : pos,
        'neutral'     : neu,
        'negative'    : neg,
        'pct_pos'     : round(pos / total * 100, 1),
        'pct_neu'     : round(neu / total * 100, 1),
        'pct_neg'     : round(neg / total * 100, 1),
        'top_positive': top_words('positive'),
        'top_neutral' : top_words('neutral'),
        'top_negative': top_words('negative'),
        'dominant'    : dominant,
        'tren'        : tren,
    }


# ════════════════════════════════════════════════════════════
# 2. BUILD DATA SUMMARY (untuk chatbot context)
# ════════════════════════════════════════════════════════════
def build_data_summary(df: pd.DataFrame) -> str:
    p = build_insight_payload(df)
    return f"""
=== RINGKASAN DATA SENTIMEN KDMP ===
Total komentar : {p['total']:,}
Positif        : {p['positive']:,} ({p['pct_pos']}%)
Netral         : {p['neutral']:,}  ({p['pct_neu']}%)
Negatif        : {p['negative']:,} ({p['pct_neg']}%)
Sentimen dominan: {p['dominant']}

Kata dominan POSITIF : {', '.join(p['top_positive']) if p['top_positive'] else '-'}
Kata dominan NETRAL  : {', '.join(p['top_neutral'])  if p['top_neutral']  else '-'}
Kata dominan NEGATIF : {', '.join(p['top_negative']) if p['top_negative'] else '-'}
=====================================
""".strip()


# ════════════════════════════════════════════════════════════
# 3. GENERATE INSIGHT
# ════════════════════════════════════════════════════════════
def _template_insight(p: dict) -> dict:
    dom     = p['dominant']
    total   = p['total']
    pos     = p['pct_pos']
    neu     = p['pct_neu']
    neg     = p['pct_neg']
    top_neg = ', '.join(p['top_negative'][:3]) if p['top_negative'] else 'tidak teridentifikasi'
    top_pos = ', '.join(p['top_positive'][:3]) if p['top_positive'] else 'tidak teridentifikasi'

    if dom == 'negative':
        return {
            'insight': (
                f"Dari total {total:,} komentar TikTok yang dianalisis, mayoritas masyarakat "
                f"memberikan respons negatif terhadap program KDMP sebesar {neg}%. "
                f"Kata yang mendominasi komentar negatif: {top_neg}. "
                f"Hal ini mengindikasikan masih banyak kritik dan ketidakpuasan masyarakat."
            ),
            'conclusion': (
                f"Program KDMP masih mendapat sorotan negatif dari masyarakat ({neg}%), "
                f"meskipun ada yang mendukung ({pos}%). "
                f"Evaluasi menyeluruh terhadap implementasi program sangat diperlukan."
            ),
            'recommendation': (
                f"Berdasarkan kata dominan negatif ({top_neg}), pemerintah perlu meningkatkan "
                f"transparansi informasi, memperbaiki komunikasi publik, dan mempercepat "
                f"realisasi manfaat nyata program KDMP di tingkat desa."
            )
        }
    elif dom == 'positive':
        return {
            'insight': (
                f"Dari total {total:,} komentar, mayoritas masyarakat memberikan respons positif "
                f"terhadap KDMP sebesar {pos}%. Kata dominan positif: {top_pos}. "
                f"Program secara umum diterima dan diapresiasi masyarakat."
            ),
            'conclusion': (
                f"Program KDMP mendapat sambutan positif dari masyarakat ({pos}%), "
                f"mencerminkan relevansi dan manfaat program. "
                f"Komentar negatif ({neg}%) tetap perlu diperhatikan."
            ),
            'recommendation': (
                f"Program KDMP perlu dipertahankan dan diperluas. "
                f"Fokus pada aspek yang masih dikritik ({top_neg}) dapat meningkatkan "
                f"kepuasan masyarakat secara keseluruhan."
            )
        }
    else:
        return {
            'insight': (
                f"Dari total {total:,} komentar, sentimen masyarakat berimbang "
                f"dengan dominasi netral ({neu}%). Positif {pos}% dan negatif {neg}%, "
                f"mengindikasikan masyarakat masih dalam tahap observasi."
            ),
            'conclusion': (
                f"Program KDMP belum membentuk kesan kuat di masyarakat (netral {neu}%). "
                f"Masyarakat menunggu bukti nyata sebelum memberikan penilaian definitif."
            ),
            'recommendation': (
                f"Tingkatkan sosialisasi dan komunikasi manfaat konkret KDMP. "
                f"Publikasi capaian nyata dan testimoni penerima manfaat dapat mendorong "
                f"pergeseran dari sentimen netral ke positif."
            )
        }


def generate_insight(df: pd.DataFrame):
    """
    Generate insight dari data.
    Returns: (result_dict, source_str)
    """
    p       = build_insight_payload(df)
    summary = build_data_summary(df)
    model   = _get_gemini()

    if model is not None:
        try:
            prompt = f"""Kamu adalah analis sentimen program pemerintah Indonesia.

Berdasarkan data analisis sentimen komentar TikTok tentang program 
Koperasi Desa Merah Putih (KDMP):

{summary}

Buatkan analisis dengan format TEPAT berikut:

**Insight:**
[Jelaskan kondisi sentimen secara objektif berdasarkan data, sebutkan angka dan kata dominan]

**Kesimpulan:**
[Simpulkan makna dari pola sentimen terhadap penerimaan masyarakat atas program KDMP]

**Rekomendasi:**
[Berikan 2-3 rekomendasi konkret berdasarkan temuan data]

Bahasa Indonesia profesional, maksimal 4 kalimat per bagian."""

            response = model.generate_content(prompt)
            return {'text': response.text}, 'Gemini'
        except Exception:
            pass

    return _template_insight(p), 'Template'


# ════════════════════════════════════════════════════════════
# 4. INSIGHT PER GRAFIK
# ════════════════════════════════════════════════════════════
def insight_distribusi(p: dict) -> str:
    dom = p['dominant']
    pos, neu, neg = p['pct_pos'], p['pct_neu'], p['pct_neg']
    if dom == 'negative':
        return (
            f"📌 **Interpretasi:** Distribusi sentimen didominasi negatif **{neg}%**, "
            f"menandakan banyak kritik terhadap KDMP. "
            f"Positif {pos}% dan netral {neu}%."
        )
    elif dom == 'positive':
        return (
            f"📌 **Interpretasi:** Program KDMP mendapat respons positif **{pos}%**, "
            f"menandakan penerimaan yang baik. "
            f"Komentar negatif ({neg}%) tetap perlu diperhatikan."
        )
    else:
        return (
            f"📌 **Interpretasi:** Sentimen berimbang dengan dominasi netral **{neu}%**, "
            f"mengindikasikan masyarakat masih observasi. "
            f"Positif {pos}%, negatif {neg}%."
        )


def insight_tren(df: pd.DataFrame) -> str:
    sent_col = next((c for c in ['_sentiment','sentiment'] if c in df.columns), None)
    time_col = next((c for c in ['create_time','date','tanggal'] if c in df.columns), None)
    if sent_col is None or time_col is None:
        return "📌 **Interpretasi:** Data tren tidak tersedia."
    try:
        df2 = df.copy()
        df2['_t'] = pd.to_datetime(df2[time_col], errors='coerce')
        df2['_m'] = df2['_t'].dt.to_period('M').astype(str)
        monthly   = df2.groupby(['_m', sent_col]).size().unstack(fill_value=0)
        if monthly.empty:
            return "📌 **Interpretasi:** Data tren tidak cukup."
        last_month    = monthly.index[-1]
        dominant_last = monthly.loc[last_month].idxmax()
        if 'negative' in monthly.columns and len(monthly) >= 2:
            neg_trend = "meningkat" if monthly['negative'].values[-1] > monthly['negative'].values[-2] else "menurun"
        else:
            neg_trend = "stabil"
        return (
            f"📌 **Interpretasi:** Pada **{last_month}**, sentimen dominan adalah **{dominant_last}**. "
            f"Tren sentimen negatif cenderung **{neg_trend}** dibanding bulan sebelumnya."
        )
    except Exception:
        return "📌 **Interpretasi:** Tren tidak dapat dianalisis otomatis."


def insight_wordcloud(p: dict) -> str:
    top_neg = ', '.join(p['top_negative'][:3]) if p['top_negative'] else '-'
    top_pos = ', '.join(p['top_positive'][:3]) if p['top_positive'] else '-'
    return (
        f"📌 **Interpretasi:** Kata dominan negatif (**{top_neg}**) mencerminkan aspek yang paling dikeluhkan. "
        f"Kata dominan positif (**{top_pos}**) menggambarkan hal yang diapresiasi masyarakat."
    )


def insight_top_kata(p: dict) -> str:
    top_neg = ', '.join(p['top_negative'][:5]) if p['top_negative'] else '-'
    top_pos = ', '.join(p['top_positive'][:5]) if p['top_positive'] else '-'
    return (
        f"📌 **Interpretasi:** Kata negatif dominan ({top_neg}) menjadi indikator area yang perlu diperbaiki. "
        f"Kata positif dominan ({top_pos}) menunjukkan kekuatan program di mata masyarakat."
    )


# ════════════════════════════════════════════════════════════
# 5. CHATBOT
# ════════════════════════════════════════════════════════════
def rule_based_answer(question: str, df: pd.DataFrame):
    q = question.lower().strip()
    p = build_insight_payload(df)
    total = p['total']

    if any(k in q for k in ['total', 'berapa data', 'jumlah data', 'berapa baris']):
        return f"Total data yang dianalisis adalah **{total:,} komentar**."
    if any(k in q for k in ['persen negatif', 'negatif berapa', '% negatif', 'berapa negatif']):
        return f"Sentimen negatif: **{p['pct_neg']}%** ({p['negative']:,} komentar)."
    if any(k in q for k in ['persen positif', 'positif berapa', '% positif', 'berapa positif']):
        return f"Sentimen positif: **{p['pct_pos']}%** ({p['positive']:,} komentar)."
    if any(k in q for k in ['persen netral', 'netral berapa', '% netral', 'berapa netral']):
        return f"Sentimen netral: **{p['pct_neu']}%** ({p['neutral']:,} komentar)."
    if any(k in q for k in ['paling dominan', 'terbanyak', 'mayoritas', 'paling banyak']):
        dom_pct = {'positive': p['pct_pos'], 'neutral': p['pct_neu'], 'negative': p['pct_neg']}[p['dominant']]
        return f"Sentimen dominan: **{p['dominant']}** ({dom_pct}%)."
    if any(k in q for k in ['kata', 'keyword', 'dominan', 'sering muncul']):
        return (
            f"Kata paling sering muncul:\n"
            f"- **Positif:** {', '.join(p['top_positive']) or '-'}\n"
            f"- **Netral:** {', '.join(p['top_neutral']) or '-'}\n"
            f"- **Negatif:** {', '.join(p['top_negative']) or '-'}"
        )
    if any(k in q for k in ['distribusi', 'breakdown', 'rincian', 'semua sentimen']):
        return (
            f"Distribusi dari **{total:,}** komentar:\n"
            f"- 😊 Positif : {p['positive']:,} ({p['pct_pos']}%)\n"
            f"- 😐 Netral  : {p['neutral']:,}  ({p['pct_neu']}%)\n"
            f"- 😠 Negatif : {p['negative']:,} ({p['pct_neg']}%)"
        )
    return None


def ask_gemini(question: str, summary: str, history: list) -> str:
    model = _get_gemini()
    if model is None:
        return (
            "⚠️ Gemini API tidak tersedia. Set `GEMINI_API_KEY` di file `.env`. "
            "API key gratis: https://aistudio.google.com/apikey"
        )
    system = f"""Kamu asisten analisis sentimen KDMP. 
Jawab HANYA berdasarkan data berikut, bahasa Indonesia santai:

{summary}

Jika tidak relevan, tolak sopan. Jangan karang data di luar ringkasan."""
    try:
        gemini_history = []
        for msg in history[-6:]:
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [msg['content']]})
        chat     = model.start_chat(history=gemini_history)
        response = chat.send_message(f"{system}\n\nPertanyaan: {question}")
        return response.text
    except Exception as e:
        return f"⚠️ Gagal menghubungi Gemini: {str(e)}"