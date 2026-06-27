import streamlit as st
import os
from utils.chatbot import build_data_summary, ask_gemini, rule_based_answer

st.set_page_config(page_title="Chatbot — KDMP", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
[data-testid="stSidebar"] * { color: #f4f7fa !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.button("Hapus Riwayat Chat", use_container_width=True):
        st.session_state['chat_history'] = []
        st.rerun()
    st.caption("Tugas Akhir • Teknik Informatika • UNESA")

# ── Header ────────────────────────────────────────────────────
st.title("Chatbot Analisis Sentimen")

# ── Guard ─────────────────────────────────────────────────────
if 'user_df_pred' not in st.session_state:
    st.warning("Mohon upload data CSV terlebih dahulu di halaman User Analysis.")
    st.page_link("pages/1_User_Analysis.py", label="Pergi ke User Analysis")
    st.stop()

df_user = st.session_state['user_df_pred']

# ── Metric ringkas ────────────────────────────────────────────
total  = len(df_user)
counts = df_user['sentimen_indobert'].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Data", f"{total:,}")
c2.metric("Positif",    f"{counts.get('positive',0):,}")
c3.metric("Netral",     f"{counts.get('neutral',0):,}")
c4.metric("Negatif",    f"{counts.get('negative',0):,}")

st.markdown("---")

# ── Contoh pertanyaan ─────────────────────────────────────────
st.markdown("**Contoh pertanyaan:**")
examples = [
    "Berapa total data yang dianalisis?",
    "Berapa persen sentimen negatif?",
    "Sentimen apa yang paling dominan?",
    "Kata apa yang paling sering muncul di sentimen negatif?",
    "Bagaimana distribusi sentimennya?",
    "Apa kesimpulan dari data ini?",
]
ex_cols = st.columns(3)
for i, q in enumerate(examples):
    with ex_cols[i % 3]:
        if st.button(q, use_container_width=True, key=f"q_{i}"):
            st.session_state['quick_q'] = q

st.markdown("---")

# ── Chat ──────────────────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for msg in st.session_state['chat_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if not st.session_state['chat_history']:
    with st.chat_message("assistant"):
        st.markdown("Halo! Saya siap menjawab pertanyaan seputar data sentimen yang sudah dianalisis. Silakan bertanya.")

quick_q    = st.session_state.pop('quick_q', None)
user_input = st.chat_input("Tanya sesuatu tentang data kamu...")
question   = user_input or quick_q

if question:
    st.session_state['chat_history'].append({'role': 'user', 'content': question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Memproses..."):
            df_renamed = df_user.rename(columns={'sentimen_indobert': '_sentiment'})
            answer     = rule_based_answer(question, df_renamed)
            if answer is None:
                summary = build_data_summary(df_renamed)
                answer  = ask_gemini(question, summary, st.session_state['chat_history'])
        st.markdown(answer)
        st.session_state['chat_history'].append({'role': 'assistant', 'content': answer})

st.markdown("---")
st.caption("Dashboard Analisis Sentimen KDMP • Tugas Akhir Teknik Informatika UNESA")