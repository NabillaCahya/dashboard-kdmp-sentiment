import re
import string
import numpy as np

# Lazy load Sastrawi biar startup cepet
_stopword_remover = None
_stemmer = None

def get_stopword_remover():
    global _stopword_remover
    if _stopword_remover is None:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        _stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return _stopword_remover

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer

def clean_text_basic(text: str) -> str:
    """Basic cleaning: lowercase, hapus URL, mention, hashtag, angka, punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # hapus URL
    text = re.sub(r'@\w+', '', text)                  # hapus mention
    text = re.sub(r'#\w+', '', text)                  # hapus hashtag
    text = re.sub(r'\d+', '', text)                   # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_ml(text: str) -> str:
    """Full preprocessing untuk model ML klasik (sama persis dengan Notebook 2)."""
    text = clean_text_basic(text)
    text = get_stopword_remover().remove(text)
    text = get_stemmer().stem(text)
    return text

def preprocess_bert(text: str) -> str:
    """Preprocessing minimal untuk IndoBERT (hanya basic cleaning, tidak di-stem)."""
    return clean_text_basic(text)

LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}
IDX_MAP   = {0: 'positive', 1: 'neutral', 2: 'negative'}
LABEL_COLOR = {
    'positive': '#2ecc71',
    'neutral' : '#3498db',
    'negative': '#e74c3c',
}
LABEL_EMOJI = {
    'positive': '😊 Positif',
    'neutral' : '😐 Netral',
    'negative': '😠 Negatif',
}
