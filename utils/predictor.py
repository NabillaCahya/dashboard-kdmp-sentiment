import pickle
import os
import numpy as np
import streamlit as st
from utils.preprocessing import preprocess_ml, preprocess_bert, IDX_MAP

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ─── Load ML Models ──────────────────────────────────────────────────────────

@st.cache_resource
def load_tfidf():
    path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_ml_models():
    models = {}
    model_files = {
        'Naive Bayes'        : 'model_naive_bayes.pkl',
        'SVM'                : 'model_svm.pkl',
        'Logistic Regression': 'model_logistic_regression.pkl',
        'Decision Tree'      : 'model_decision_tree.pkl',
    }
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    return models

@st.cache_resource
def load_bert_model():
    """Load IndoBERT fine-tuned dari folder bert_kdmp_final/."""
    bert_dir = os.path.join(MODELS_DIR, 'bert_kdmp_final')
    if not os.path.exists(bert_dir):
        return None, None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"Gagal load IndoBERT: {e}")
        return None, None

# ─── Prediction Functions ─────────────────────────────────────────────────────

def predict_ml(text: str, model_name: str):
    """
    Returns: (label: str, proba: dict or None)
    proba keys: 'positive', 'neutral', 'negative'
    """
    tfidf = load_tfidf()
    models = load_ml_models()

    if model_name not in models:
        return None, None

    m = models[model_name]
    text_clean = preprocess_ml(text)
    vec = tfidf.transform([text_clean])

    pred = m.predict(vec)[0]           # int dari LabelEncoder (0/1/2) atau string
    label = IDX_MAP.get(pred, pred) if isinstance(pred, int) else pred

    proba = None
    if hasattr(m, 'predict_proba'):
        p = m.predict_proba(vec)[0]
        classes = m.classes_
        # classes bisa int atau string tergantung LabelEncoder
        if isinstance(classes[0], (int, np.integer)):
            proba = {IDX_MAP[int(c)]: float(v) for c, v in zip(classes, p)}
        else:
            proba = {str(c): float(v) for c, v in zip(classes, p)}

    return label, proba

def predict_bert(text: str):
    """
    Returns: (label: str, proba: dict)
    proba keys: 'positive', 'neutral', 'negative'
    """
    import torch
    tokenizer, model = load_bert_model()
    if tokenizer is None:
        return None, None

    text_clean = preprocess_bert(text)
    enc = tokenizer(
        text_clean,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1).numpy()[0]
    pred_idx = int(probs.argmax())
    label = IDX_MAP[pred_idx]
    proba = {IDX_MAP[i]: float(probs[i]) for i in range(3)}
    return label, proba

def predict_all_models(text: str):
    """Prediksi dengan semua model sekaligus. Returns list of dicts."""
    results = []
    models = load_ml_models()

    for name in models:
        label, proba = predict_ml(text, name)
        if label is not None:
            results.append({
                'model' : name,
                'label' : label,
                'proba' : proba,
                'is_bert': False,
            })

    # IndoBERT
    label, proba = predict_bert(text)
    if label is not None:
        results.append({
            'model' : 'IndoBERT Fine-tuned',
            'label' : label,
            'proba' : proba,
            'is_bert': True,
        })

    return results
