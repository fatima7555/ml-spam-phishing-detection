"""
PhishGuard Flask web application.

Routes:
    GET  /              → serve index.html
    POST /predict/email → classify email text (spam / legitimate)
    POST /predict/url   → classify URL (phishing / legitimate)

Models are loaded once at startup to minimise prediction latency.
"""

import os
import sys
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.preprocessing.url_preprocessor import URLPreprocessor

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Load artefacts once at startup                                               #
# --------------------------------------------------------------------------- #
_BASE = os.path.join(os.path.dirname(__file__), "..")
_MODELS_DIR = os.path.join(_BASE, "saved_models")

_email_model = None
_email_extractor = None
_url_model = None
_url_extractor = None
_email_preprocessor = EmailPreprocessor()
_url_preprocessor = URLPreprocessor()


def _load_models():
    global _email_model, _email_extractor, _url_model, _url_extractor
    try:
        _email_model = joblib.load(os.path.join(_MODELS_DIR, "email_model.pkl"))
        _email_extractor = joblib.load(os.path.join(_MODELS_DIR, "email_vectorizer.pkl"))
        print("Email classifier loaded.")
    except FileNotFoundError:
        print("WARNING: email model not found. Run train_email.py first.")

    try:
        _url_model = joblib.load(os.path.join(_MODELS_DIR, "url_model.pkl"))
        _url_extractor = joblib.load(os.path.join(_MODELS_DIR, "url_vectorizer.pkl"))
        print("URL classifier loaded.")
    except FileNotFoundError:
        print("WARNING: URL model not found. Run train_url.py first.")


_load_models()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _confidence(model, X) -> float:
    """Return prediction probability for the predicted class [0, 1]."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return float(np.max(proba))
    # CalibratedClassifierCV always has predict_proba, but fallback for safety
    df = model.decision_function(X)[0]
    if hasattr(df, "__len__"):
        df = float(np.max(df))
    return float(1 / (1 + np.exp(-df)))  # sigmoid


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/email", methods=["POST"])
def predict_email():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if _email_model is None:
        return jsonify({"error": "Email model not loaded. Run train_email.py."}), 503

    cleaned = _email_preprocessor.transform([text])
    X = _email_extractor.transform(cleaned, [text])
    pred = int(_email_model.predict(X)[0])
    conf = _confidence(_email_model, X)

    return jsonify({
        "label": "SPAM" if pred == 1 else "LEGITIMATE",
        "is_malicious": pred == 1,
        "confidence": round(conf * 100, 1),
    })


@app.route("/predict/url", methods=["POST"])
def predict_url():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400
    if _url_model is None:
        return jsonify({"error": "URL model not loaded. Run train_url.py."}), 503

    tokenized = _url_preprocessor.transform([url])
    X = _url_extractor.transform(tokenized, [url])
    pred = int(_url_model.predict(X)[0])
    conf = _confidence(_url_model, X)

    return jsonify({
        "label": "PHISHING" if pred == 1 else "LEGITIMATE",
        "is_malicious": pred == 1,
        "confidence": round(conf * 100, 1),
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "email_model_loaded": _email_model is not None,
        "url_model_loaded": _url_model is not None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
