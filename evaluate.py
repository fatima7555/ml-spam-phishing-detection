"""
Standalone evaluation script.

Loads saved models and prints a full comparison report suitable for
copying into the IEEE project report. Run after both training scripts.

Usage:
    python evaluate.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.preprocessing.url_preprocessor import URLPreprocessor


def _separator(char="=", width=70):
    print(char * width)


def evaluate_email():
    _separator()
    print("EMAIL SPAM CLASSIFIER — EVALUATION REPORT")
    _separator()

    model_path = os.path.join("saved_models", "email_model.pkl")
    vec_path = os.path.join("saved_models", "email_vectorizer.pkl")
    data_path = os.path.join("data", "raw", "sms_spam.csv")

    if not all(os.path.exists(p) for p in [model_path, vec_path, data_path]):
        print("Missing files. Run: python data/download_data.py && python train_email.py")
        return

    model = joblib.load(model_path)
    extractor = joblib.load(vec_path)

    df = pd.read_csv(data_path).dropna(subset=["label", "text"])
    df["label_enc"] = (df["label"].str.lower().str.strip() == "spam").astype(int)

    X_text = df["text"].tolist()
    y = df["label_enc"].tolist()
    _, X_te_raw, _, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = EmailPreprocessor()
    X_te_clean = preprocessor.transform(X_te_raw)
    X_test = extractor.transform(X_te_clean, X_te_raw)
    y_pred = model.predict(X_test)

    print(f"\nTest set size : {len(y_test)} messages")
    print(f"Spam in test  : {sum(y_test)} ({np.mean(y_test):.1%})\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (Legitimate)", "Spam"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("              Ham   Spam")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Ham   {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Spam  {cm[1][0]:>6}  {cm[1][1]:>6}")
    print(f"\nMacro F1  : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='macro'):.4f}")


def evaluate_url():
    _separator()
    print("PHISHING URL CLASSIFIER — EVALUATION REPORT")
    _separator()

    model_path = os.path.join("saved_models", "url_model.pkl")
    vec_path = os.path.join("saved_models", "url_vectorizer.pkl")
    data_path = os.path.join("data", "raw", "phishing_urls.csv")

    if not all(os.path.exists(p) for p in [model_path, vec_path, data_path]):
        print("Missing files. Run: python data/download_data.py && python train_url.py")
        return

    model = joblib.load(model_path)
    extractor = joblib.load(vec_path)

    df = pd.read_csv(data_path).dropna(subset=["url", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df[df["label"].isin([0, 1])]

    X_urls = df["url"].tolist()
    y = df["label"].tolist()
    _, X_te_raw, _, y_test = train_test_split(
        X_urls, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = URLPreprocessor()
    X_te_tok = preprocessor.transform(X_te_raw)
    X_test = extractor.transform(X_te_tok, X_te_raw)
    y_pred = model.predict(X_test)

    print(f"\nTest set size    : {len(y_test)} URLs")
    print(f"Phishing in test : {sum(y_test)} ({np.mean(y_test):.1%})\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("                 Legit   Phish")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Legitimate {cm[0][0]:>8}  {cm[0][1]:>7}")
    print(f"  Phishing   {cm[1][0]:>8}  {cm[1][1]:>7}")
    print(f"\nMacro F1  : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='macro'):.4f}")


if __name__ == "__main__":
    evaluate_email()
    print()
    evaluate_url()
