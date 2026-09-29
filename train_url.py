"""
Phishing URL classifier training pipeline.

Steps:
  1. Load phishing URL dataset
  2. Preprocess URLs (tokenize for TF-IDF)
  3. Extract features (TF-IDF char n-grams + 15 hand-crafted URL features)
  4. Compare 5 ML models with 5-fold GridSearchCV
  5. Select best model via 3-tier criterion
  6. Save model + feature extractor to saved_models/
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing.url_preprocessor import URLPreprocessor
from src.features.url_features import URLFeatureExtractor
from src.models.model_selector import run_all_models, select_best_model

DATA_PATH = os.path.join("data", "raw", "phishing_urls.csv")
MODEL_PATH = os.path.join("saved_models", "url_model.pkl")
VECTORIZER_PATH = os.path.join("saved_models", "url_vectorizer.pkl")


def main():
    print("=" * 60)
    print("PHISHING URL CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run: python data/download_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["url", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df[df["label"].isin([0, 1])]

    print(f"\nDataset: {len(df)} URLs")
    print(f"  Phishing:   {df['label'].sum()} ({df['label'].mean():.1%})")
    print(f"  Legitimate: {(1 - df['label']).sum()}")

    # 2. Train/test split
    X_urls = df["url"].tolist()
    y = df["label"].tolist()
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_urls, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Preprocess URLs
    print("\nTokenizing URLs...")
    preprocessor = URLPreprocessor()
    X_tr_tok = preprocessor.transform(X_tr_raw)
    X_te_tok = preprocessor.transform(X_te_raw)

    # 4. Feature extraction
    print("Extracting features (TF-IDF char n-grams + 15 hand-crafted features)...")
    extractor = URLFeatureExtractor(max_features=10_000)
    X_train = extractor.fit_transform(X_tr_tok, X_tr_raw)
    X_test = extractor.transform(X_te_tok, X_te_raw)
    print(f"  Feature matrix shape: {X_train.shape}")

    # 5. Train & compare all models
    print("\nComparing 5 models with GridSearchCV (cv=5)...")
    results = run_all_models(X_train, y_train, X_test, y_test, cv=5)

    # 6. Select best
    best_name, best_model, reason = select_best_model(results)
    print(f"\n{'=' * 60}")
    print("BEST MODEL SELECTED:")
    print(f"  {reason}")
    print("=" * 60)

    # 7. Save
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    joblib.dump(extractor, VECTORIZER_PATH, compress=3)
    print(f"\nSaved model      -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")

    # 8. Summary table
    print("\n--- Model Comparison Summary ---")
    print(f"{'Model':<22} {'CV F1':>8} {'Test F1':>9} {'Recall(mal)':>12} {'Infer(ms)':>10}")
    print("-" * 65)
    for name, r in results.items():
        marker = " <-- SELECTED" if name == best_name else ""
        print(
            f"{name:<22} {r['cv_f1']:>8.4f} {r['test_f1']:>9.4f} "
            f"{r['malicious_recall']:>12.4f} {r['inference_time_ms']:>10.1f}{marker}"
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
