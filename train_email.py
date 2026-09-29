"""
Email spam classifier training pipeline.

Steps:
  1. Load SMS Spam Collection dataset
  2. Preprocess text (NLP cleaning + stemming)
  3. Extract features (TF-IDF bigrams + 8 metadata features)
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
from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.features.email_features import EmailFeatureExtractor
from src.models.model_selector import run_all_models, select_best_model

DATA_PATH = os.path.join("data", "raw", "sms_spam.csv")
MODEL_PATH = os.path.join("saved_models", "email_model.pkl")
VECTORIZER_PATH = os.path.join("saved_models", "email_vectorizer.pkl")


def main():
    print("=" * 60)
    print("EMAIL SPAM CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run: python data/download_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label", "text"])
    df["label_enc"] = (df["label"].str.lower().str.strip() == "spam").astype(int)

    print(f"\nDataset: {len(df)} messages")
    print(f"  Spam: {df['label_enc'].sum()} ({df['label_enc'].mean():.1%})")
    print(f"  Ham:  {(1 - df['label_enc']).sum()}")

    # 2. Train/test split (80/20, stratified)
    X_text = df["text"].tolist()
    y = df["label_enc"].tolist()
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Preprocess
    print("\nPreprocessing text...")
    preprocessor = EmailPreprocessor()
    X_tr_clean = preprocessor.transform(X_tr_raw)
    X_te_clean = preprocessor.transform(X_te_raw)

    # 4. Feature extraction
    print("Extracting features (TF-IDF + metadata)...")
    extractor = EmailFeatureExtractor(max_features=10_000)
    X_train = extractor.fit_transform(X_tr_clean, X_tr_raw)
    X_test = extractor.transform(X_te_clean, X_te_raw)
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

    # 8. Print summary table for report
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
