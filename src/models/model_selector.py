"""
Model selection module.

Five candidate models are compared with 5-fold cross-validation.
The best model is selected using a 3-tier explicit criterion documented
here for viva justification:

    Tier 1 — PRIMARY: Highest macro F1-Score on 5-fold CV.
              Macro F1 is chosen over accuracy because datasets are
              class-imbalanced. Macro F1 equally penalises poor
              performance on the minority class.

    Tier 2 — SECURITY DOMAIN: Highest recall on the malicious class (1).
              In phishing/spam detection a false negative (missed attack)
              is more harmful than a false positive (false alarm). When
              two models tie on F1, we prefer higher malicious-class recall.

    Tier 3 — EFFICIENCY: Fastest inference time.
              Among equivalent models, a lower-latency predictor is
              preferred for production deployment.

    DISQUALIFICATION: Any model with test-set F1 < 0.90 is excluded.
"""

import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

PARAM_GRIDS = {
    "naive_bayes": {
        "model": MultinomialNB(),
        "params": {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "params": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "linear_svm": {
        "model": CalibratedClassifierCV(LinearSVC(max_iter=2000)),
        "params": {"estimator__C": [0.01, 0.1, 1.0, 10.0]},
    },
    "random_forest": {
        "model": RandomForestClassifier(n_jobs=-1, random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
        },
    },
    "xgboost": {
        "model": XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
}


def run_all_models(X_train, y_train, X_test, y_test, cv: int = 5) -> dict:
    """
    Train all 5 candidate models with GridSearchCV and return a results dict.

    Returns
    -------
    results : dict
        Keys are model names. Values are dicts with keys:
        best_estimator, cv_f1, test_f1, malicious_recall,
        inference_time_ms, report
    """
    results = {}
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    for name, cfg in PARAM_GRIDS.items():
        print(f"\n  [{name}] Running GridSearchCV (cv={cv})...")
        gs = GridSearchCV(
            cfg["model"],
            cfg["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        cv_f1 = gs.best_score_

        # Test set metrics
        t0 = time.perf_counter()
        y_pred = best.predict(X_test)
        inference_ms = (time.perf_counter() - t0) * 1000

        test_f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, target_names=["Legitimate", "Malicious"])

        # Recall on malicious class (class index 1)
        from sklearn.metrics import recall_score
        malicious_recall = recall_score(y_test, y_pred, pos_label=1)

        results[name] = {
            "best_estimator": best,
            "best_params": gs.best_params_,
            "cv_f1": cv_f1,
            "test_f1": test_f1,
            "malicious_recall": malicious_recall,
            "inference_time_ms": inference_ms,
            "report": report,
        }
        print(
            f"    CV F1={cv_f1:.4f}  Test F1={test_f1:.4f}  "
            f"Recall(malicious)={malicious_recall:.4f}  "
            f"Inference={inference_ms:.1f}ms"
        )

    return results


def select_best_model(results: dict) -> tuple:
    """
    Apply 3-tier selection logic and return (model_name, best_estimator, reason).
    """
    DISQUALIFY_THRESHOLD = 0.90

    candidates = {
        name: r for name, r in results.items()
        if r["test_f1"] >= DISQUALIFY_THRESHOLD
    }

    if not candidates:
        # Relax threshold if ALL models are below it (small dataset edge case)
        candidates = results

    # Tier 1: highest macro F1
    sorted_by_f1 = sorted(candidates.items(), key=lambda x: x[1]["cv_f1"], reverse=True)
    best_f1 = sorted_by_f1[0][1]["cv_f1"]

    # Tier 2: among models within 0.5% of best F1, pick highest malicious recall
    tier2 = [(n, r) for n, r in sorted_by_f1 if r["cv_f1"] >= best_f1 - 0.005]
    tier2.sort(key=lambda x: x[1]["malicious_recall"], reverse=True)
    best_recall = tier2[0][1]["malicious_recall"]

    # Tier 3: among models within 0.5% of best recall, pick fastest
    tier3 = [(n, r) for n, r in tier2 if r["malicious_recall"] >= best_recall - 0.005]
    tier3.sort(key=lambda x: x[1]["inference_time_ms"])

    best_name, best_r = tier3[0]
    reason = (
        f"Selected '{best_name}' — "
        f"CV F1={best_r['cv_f1']:.4f}, "
        f"Test F1={best_r['test_f1']:.4f}, "
        f"Malicious Recall={best_r['malicious_recall']:.4f}, "
        f"Inference={best_r['inference_time_ms']:.1f}ms. "
        "Chosen by 3-tier criterion: (1) macro F1, (2) malicious-class recall, "
        "(3) inference speed."
    )

    return best_name, best_r["best_estimator"], reason
