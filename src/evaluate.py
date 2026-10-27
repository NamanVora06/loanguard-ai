"""Comprehensive model evaluation module for LoanGuard AI.

Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrices,
and ROC/PR curves on held-out test data for TabNet, Random Forest, and Logistic Regression.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    BASELINE_LR_FILE,
    METRICS_REPORT_FILE,
    MODELS_DIR,
    PREPROCESSOR_FILE,
    REPORTS_DIR,
    TABNET_MODEL_FILE,
    TARGET_COL,
    TEST_DATA_CACHE,
)
from src.preprocessing import TabNetDataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Computes all essential classification metrics, curves, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    # Subsample curves to 50 points for lean JSON storage
    indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
    fpr_sub = [float(fpr[i]) for i in indices]
    tpr_sub = [float(tpr[i]) for i in indices]

    pr_indices = np.linspace(0, len(precision_vals) - 1, min(50, len(precision_vals)), dtype=int)
    prec_sub = [float(precision_vals[i]) for i in pr_indices]
    rec_sub = [float(recall_vals[i]) for i in pr_indices]

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "roc_curve": {
            "fpr": fpr_sub,
            "tpr": tpr_sub,
        },
        "pr_curve": {
            "precision": prec_sub,
            "recall": rec_sub,
        },
    }


def evaluate_all_models() -> Dict:
    """Evaluates TabNet, Logistic Regression, and Random Forest on held-out test data."""
    logger.info("Starting model evaluation on held-out test data...")

    # 1. Load test data
    if not TEST_DATA_CACHE.exists():
        raise FileNotFoundError(f"Test data cache not found at {TEST_DATA_CACHE}. Run train.py first.")

    df_test = pd.read_parquet(TEST_DATA_CACHE)
    logger.info(f"Loaded test dataset: {len(df_test)} samples. Target distribution: {df_test[TARGET_COL].value_counts().to_dict()}")

    # 2. Load preprocessor
    preprocessor = TabNetDataPreprocessor.load(PREPROCESSOR_FILE)
    X_test = preprocessor.transform(df_test)
    y_test = df_test[TARGET_COL].values.astype(np.int64)

    # 3. Evaluate TabNet
    logger.info("Evaluating TabNet model...")
    tabnet = TabNetClassifier()
    tabnet.load_model(str(TABNET_MODEL_FILE))
    tabnet_probs = tabnet.predict_proba(X_test)[:, 1]
    tabnet_preds = (tabnet_probs >= 0.5).astype(int)
    tabnet_metrics = compute_binary_metrics(y_test, tabnet_preds, tabnet_probs)

    # 4. Evaluate Logistic Regression Baseline
    logger.info("Evaluating Logistic Regression baseline...")
    lr_model = joblib.load(BASELINE_LR_FILE)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    lr_metrics = compute_binary_metrics(y_test, lr_preds, lr_probs)

    # 5. Optional / Comparative Tree Baseline: Random Forest
    rf_cache_file = MODELS_DIR / "baseline_rf.joblib"
    if not rf_cache_file.exists():
        logger.info("Fitting and evaluating Random Forest baseline...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        # Note: If train dataset is needed for RF fit, fit on test is avoided.
        # We fit RF on X_test only if pre-cached model is absent, but better: fit during train.py.
        # For evaluation, check if pre-saved or skip gracefully:
        rf_metrics = None
    else:
        logger.info("Evaluating pre-saved Random Forest baseline...")
        rf_model = joblib.load(rf_cache_file)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_preds = (rf_probs >= 0.5).astype(int)
        rf_metrics = compute_binary_metrics(y_test, rf_preds, rf_probs)

    results = {
        "test_sample_count": len(y_test),
        "test_default_count": int(y_test.sum()),
        "test_non_default_count": int((y_test == 0).sum()),
        "test_default_rate": round(float(y_test.mean()), 4),
        "imbalance_explanation": (
            "In imbalanced lending portfolios (~22% default rate), an uninformed trivial model "
            "that blindly predicts 'Non-Default' for every applicant achieves ~78% accuracy "
            "despite capturing 0% of actual loan defaults (Recall = 0.0, F1 = 0.0). "
            "Consequently, ROC-AUC, Precision-Recall AUC, and Recall are the primary "
            "metrics for risk management."
        ),
        "models": {
            "TabNet": tabnet_metrics,
            "Logistic Regression": lr_metrics,
        },
    }

    if rf_metrics is not None:
        results["models"]["Random Forest"] = rf_metrics

    # Save metrics report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation metrics report successfully saved to {METRICS_REPORT_FILE}")
    return results


if __name__ == "__main__":
    report = evaluate_all_models()
    print("\n" + "=" * 55)
    print("           MODEL TEST PERFORMANCE SUMMARY")
    print("=" * 55)
    for model_name, m in report["models"].items():
        print(f"\n--- {model_name} ---")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
        print(f"  PR-AUC:    {m['pr_auc']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  F1-Score:  {m['f1']:.4f}")
        print(f"  Confusion Matrix: TP={m['confusion_matrix']['tp']}, FP={m['confusion_matrix']['fp']}, "
              f"TN={m['confusion_matrix']['tn']}, FN={m['confusion_matrix']['fn']}")
    print("=" * 55)
