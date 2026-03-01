"""
Training script — preprocesses data, trains both models, evaluates,
and saves the best model + scaler as .pkl files.
"""

import os
import sys
import joblib
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_pipeline
from src.models import (
    train_logistic_regression,
    train_decision_tree,
    tune_decision_tree,
    evaluate_model,
    cross_validate_model,
)


def main():
    # ── Paths ──────────────────────────────────────────────
    data_path = os.path.join(PROJECT_ROOT, "data", "EV_Predictive_Maintenance_Dataset.csv")
    model_dir = PROJECT_ROOT
    os.makedirs(model_dir, exist_ok=True)

    # ── Preprocess ─────────────────────────────────────────
    print("=" * 60)
    print("1. PREPROCESSING")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(data_path)
    print(f"   Train set size : {X_train.shape[0]}")
    print(f"   Test  set size : {X_test.shape[0]}")
    print(f"   Features       : {X_train.shape[1]}")
    print(f"   Feature names  : {feature_names}")

    # ── Logistic Regression ────────────────────────────────
    print("\n" + "=" * 60)
    print("2. LOGISTIC REGRESSION")
    print("=" * 60)
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print(f"   Accuracy  : {lr_metrics['accuracy']:.4f}")
    print(f"   Precision : {lr_metrics['precision']:.4f}")
    print(f"   Recall    : {lr_metrics['recall']:.4f}")
    print(f"   F1-score  : {lr_metrics['f1']:.4f}")
    print(f"   ROC AUC   : {lr_metrics['roc_auc']:.4f}")
    print(f"\n   Confusion Matrix:\n{lr_metrics['confusion_matrix']}")
    print(f"\n{lr_metrics['classification_report']}")

    # Cross-validation
    lr_cv = cross_validate_model(lr_model, X_train, y_train, cv=5)
    print(f"   CV F1 scores : {lr_cv}")
    print(f"   CV F1 mean   : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    # ── Decision Tree ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. DECISION TREE (with hyperparameter tuning)")
    print("=" * 60)
    dt_model, best_params = tune_decision_tree(X_train, y_train, cv=5)
    print(f"   Best params : {best_params}")
    dt_metrics = evaluate_model(dt_model, X_test, y_test)
    print(f"   Accuracy  : {dt_metrics['accuracy']:.4f}")
    print(f"   Precision : {dt_metrics['precision']:.4f}")
    print(f"   Recall    : {dt_metrics['recall']:.4f}")
    print(f"   F1-score  : {dt_metrics['f1']:.4f}")
    print(f"   ROC AUC   : {dt_metrics['roc_auc']:.4f}")
    print(f"\n   Confusion Matrix:\n{dt_metrics['confusion_matrix']}")
    print(f"\n{dt_metrics['classification_report']}")

    # Cross-validation
    dt_cv = cross_validate_model(dt_model, X_train, y_train, cv=5)
    print(f"   CV F1 scores : {dt_cv}")
    print(f"   CV F1 mean   : {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")

    # ── Feature Importance (Decision Tree) ─────────────────
    print("\n" + "=" * 60)
    print("4. FEATURE IMPORTANCE (Decision Tree)")
    print("=" * 60)
    importances = dt_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i, idx in enumerate(sorted_idx[:15]):
        print(f"   {i+1:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")

    # ── Comparison ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Logistic Reg':>14} {'Decision Tree':>14}")
    print("-" * 45)
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{m:<15} {lr_metrics[m]:>14.4f} {dt_metrics[m]:>14.4f}")

    # ── Save artefacts ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. SAVING ARTEFACTS")
    print("=" * 60)

    # Choose the model with higher F1 as the primary model
    if dt_metrics["f1"] >= lr_metrics["f1"]:
        best_model = dt_model
        best_name = "Decision Tree"
    else:
        best_model = lr_model
        best_name = "Logistic Regression"

    artefacts = {
        "best_model": best_model,
        "best_model_name": best_name,
        "lr_model": lr_model,
        "dt_model": dt_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "lr_metrics": {k: v for k, v in lr_metrics.items() if k not in ("fpr", "tpr", "y_pred", "y_proba")},
        "dt_metrics": {k: v for k, v in dt_metrics.items() if k not in ("fpr", "tpr", "y_pred", "y_proba")},
    }

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(artefacts, model_path)
    print(f"   Saved → {model_path}")
    print(f"   Best model: {best_name}")
    print("\nDone ✅")


if __name__ == "__main__":
    main()
