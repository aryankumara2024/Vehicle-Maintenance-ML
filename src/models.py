"""
Model training and evaluation module.

Implements:
- Logistic Regression
- Decision Tree Classifier
- Cross-validation
- Hyperparameter tuning
- Evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)


def train_logistic_regression(X_train, y_train, max_iter=1000):
    """Train a Logistic Regression model with balanced class weights."""
    lr = LogisticRegression(
        max_iter=max_iter, random_state=42, solver="lbfgs",
        class_weight="balanced"
    )
    lr.fit(X_train, y_train)
    return lr


def train_decision_tree(X_train, y_train, max_depth=None):
    """Train a Decision Tree Classifier with balanced class weights."""
    dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth, class_weight="balanced")
    dt.fit(X_train, y_train)
    return dt


def tune_decision_tree(X_train, y_train, cv=5):
    """Hyperparameter tuning for Decision Tree using GridSearchCV."""
    param_grid = {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return metrics dictionary.

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix,
        classification_report, roc_auc, fpr, tpr
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        fpr, tpr, auc = None, None, None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "roc_auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring="f1"):
    """Perform cross-validation and return scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores
