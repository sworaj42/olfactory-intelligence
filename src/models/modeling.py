from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_random_forest(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
):
    """
    Train a Random Forest classifier and return the fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a classification model using a probability threshold.
    Returns predictions, probabilities, metrics, report, and confusion matrix.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
    }


def find_best_threshold_by_f1(model, X_valid, y_valid, thresholds=None):
    """
    Search for the threshold that gives the best F1 score.
    Returns the best threshold and a dataframe of all tested thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    y_proba = model.predict_proba(X_valid)[:, 1]
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": accuracy_score(y_valid, y_pred),
                "precision": precision_score(y_valid, y_pred, zero_division=0),
                "recall": recall_score(y_valid, y_pred, zero_division=0),
                "f1": f1_score(y_valid, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "precision"],
        ascending=False,
    ).reset_index(drop=True)

    best_threshold = float(results_df.loc[0, "threshold"])
    return best_threshold, results_df


def save_model(model, save_path):
    """
    Save a fitted model to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)


def load_model(model_path):
    """
    Load a saved model from disk.
    """
    return joblib.load(model_path)


def get_feature_importance_df(model, feature_names):
    """
    Return feature importances as a sorted dataframe.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df