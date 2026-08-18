import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_logistic_regression(X, y, random_state=42, test_size=0.2,
                              model_save_path="models/logistic_baseline.joblib",
                              metrics_save_path="outputs/logistic_metrics.json"):
    """
    Trains Logistic Regression as a BASELINE model to predict game Result from player features.

    Features: ['white_rating', 'black_rating', 'rating_diff', 'rated', 'opening_ply']
    Target: Result (0 = Black win, 1 = Draw, 2 = White win)
    Includes 5-fold StratifiedKFold Cross-Validation, GridSearchCV, and Feature Importance.
    """
    print("\n" + "=" * 60)
    print("   TRAINING LOGISTIC REGRESSION (BASELINE WITH 5-FOLD CV)")
    print("=" * 60)

    # 80/20 Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[*] Train set size: {len(X_train):,}, Test set size: {len(X_test):,}")

    # Build Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))
    ])

    # 5-fold Stratified CV + GridSearchCV
    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "saga"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)

    print("[*] Running 5-fold Stratified CV & GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    cv_mean = float(grid_search.best_score_)
    cv_std = float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])
    print(f"[+] Best Params: {grid_search.best_params_}")
    print(f"[+] 5-Fold CV F1-Score (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}")

    # Evaluate on Test set
    y_pred = best_pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature Importance (average absolute coefficient magnitude across classes)
    clf = best_pipeline.named_steps["classifier"]
    feature_names = list(X.columns)
    avg_coefs = np.mean(np.abs(clf.coef_), axis=0)
    feat_imp = {feat: float(score) for feat, score in zip(feature_names, avg_coefs)}

    metrics = {
        "model_name": "Logistic Regression (Baseline)",
        "best_params": grid_search.best_params_,
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "feature_importance": feat_imp,
        "features_used": feature_names,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    print("\n--- Logistic Regression Baseline Test Results ---")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall   : {recall:.4f} (weighted)")
    print(f"  F1-Score : {f1:.4f} (weighted)")
    print(f"\nFeature Importances (Coef Magnitude):\n  {feat_imp}")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted [0:Black, 1:Draw, 2:White]):")
    print(np.array(cm))

    class_names = ["Black Win (0)", "Draw (1)", "White Win (2)"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Save trained pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_pipeline, model_save_path)
    print(f"[+] Saved Logistic Regression model to '{model_save_path}'.")

    # Save metrics JSON
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Saved metrics to '{metrics_save_path}'.")

    print("=" * 60 + "\n")

    return best_pipeline, metrics, (X_train, X_test, y_train, y_test, y_pred)


if __name__ == "__main__":
    from data_loader import prepare_and_cache_dataset
    from preprocessing import preprocess_data

    df = prepare_and_cache_dataset(max_games=10000)
    _, X, y, _ = preprocess_data(df)
    train_logistic_regression(X, y)
