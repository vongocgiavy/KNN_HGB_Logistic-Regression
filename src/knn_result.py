import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_knn_classifier(X, y, random_state=42, test_size=0.2,
                         model_save_path="models/knn_result.joblib",
                         metrics_save_path="outputs/knn_result_metrics.json"):
    """
    Trains KNeighborsClassifier (KNN) to predict game Result (0=Black, 1=Draw, 2=White) from tabular features.

    Features: ['white_rating', 'black_rating', 'rating_diff', 'rated', 'opening_ply']
    Includes 5-fold StratifiedKFold Cross-Validation, GridSearchCV, and Permutation Feature Importance.
    """
    print("\n" + "=" * 60)
    print("      TRAINING KNN CLASSIFIER (RESULT PREDICTION)")
    print("=" * 60)

    # 80/20 Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[*] Train set size: {len(X_train):,}, Test set size: {len(X_test):,}")

    # Build Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier())
    ])

    # 5-fold Stratified CV + GridSearchCV
    param_grid = {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)

    print("[*] Running 5-fold Stratified CV & GridSearchCV for KNN Classifier...")
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

    # Permutation Feature Importance
    print("[*] Computing Permutation Feature Importance...")
    perm_imp = permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1)
    feature_names = list(X.columns)
    feat_imp = {feat: float(mean_score) for feat, mean_score in zip(feature_names, perm_imp.importances_mean)}

    metrics = {
        "model_name": "K-Nearest Neighbors Classifier",
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

    print("\n--- KNN Classifier Test Results ---")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall   : {recall:.4f} (weighted)")
    print(f"  F1-Score : {f1:.4f} (weighted)")
    print(f"\nFeature Importances (Permutation Importance):\n  {feat_imp}")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted [0:Black, 1:Draw, 2:White]):")
    print(np.array(cm))

    class_names = ["Black Win (0)", "Draw (1)", "White Win (2)"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Save trained pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_pipeline, model_save_path)
    print(f"[+] Saved KNN Classifier model to '{model_save_path}'.")

    # Save metrics JSON
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Saved metrics to '{metrics_save_path}'.")

    print("=" * 60 + "\n")

    return best_pipeline, metrics, (X_train, X_test, y_train, y_test, y_pred)


def predict_result_knn(white_rating, black_rating, rated=1, opening_ply=8, model_path="models/knn_result.joblib"):
    """
    Predicts game result using trained KNN Classifier model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"KNN Classifier model not found at '{model_path}'. Please train it first.")

    pipeline = joblib.load(model_path)

    rating_diff = white_rating - black_rating
    X_input = pd.DataFrame([{
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": rating_diff,
        "rated": rated,
        "opening_ply": opening_ply
    }])

    pred_class = pipeline.predict(X_input)[0]
    probs = pipeline.predict_proba(X_input)[0] if hasattr(pipeline, "predict_proba") else [0.33, 0.33, 0.34]

    labels = {0: "Black thắng (0-1)", 1: "Hòa (1/2-1/2)", 2: "White thắng (1-0)"}

    return {
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": rating_diff,
        "rated": rated,
        "opening_ply": opening_ply,
        "predicted_class": int(pred_class),
        "predicted_label": labels[pred_class],
        "probabilities": {
            "Black thắng (0-1)": float(probs[0] * 100),
            "Hòa (1/2-1/2)": float(probs[1] * 100),
            "White thắng (1-0)": float(probs[2] * 100)
        }
    }


if __name__ == "__main__":
    from data_loader import prepare_and_cache_dataset
    from preprocessing import preprocess_data

    df = prepare_and_cache_dataset(max_games=10000)
    _, X, y, _ = preprocess_data(df)
    train_knn_classifier(X, y)
