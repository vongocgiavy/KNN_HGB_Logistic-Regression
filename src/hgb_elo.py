import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure UTF-8 stdout encoding for Windows terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_hgb_classifier(X, y, random_state=42, test_size=0.2,
                         model_save_path="models/hgb_elo.joblib",
                         metrics_save_path="outputs/hgb_metrics.json"):
    """
    Trains HistGradientBoostingClassifier (HGB) to predict game Result based on player features.

    Features: ['white_rating', 'black_rating', 'rating_diff', 'rated', 'opening_ply']
    Target: Result (0 = Black win, 1 = Draw, 2 = White win)
    Includes 5-fold StratifiedKFold Cross-Validation, GridSearchCV, and Permutation Feature Importance.
    """
    print("\n" + "=" * 60)
    print("      TRAINING HIST GRADIENT BOOSTING (HGB MODEL)")
    print("=" * 60)

    # 80/20 Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[*] Train set size: {len(X_train):,}, Test set size: {len(X_test):,}")

    base_hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=random_state
    )

    # 5-fold Stratified CV + GridSearchCV
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_iter": [100, 200],
        "max_depth": [4, 6],
        "min_samples_leaf": [20, 50]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(base_hgb, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)

    print("[*] Running 5-fold Stratified CV & GridSearchCV for HGB...")
    grid_search.fit(X_train, y_train)

    best_hgb = grid_search.best_estimator_
    cv_mean = float(grid_search.best_score_)
    cv_std = float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])
    print(f"[+] Best Params: {grid_search.best_params_}")
    print(f"[+] 5-Fold CV F1-Score (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}")

    # Evaluate on Test set
    y_pred = best_hgb.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    class_names = ["Black Win (0)", "Draw (1)", "White Win (2)"]
    report_str = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    # Permutation Feature Importance
    print("[*] Computing Permutation Feature Importance for HGB...")
    perm_imp = permutation_importance(best_hgb, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1)
    feature_names = list(X.columns)
    feat_imp = {feat: float(mean_score) for feat, mean_score in zip(feature_names, perm_imp.importances_mean)}

    metrics = {
        "model_name": "HistGradientBoostingClassifier",
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

    print("\n--- HistGradientBoosting Classifier Test Results ---")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall   : {recall:.4f} (weighted)")
    print(f"  F1-Score : {f1:.4f} (weighted)")
    print(f"\nFeature Importances (Permutation Importance):\n  {feat_imp}")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted [0:Black, 1:Draw, 2:White]):")
    print(np.array(cm))
    print("\nClassification Report:")
    print(report_str)

    # Save trained model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    artifacts = {
        "model": best_hgb,
        "feature_names": feature_names,
        "target_classes": [0, 1, 2],
        "class_labels": {0: "Black thắng", 1: "Hòa", 2: "White thắng"}
    }
    joblib.dump(artifacts, model_save_path)
    print(f"[+] Saved HGB model artifact to '{model_save_path}'.")

    # Save metrics JSON
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Saved metrics to '{metrics_save_path}'.")

    print("=" * 60 + "\n")

    return best_hgb, metrics, report_str, (X_train, X_test, y_train, y_test, y_pred)


def predict_game_result(white_rating, black_rating, rated=1, opening_ply=8, model_path="models/hgb_elo.joblib"):
    """
    Predicts game outcome (Result) given player ratings, rated status, and opening ply using 5 features.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train HGB classifier first.")

    model = joblib.load(model_path)
    rating_diff = white_rating - black_rating
    X_input = pd.DataFrame([{
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": rating_diff,
        "rated": rated,
        "opening_ply": opening_ply
    }])

    pred_class = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]

    labels = {0: "Black thắng (0-1)", 1: "Hòa (1/2-1/2)", 2: "White thắng (1-0)"}

    prob_dict = {
        "Black thắng (0-1)": float(probs[0] * 100),
        "Hòa (1/2-1/2)": float(probs[1] * 100),
        "White thắng (1-0)": float(probs[2] * 100)
    }

    return {
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": rating_diff,
        "rated": rated,
        "opening_ply": opening_ply,
        "predicted_class": int(pred_class),
        "predicted_label": labels[pred_class],
        "probabilities": prob_dict
    }


if __name__ == "__main__":
    from data_loader import prepare_and_cache_dataset
    from preprocessing import preprocess_data

    df = prepare_and_cache_dataset(max_games=10000)
    _, X, y, _ = preprocess_data(df)
    train_hgb_classifier(X, y)

    # Test interactive prediction
    res = predict_game_result(1800, 1500)
    print("\nTest Prediction for WhiteElo=1800, BlackElo=1500:")
    print(f"EloDiff: {res['elo_diff']}")
    print(f"Prediction: {res['predicted_label']}")
    for k, v in res['probabilities'].items():
        print(f"  P({k}) = {v:.2f}%")
