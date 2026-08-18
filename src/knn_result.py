import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================
# 1. TIỀN XỬ LÝ: CHUẨN HÓA DỮ LIỆU THUẦN TÚY (STANDARD SCALER)
# =====================================================================
class StandardScaler:
    """Chuẩn hóa dữ liệu về phân phối chuẩn (mean=0, std=1) từ đầu bằng NumPy."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X_arr = np.array(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        # Tránh chia cho 0 nếu std = 0
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =====================================================================
# 2. THUẬT TOÁN KNN CLASSIFIER THUẦN TÚY (ROBUST KNN FROM SCRATCH)
# =====================================================================
class RobustKNNClassifier:
    def __init__(self, n_neighbors=7, metric='euclidean', p=2, weights='distance'):
        """
        KNN Classifier thuần túy tối ưu hóa ma trận khoảng cách Vectorized.
        
        Parameters:
        -----------
        - n_neighbors: int, Số lượng láng giềng k
        - metric     : str, 'euclidean', 'manhattan', 'minkowski'
        - p          : float, Bậc Minkowski
        - weights    : str, 'uniform' hoặc 'distance'
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """Lưu trữ dữ liệu huấn luyện."""
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, X):
        """Tính toán ma trận khoảng cách giữa X (m mẫu) và X_train (n mẫu)."""
        X_arr = np.array(X, dtype=np.float64)
        if self.metric == 'euclidean':
            # Tối ưu vector hóa: ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A . B_T)
            dists_sq = np.sum(X_arr**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X_arr, self.X_train.T)
            return np.sqrt(np.maximum(dists_sq, 0.0))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X_arr[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.metric == 'minkowski':
            diff = np.abs(X_arr[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
        else:
            raise ValueError(f"Không hỗ trợ metric '{self.metric}'")

    def predict_proba(self, X):
        """Dự đoán phân phối xác suất cho từng mẫu dữ liệu."""
        distances = self._compute_distances(X)
        k = min(self.n_neighbors, self.X_train.shape[0])
        knn_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        
        probabilities = []
        eps = 1e-10
        
        for i in range(distances.shape[0]):
            k_idx = knn_indices[i]
            k_dists = distances[i, k_idx]
            k_labels = self.y_train[k_idx]
            
            if self.weights == 'distance':
                weights_arr = 1.0 / (k_dists + eps)
            else:
                weights_arr = np.ones_like(k_dists)
                
            class_probs = []
            for c in self.classes_:
                weight_c = np.sum(weights_arr[k_labels == c])
                class_probs.append(weight_c)
                
            total_w = np.sum(weights_arr)
            class_probs = np.array(class_probs) / (total_w if total_w > 0 else 1.0)
            probabilities.append(class_probs)
            
        return np.array(probabilities)

    def predict(self, X):
        """Dự đoán nhãn bằng cách chọn lớp có xác suất cao nhất."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Tính Accuracy trực tiếp."""
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))


# =====================================================================
# 3. PIPELINE THUẦN TÚY (SCALER + CLASSIFIER)
# =====================================================================
class CustomPipeline:
    """Pipeline thuần túy ghép nối StandardScaler và RobustKNNClassifier."""
    def __init__(self, scaler, classifier):
        self.scaler = scaler
        self.classifier = classifier

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def score(self, X, y):
        X_scaled = self.scaler.transform(X)
        return self.classifier.score(X_scaled, y)


# =====================================================================
# 4. HÀM ĐÁNH GIÁ & CHIA TẬP DỮ LIỆU THUẦN TÚY
# =====================================================================
def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """Phân chia Train/Test bảo toàn tỷ lệ nhãn (Stratified Split) thuần túy."""
    if random_state is not None:
        np.random.seed(random_state)
        
    X_arr = np.array(X)
    y_arr = np.array(y)
    classes = np.unique(y_arr)
    
    train_indices = []
    test_indices = []
    
    for c in classes:
        cls_idx = np.where(y_arr == c)[0]
        np.random.shuffle(cls_idx)
        n_test = int(np.round(len(cls_idx) * test_size))
        
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])
        
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    if isinstance(X, pd.DataFrame):
        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
    return X_arr[train_indices], X_arr[test_indices], y_arr[train_indices], y_arr[test_indices]


def evaluate_multiclass_metrics(y_true, y_pred, classes=None):
    """Tính toán chi tiết Accuracy, Precision, Recall, F1 và Confusion Matrix."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    if classes is None:
        classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true_arr, y_pred_arr):
        i = np.where(classes == t)[0][0]
        j = np.where(classes == p)[0][0]
        cm[i, j] += 1
        
    total_samples = len(y_true_arr)
    accuracy = float(np.sum(np.diag(cm)) / total_samples) if total_samples > 0 else 0.0
    
    # Weighted precision, recall, f1
    precisions = []
    recalls = []
    f1s = []
    weights = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        weights.append(support)
        
    weighted_p = float(np.average(precisions, weights=weights)) if sum(weights) > 0 else 0.0
    weighted_r = float(np.average(recalls, weights=weights)) if sum(weights) > 0 else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if sum(weights) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": weighted_p,
        "recall": weighted_r,
        "f1_score": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "classes": [int(c) for c in classes]
    }


def compute_permutation_importance(pipeline, X_val, y_val, n_repeats=3, random_state=42):
    """Tính Permutation Feature Importance thuần túy."""
    if random_state is not None:
        np.random.seed(random_state)
        
    baseline_acc = pipeline.score(X_val, y_val)
    X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val.copy()
    feature_names = list(X_val_df.columns)
    importances = {}
    
    for col in feature_names:
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_val_df.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
            shuffled_acc = pipeline.score(X_shuffled, y_val)
            drops.append(baseline_acc - shuffled_acc)
        importances[str(col)] = float(np.mean(drops))
        
    return importances


# =====================================================================
# 5. HUẤN LUYỆN VÀ DỰ ĐOÁN CHO DỰ ÁN VÁN CỜ LICHESS
# =====================================================================
def train_knn_classifier(X, y, random_state=42, test_size=0.2,
                         k=7, weights='distance',
                         model_save_path="models/knn_result.joblib",
                         metrics_save_path="outputs/knn_result_metrics.json"):
    """
    Huấn luyện KNN thuần túy dự đoán Result ván cờ (0=Black, 1=Draw, 2=White).
    """
    print("\n" + "=" * 60)
    print("   TRAINING KNN RESULT CLASSIFIER (FROM SCRATCH - NO SKLEARN)")
    print("=" * 60)

    # 1. Chia Train/Test Stratified
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[*] Train set size: {len(X_train):,}, Test set size: {len(X_test):,}")

    # 2. Xây dựng Pipeline thuần túy
    scaler = StandardScaler()
    knn = RobustKNNClassifier(n_neighbors=k, metric='euclidean', weights=weights)
    pipeline = CustomPipeline(scaler=scaler, classifier=knn)

    print(f"[*] Đang huấn luyện KNN Classifier thuần túy (k={k}, weights='{weights}')...")
    pipeline.fit(X_train, y_train)

    # 3. Đánh giá trên tập Test
    y_pred = pipeline.predict(X_test)
    eval_res = evaluate_multiclass_metrics(y_test, y_pred, classes=knn.classes_)

    # 4. Feature Importance thuần túy
    print("[*] Đang tính Permutation Feature Importance thuần túy...")
    feat_imp = compute_permutation_importance(pipeline, X_test, y_test, n_repeats=3, random_state=random_state)

    metrics = {
        "model_name": "K-Nearest Neighbors Result Classifier (From Scratch)",
        "params": {"n_neighbors": k, "weights": weights, "metric": "euclidean"},
        "accuracy": eval_res["accuracy"],
        "precision": eval_res["precision"],
        "recall": eval_res["recall"],
        "f1_score": eval_res["f1_score"],
        "confusion_matrix": eval_res["confusion_matrix"],
        "feature_importance": feat_imp,
        "features_used": list(X.columns) if hasattr(X, "columns") else [f"feat_{i}" for i in range(X.shape[1])],
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    print("\n--- KNN Classifier Test Results ---")
    print(f"  Accuracy : {eval_res['accuracy'] * 100:.2f}%")
    print(f"  Precision: {eval_res['precision'] * 100:.2f}% (weighted)")
    print(f"  Recall   : {eval_res['recall'] * 100:.2f}% (weighted)")
    print(f"  F1-Score : {eval_res['f1_score'] * 100:.2f}% (weighted)")
    print(f"\nFeature Importances:\n  {feat_imp}")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted [0:Black, 1:Draw, 2:White]):")
    print(np.array(eval_res["confusion_matrix"]))

    # 5. Lưu Model và Metrics
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"[+] Đã lưu KNN model thuần túy vào '{model_save_path}'.")

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Đã lưu metrics vào '{metrics_save_path}'.")

    print("=" * 60 + "\n")
    return pipeline, metrics, (X_train, X_test, y_train, y_test, y_pred)


def predict_result_knn(white_rating, black_rating, rated=1, opening_ply=8, model_path="models/knn_result.joblib"):
    """
    Dự đoán kết quả ván cờ bằng mô hình KNN thuần túy đã huấn luyện.
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
    probs = pipeline.predict_proba(X_input)[0]

    labels = {0: "Black thắng (0-1)", 1: "Hòa (1/2-1/2)", 2: "White thắng (1-0)"}

    # Đảm bảo 3 xác suất
    p0 = float(probs[0] * 100) if len(probs) > 0 else 33.3
    p1 = float(probs[1] * 100) if len(probs) > 1 else 33.3
    p2 = float(probs[2] * 100) if len(probs) > 2 else 33.4

    return {
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": rating_diff,
        "rated": rated,
        "opening_ply": opening_ply,
        "predicted_class": int(pred_class),
        "predicted_label": labels.get(int(pred_class), f"Class {pred_class}"),
        "probabilities": {
            "Black thắng (0-1)": p0,
            "Hòa (1/2-1/2)": p1,
            "White thắng (1-0)": p2
        }
    }


if __name__ == "__main__":
    from data_loader import prepare_and_cache_dataset
    from preprocessing import preprocess_data

    df = prepare_and_cache_dataset(max_games=5000)
    _, X, y, _ = preprocess_data(df)
    train_knn_classifier(X, y)
