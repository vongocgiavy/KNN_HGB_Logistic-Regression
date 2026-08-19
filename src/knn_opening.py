import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. TIỀN XỬ LÝ: CHUẨN HÓA DỮ LIỆU
# ==========================================
class StandardScaler:
    """KNN tính khoảng cách hình học nên bắt buộc các đặc trưng phải cùng thang đo."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_arr = np.array(X, dtype=np.float64)
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        self.std[self.std == 0] = 1e-8
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float64)
        return (X_arr - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==========================================
# 2. THUẬT TOÁN K-NEAREST NEIGHBORS HOÀN CHỈNH
# ==========================================
class RobustKNNClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', p=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def _compute_distances(self, X):
        X_arr = np.array(X, dtype=np.float64)
        if self.metric == 'euclidean':
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
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


def compute_multiclass_metrics(y_true, y_pred, classes=None):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_t, y_p]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_t, y_p):
        i = np.where(classes == t)[0][0]
        j = np.where(classes == p)[0][0]
        cm[i, j] += 1
    accuracy = float(np.sum(np.diag(cm)) / len(y_t)) if len(y_t) > 0 else 0.0
    return {"Accuracy": accuracy, "Confusion_Matrix": cm}


# ==========================================
# 3. HÀM TÌM KIẾM KHAI CUỘC (OPENING RETRIEVAL FROM SCRATCH)
# ==========================================
class SimpleTextVectorizer:
    """Bộ vector hóa chuỗi nước đi viết tay theo tần suất từ (Term Frequency)."""
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab_ = {}

    def fit(self, texts):
        token_counts = {}
        for text in texts:
            for token in str(text).split()[:15]:
                token_counts[token] = token_counts.get(token, 0) + 1
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocab_ = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
        return self

    def transform(self, texts):
        matrix = np.zeros((len(texts), len(self.vocab_)), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in str(text).split()[:15]:
                if token in self.vocab_:
                    matrix[i, self.vocab_[token]] += 1.0
            # Chuẩn hóa L2 norm
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix


def train_knn_opening(df=None, K_list=[3, 5, 7], model_save_path="models/knn_opening.joblib"):
    """Huấn luyện và lập chỉ mục tìm kiếm Khai cuộc bằng KNN thuần túy."""
    if df is None:
        data_path = "data/filtered_processed_games.csv"
        if not os.path.exists(data_path):
            data_path = "data/processed_games.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, nrows=5000)
        else:
            df = pd.DataFrame({
                "CleanedMoves": ["e4 e5 Nf3 Nc6", "d4 d5 c4 e6", "e4 c5 Nf3 d6", "c4 e5 Nc3 Nf6"],
                "Opening": ["Italian Game", "Queen's Gambit", "Sicilian Defense", "English Opening"],
                "ECO": ["C50", "D30", "B50", "A20"],
                "White": ["Player1", "Player2", "Player3", "Player4"],
                "Black": ["PlayerA", "PlayerB", "PlayerC", "PlayerD"]
            })

    col_moves = "CleanedMoves" if "CleanedMoves" in df.columns else "Moves"
    moves_list = df[col_moves].fillna("").astype(str).tolist()

    vectorizer = SimpleTextVectorizer(max_features=500)
    X_vec = vectorizer.fit(moves_list).transform(moves_list)

    knn = RobustKNNClassifier(n_neighbors=max(K_list), metric='euclidean')
    # Lưu chỉ số dòng thay vì nhãn đơn lẻ
    knn.fit(X_vec, np.arange(len(moves_list)))

    payload = {
        "vectorizer": vectorizer,
        "knn": knn,
        "df_metadata": df[["Opening", "ECO", "White", "Black", col_moves]].copy()
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(payload, model_save_path)
    return payload


def predict_opening(moves_input, K=5, model_or_path="models/knn_opening.joblib"):
    """Dự đoán khai cuộc dựa trên nước đi nhập vào."""
    # Khắc phục triệt để lỗi unpickle khi gọi từ các file entry point khác nhau (main.py, app.py)
    import sys
    main_mod = sys.modules.get("__main__")
    if main_mod:
        if not hasattr(main_mod, "SimpleTextVectorizer"):
            setattr(main_mod, "SimpleTextVectorizer", SimpleTextVectorizer)
        if not hasattr(main_mod, "RobustKNNClassifier"):
            setattr(main_mod, "RobustKNNClassifier", RobustKNNClassifier)

    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            train_knn_opening(model_save_path=model_or_path)
        payload = joblib.load(model_or_path)
    else:
        payload = model_or_path

    vectorizer = payload["vectorizer"]
    knn = payload["knn"]
    df_meta = payload["df_metadata"]

    cleaned_input = " ".join(str(moves_input).split())
    x_in = vectorizer.transform([cleaned_input])

    distances = knn._compute_distances(x_in)[0]
    nearest_idx = np.argsort(distances)[:K]

    nearest_games = []
    for rank, idx in enumerate(nearest_idx, start=1):
        row = df_meta.iloc[idx]
        d = float(distances[idx])
        sim_pct = max(0.0, min(100.0, (1.0 - d / 2.0) * 100.0))
        nearest_games.append({
            "rank": rank,
            "opening": str(row.get("Opening", "Unknown")),
            "eco": str(row.get("ECO", "?")),
            "white": str(row.get("White", "N/A")),
            "black": str(row.get("Black", "N/A")),
            "similarity_percent": sim_pct,
            "distance": d,
            "moves_excerpt": str(row.get("CleanedMoves", row.get("Moves", "")))[:60] + "..."
        })

    top_opening = nearest_games[0]["opening"] if nearest_games else "Unknown Opening"
    top_eco = nearest_games[0]["eco"] if nearest_games else "?"

    return {
        "predicted_opening": top_opening,
        "predicted_eco": top_eco,
        "nearest_games": nearest_games
    }


if __name__ == "__main__":
    sys.modules['knn_opening'] = sys.modules['__main__']
    train_knn_opening()
