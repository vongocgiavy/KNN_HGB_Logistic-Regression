import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. TIỀN XỬ LÝ: CHUẨN HÓA DỮ LIỆU (CỰC KỲ QUAN TRỌNG VỚI KNN)
# ==========================================
class StandardScaler:
    """KNN tính khoảng cách hình học nên bắt buộc các đặc trưng phải cùng thang đo."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==========================================
# 2. THUẬT TOÁN K-NEAREST NEIGHBORS HOÀN CHỈNH
# ==========================================
class RobustKNNClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', p=2, weights='uniform'):
        """
        KNN Classifier tối ưu hóa vector hóa (Vectorized).
        
        Parameters:
        -----------
        - n_neighbors: int, Số lượng láng giềng gần nhất k
        - metric     : str, Loại khoảng cách ('euclidean', 'manhattan', 'minkowski')
        - p          : int/float, Bậc p cho khoảng cách Minkowski (khi metric='minkowski')
        - weights    : str, 'uniform' (bầu cử ngang nhau) hoặc 'distance' (ưu tiên điểm gần)
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """Lưu trữ dữ liệu huấn luyện (Lazy Learner)."""
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def _compute_distances(self, X):
        """
        Tính ma trận khoảng cách giữa tập X (m mẫu) và X_train (n mẫu).
        Trả về ma trận kích thước (m, n).
        """
        if self.metric == 'euclidean':
            # Tối ưu hóa Vectorized: ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A.B_T)
            # Giúp tính toán cực nhanh cho toàn bộ ma trận
            dists_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X, self.X_train.T)
            return np.sqrt(np.maximum(dists_sq, 0.0))
        
        elif self.metric == 'manhattan':
            # Khoảng cách L1: sum(|x_i - y_i|)
            return np.sum(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        
        elif self.metric == 'minkowski':
            # Khoảng cách tổng quát L_p: (sum(|x_i - y_i|^p))^(1/p)
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
        
        else:
            raise ValueError(f"Không hỗ trợ metric '{self.metric}'")

    def predict_proba(self, X):
        """Dự đoán phân phối xác suất cho từng mẫu dữ liệu."""
        X = np.array(X, dtype=np.float64)
        distances = self._compute_distances(X)  # shape: (n_test, n_train)
        
        # Lấy chỉ số của k láng giềng gần nhất cho từng mẫu
        knn_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        
        probabilities = []
        eps = 1e-10 # Tránh chia cho 0 khi khoảng cách = 0
        
        for i in range(X.shape[0]):
            k_idx = knn_indices[i]
            k_dists = distances[i, k_idx]
            k_labels = self.y_train[k_idx]
            
            # Tính trọng số phiếu bầu
            if self.weights == 'distance':
                weights_arr = 1.0 / (k_dists + eps)
            else: # 'uniform'
                weights_arr = np.ones_like(k_dists)
                
            # Tính tổng trọng số cho từng lớp
            class_probs = []
            for c in self.classes_:
                weight_c = np.sum(weights_arr[k_labels == c])
                class_probs.append(weight_c)
                
            # Chuẩn hóa về xác suất tổng = 1
            class_probs = np.array(class_probs) / np.sum(weights_arr)
            probabilities.append(class_probs)
            
        return np.array(probabilities)

    def predict(self, X):
        """Dự đoán nhãn bằng cách chọn lớp có xác suất cao nhất."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Đánh giá Accuracy trên tập kiểm thử."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ==========================================
# 3. CÁC HÀM ĐÁNH GIÁ (METRICS)
# ==========================================
def compute_multiclass_metrics(y_true, y_pred, classes):
    """Tính toán ma trận nhầm lẫn (Confusion Matrix) và Accuracy tổng quát."""
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        i = np.where(classes == t)[0][0]
        j = np.where(classes == p)[0][0]
        cm[i, j] += 1
        
    accuracy = np.mean(y_true == y_pred)
    return {
        "Accuracy": accuracy,
        "Confusion_Matrix": cm
    }


# ==========================================
# 4. CHẠY THỬ NGHIỆM VÀ TRỰC QUAN HÓA
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Tạo tập dữ liệu phi tuyến tính gồm 3 lớp (Multi-class)
    print("[1] Đang tạo dữ liệu mẫu 3 lớp...")
    n_per_class = 120
    
    # Lớp 0: Tâm (-2, -1)
    X0 = np.random.randn(n_per_class, 2) * 0.7 + np.array([-2.0, -1.0])
    y0 = np.zeros(n_per_class, dtype=int)
    
    # Lớp 1: Tâm (2, -1)
    X1 = np.random.randn(n_per_class, 2) * 0.7 + np.array([2.0, -1.0])
    y1 = np.ones(n_per_class, dtype=int)
    
    # Lớp 2: Tâm (0, 2)
    X2 = np.random.randn(n_per_class, 2) * 0.7 + np.array([0.0, 2.0])
    y2 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack((X0, X1, X2))
    y = np.concatenate((y0, y1, y2))

    # Xáo trộn dữ liệu
    indices = np.random.permutation(len(y))
    X, y = X[indices], y[indices]

    # 2. Chia Train / Test (80% - 20%)
    train_size = int(0.8 * len(y))
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 4. Huấn luyện và dự đoán với KNN (k=7, Trọng số theo khoảng cách)
    print("[2] Đang chạy dự đoán với Robust KNN (k=7, weights='distance')...")
    knn = RobustKNNClassifier(n_neighbors=7, metric='euclidean', weights='distance')
    knn.fit(X_train, y_train)

    # 5. Đánh giá kết quả
    y_test_pred = knn.predict(X_test)
    res = compute_multiclass_metrics(y_test, y_test_pred, knn.classes_)

    print("\n[3] Kết quả đánh giá trên tập Test:")
    print(f" • Accuracy: {res['Accuracy'] * 100:.2f}%")
    print(f" • Confusion Matrix (3x3):\n{res['Confusion_Matrix']}")

    # 6. Trực quan hóa Ranh giới quyết định phi tuyến (Non-linear Decision Boundary)
    print("\n[4] Đang vẽ ranh giới phân loại phi tuyến của KNN...")
    plt.figure(figsize=(9, 7))

    # Tạo lưới điểm để quét toàn bộ không gian
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Dự đoán cho từng điểm trên lưới
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Vẽ màu nền ranh giới phân loại
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Vẽ các điểm Train thực tế
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
    
    plt.title(f"Ranh giới phân loại phi tuyến của KNN (k={knn.n_neighbors}, weights='{knn.weights}')")
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
