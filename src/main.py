import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Đảm bảo mã hóa UTF-8 trên Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================
# PHẦN 1. BỘ TIỀN XỬ LÝ & CHUẨN HÓA DỮ LIỆU VIẾT TAY (FROM SCRATCH)
# =====================================================================
class StandardScaler:
    """Chuẩn hóa dữ liệu về phân phối chuẩn (Mean = 0, Std = 1) viết tay bằng NumPy."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X_arr = np.array(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        # Tránh chia cho 0 nếu một đặc trưng có phương sai bằng 0
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    """Phân chia tập dữ liệu Train / Test viết tay thuần túy."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    train_size = int((1.0 - test_size) * n_samples)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


# =====================================================================
# PHẦN 2. THUẬT TOÁN 1: LOGISTIC REGRESSION VIẾT TAY (FROM SCRATCH)
# =====================================================================
class RobustLogisticRegression:
    """
    Mô hình Hồi quy Logistic viết tay:
    - Thuật toán tối ưu Gradient Descent
    - Hàm kích hoạt Sigmoid có chặn ngưỡng tránh tràn số học (Numerical Stability)
    - Hỗ trợ hàm phạt Regularization (L1 / L2)
    - Dừng sớm (Early Stopping) khi Gradient Norm đạt ngưỡng hội tụ
    """
    def __init__(self, lr=0.1, n_iters=1500, penalty='l2', lambda_param=0.01, tol=1e-5, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.tol = tol
        self.verbose = verbose
        
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        """Hàm kích hoạt Sigmoid an toàn số học."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, y_true, y_pred_proba):
        """Tính hàm mất mát Binary Cross-Entropy kết hợp phạt Regularization."""
        m = len(y_true)
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1.0 - eps)
        
        # Binary Cross-Entropy Loss
        bce = - (1.0 / m) * np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        
        # Thành phần phạt Regularization (không phạt bias)
        reg = 0.0
        if self.penalty == 'l2':
            reg = (self.lambda_param / (2.0 * m)) * np.sum(self.weights ** 2)
        elif self.penalty == 'l1':
            reg = (self.lambda_param / m) * np.sum(np.abs(self.weights))
            
        return bce + reg

    def fit(self, X, y):
        """Huấn luyện mô hình bằng thuật toán Gradient Descent."""
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_iters):
            # 1. Forward Pass
            linear_out = np.dot(X_arr, self.weights) + self.bias
            y_pred = self._sigmoid(linear_out)

            # 2. Tính Loss
            loss = self.compute_loss(y_arr, y_pred)
            self.loss_history.append(loss)

            # 3. Backward Pass (Tính Đạo hàm Gradient)
            dw = (1.0 / n_samples) * np.dot(X_arr.T, (y_pred - y_arr))
            db = (1.0 / n_samples) * np.sum(y_pred - y_arr)

            # Thêm đạo hàm Regularization
            if self.penalty == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.penalty == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)

            # 4. Kiểm tra điều kiện hội tụ sớm (Early Stopping)
            grad_norm = np.linalg.norm(dw)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f" -> Hội tụ sớm tại epoch {epoch} (Gradient Norm: {grad_norm:.6f} < {self.tol})")
                break

            # 5. Cập nhật tham số
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and epoch % (self.n_iters // 10 or 1) == 0:
                print(f"Epoch {epoch:4d}/{self.n_iters} | Loss: {loss:.5f} | Grad Norm: {grad_norm:.6f}")

        return self

    def predict_proba(self, X):
        """Dự đoán xác suất [P(y=0), P(y=1)]."""
        X_arr = np.array(X, dtype=np.float64)
        linear_out = np.dot(X_arr, self.weights) + self.bias
        p1 = self._sigmoid(linear_out)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X, threshold=0.5):
        """Dự đoán nhãn nhị phân {0, 1}."""
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X, y):
        """Tính tỷ lệ dự đoán chính xác Accuracy."""
        return np.mean(self.predict(X) == np.array(y))


def evaluate_logistic_metrics(y_true, y_pred):
    """Tính toán chi tiết Accuracy, Precision, Recall, F1 và Confusion Matrix viết tay."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    tp = np.sum((y_t == 1) & (y_p == 1))
    tn = np.sum((y_t == 0) & (y_p == 0))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion_Matrix": np.array([[tn, fp], [fn, tp]])
    }


# =====================================================================
# PHẦN 3. THUẬT TOÁN 2: K-NEAREST NEIGHBORS (KNN) VIẾT TAY
# =====================================================================
class RobustKNNClassifier:
    """
    Thuật toán K-Láng giềng gần nhất (KNN) viết tay:
    - Tối ưu hóa ma trận khoảng cách Vectorized (Vectorized Distance Matrix)
    - Hỗ trợ đa dạng độ đo: Euclidean, Manhattan, Minkowski
    - Hỗ trợ bỏ phiếu trọng số theo nghịch đảo khoảng cách (Distance-weighted Voting)
    - Phân loại đa lớp phi tuyến tính (Multi-class Classification)
    """
    def __init__(self, n_neighbors=7, metric='euclidean', p=2, weights='distance'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """Lưu trữ tập huấn luyện (Lazy Learning)."""
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, X):
        """Tính toán ma trận khoảng cách song song bằng phép toán ma trận."""
        X_arr = np.array(X, dtype=np.float64)
        if self.metric == 'euclidean':
            # ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A * B_T
            dists_sq = np.sum(X_arr**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X_arr, self.X_train.T)
            return np.sqrt(np.maximum(dists_sq, 0.0))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X_arr[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.metric == 'minkowski':
            diff = np.abs(X_arr[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
        else:
            raise ValueError(f"Không hỗ trợ độ đo '{self.metric}'")

    def predict_proba(self, X):
        """Dự đoán phân phối xác suất cho từng lớp."""
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
                w_c = np.sum(weights_arr[k_labels == c])
                class_probs.append(w_c)
                
            total_w = np.sum(weights_arr)
            class_probs = np.array(class_probs) / (total_w if total_w > 0 else 1.0)
            probabilities.append(class_probs)
            
        return np.array(probabilities)

    def predict(self, X):
        """Dự đoán nhãn có trọng số bình chọn cao nhất."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Tính Accuracy trên tập dữ liệu."""
        return np.mean(self.predict(X) == np.array(y))


def evaluate_multiclass_metrics(y_true, y_pred, classes=None):
    """Tính toán ma trận nhầm lẫn và chỉ số cho bài toán đa lớp viết tay."""
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
    return {
        "Accuracy": accuracy,
        "Confusion_Matrix": cm
    }


# =====================================================================
# PHẦN 4. THUẬT TOÁN 3: HISTOGRAM GRADIENT BOOSTING (HGB) VIẾT TAY
# =====================================================================
class HistBinner:
    """Rời rạc hóa đặc trưng số thực thành các thùng nguyên uint8 (0-255) qua phân vị Quantiles."""
    def __init__(self, max_bins=256):
        self.max_bins = max_bins
        self.bin_thresholds_ = []

    def fit(self, X):
        self.bin_thresholds_ = []
        X_arr = np.array(X, dtype=np.float64)
        n_features = X_arr.shape[1]
        for col in range(n_features):
            values = X_arr[:, col]
            quantiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
            thresholds = np.unique(np.percentile(values, quantiles))
            self.bin_thresholds_.append(thresholds)
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float64)
        X_binned = np.zeros(X_arr.shape, dtype=np.uint8)
        for col in range(X_arr.shape[1]):
            X_binned[:, col] = np.digitize(X_arr[:, col], self.bin_thresholds_[col])
        return X_binned

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class HistNode:
    """Node trong cây quyết định dựa trên Histogram."""
    def __init__(self, feature=None, threshold_bin=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold_bin = threshold_bin
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class HistDecisionTree:
    """Cây quyết định tối ưu hóa dựa trên Histogram Gradients & Hessians."""
    def __init__(self, max_depth=3, min_samples_split=5, l2_regularization=1.0, max_bins=256):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.root = None

    def _compute_leaf_value(self, g, h):
        """Tính giá trị tối ưu của lá: w = - sum(g) / (sum(h) + lambda)."""
        return -np.sum(g) / (np.sum(h) + self.l2_regularization)

    def _find_best_split(self, X_binned, g, h):
        best_gain = -1.0
        best_feat = None
        best_bin = None
        
        G_total, H_total = np.sum(g), np.sum(h)
        parent_score = (G_total ** 2) / (H_total + self.l2_regularization)
        _, n_features = X_binned.shape

        for feat in range(n_features):
            feat_bins = X_binned[:, feat]
            
            # Tạo Histogram tích lũy Gradient và Hessian theo từng bin
            G_hist = np.bincount(feat_bins, weights=g, minlength=self.max_bins)
            H_hist = np.bincount(feat_bins, weights=h, minlength=self.max_bins)
            
            G_L = np.cumsum(G_hist)
            H_L = np.cumsum(H_hist)
            G_R = G_total - G_L
            H_R = H_total - H_L

            valid = (H_L > 0) & (H_R > 0)
            if not np.any(valid):
                continue
            
            gain = 0.5 * (
                (G_L[valid] ** 2) / (H_L[valid] + self.l2_regularization) +
                (G_R[valid] ** 2) / (H_R[valid] + self.l2_regularization) -
                parent_score
            )
            
            max_idx = np.argmax(gain)
            if gain[max_idx] > best_gain:
                best_gain = gain[max_idx]
                best_feat = feat
                valid_bins = np.where(valid)[0]
                best_bin = valid_bins[max_idx]

        return best_feat, best_bin, best_gain

    def _build_tree(self, X_binned, g, h, depth=0):
        n_samples = X_binned.shape[0]
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return HistNode(value=self._compute_leaf_value(g, h))

        best_feat, best_bin, best_gain = self._find_best_split(X_binned, g, h)
        if best_gain <= 0 or best_feat is None:
            return HistNode(value=self._compute_leaf_value(g, h))

        left_mask = X_binned[:, best_feat] <= best_bin
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return HistNode(value=self._compute_leaf_value(g, h))

        left_child = self._build_tree(X_binned[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_child = self._build_tree(X_binned[right_mask], g[right_mask], h[right_mask], depth + 1)

        return HistNode(feature=best_feat, threshold_bin=best_bin, left=left_child, right=right_child)

    def fit(self, X_binned, g, h):
        self.root = self._build_tree(X_binned, g, h, depth=0)
        return self

    def _predict_row(self, node, row_binned):
        if node.is_leaf:
            return node.value
        if row_binned[node.feature] <= node.threshold_bin:
            return self._predict_row(node.left, row_binned)
        return self._predict_row(node.right, row_binned)

    def predict(self, X_binned):
        return np.array([self._predict_row(self.root, row) for row in X_binned])


class RobustHGBClassifier:
    """
    Thuật toán Histogram-based Gradient Boosting Classifier viết tay:
    - Chia dữ liệu vào Histogram Bins (tốc độ cao)
    - Tối ưu hóa chuỗi cây quyết định (Boosting Stages)
    - Tự động dừng sớm (Early Stopping) theo dõi Cross-Entropy Loss
    """
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, max_bins=256, 
                 l2_regularization=1.0, min_samples_split=5, early_stopping_rounds=5, verbose=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.min_samples_split = min_samples_split
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        self.binner = None
        self.trees = []
        self.base_pred = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, y_true, raw_preds):
        p = self._sigmoid(raw_preds)
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples = X_arr.shape[0]
        
        # 1. Rời rạc hóa đặc trưng
        self.binner = HistBinner(max_bins=self.max_bins)
        X_binned = self.binner.fit_transform(X_arr)

        # 2. Khởi tạo dự đoán ban đầu bằng Log-Odds
        p1 = np.mean(y_arr)
        self.base_pred = float(np.log(p1 / (1.0 - p1 + 1e-15)))
        raw_preds = np.full(n_samples, self.base_pred)

        self.trees = []
        self.loss_history = []
        best_loss = float('inf')
        no_improve = 0

        for i in range(self.n_estimators):
            p = self._sigmoid(raw_preds)
            
            # 3. Tính Gradients (g) & Hessians (h)
            g = p - y_arr
            h = p * (1.0 - p)

            # 4. Huấn luyện cây quyết định
            tree = HistDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                l2_regularization=self.l2_regularization,
                max_bins=self.max_bins
            )
            tree.fit(X_binned, g, h)

            # 5. Cập nhật dự đoán
            update = tree.predict(X_binned)
            raw_preds += self.learning_rate * update
            self.trees.append(tree)

            # Đánh giá Loss
            cur_loss = self._compute_loss(y_arr, raw_preds)
            self.loss_history.append(cur_loss)

            if cur_loss < best_loss - 1e-5:
                best_loss = cur_loss
                no_improve = 0
            else:
                no_improve += 1

            if self.verbose and (i + 1) % (self.n_estimators // 5 or 1) == 0:
                print(f"Stage {i+1:3d}/{self.n_estimators} | Loss: {cur_loss:.5f}")

            if no_improve >= self.early_stopping_rounds:
                if self.verbose:
                    print(f" -> Dừng sớm (Early Stopping) tại cây thứ {i+1}")
                break

        return self

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float64)
        X_binned = self.binner.transform(X_arr)
        raw_preds = np.full(X_arr.shape[0], self.base_pred)
        
        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict(X_binned)
            
        p1 = self._sigmoid(raw_preds)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# =====================================================================
# PHẦN 5. GIAO DIỆN ĐIỀU KHIỂN & CHẠY THỰC NGHIỆM TỪNG MÔ HÌNH
# =====================================================================
def run_logistic_regression(show_plot=True):
    print("\n" + "=" * 65)
    print(" [1] LOGISTIC REGRESSION VIẾT TAY (GRADIENT DESCENT + L2)")
    print("=" * 65)

    np.random.seed(42)
    n_samples = 500
    print(f"[*] Khởi tạo dữ liệu phân loại 2 lớp mẫu (n={n_samples})...")
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-1.5, -1.5])
    y0 = np.zeros(n_samples // 2, dtype=int)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1.5, 1.5])
    y1 = np.ones(n_samples // 2, dtype=int)

    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("[*] Đang huấn luyện Logistic Regression viết tay...")
    clf = RobustLogisticRegression(lr=0.1, n_iters=1500, penalty='l2', lambda_param=0.01, tol=1e-5, verbose=True)
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    metrics = evaluate_logistic_metrics(y_test, y_test_pred)

    print("\n" + "-" * 50)
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f" • Accuracy        : {metrics['Accuracy'] * 100:.2f}%")
    print(f" • Precision       : {metrics['Precision'] * 100:.2f}%")
    print(f" • Recall          : {metrics['Recall'] * 100:.2f}%")
    print(f" • F1-Score        : {metrics['F1-Score'] * 100:.2f}%")
    print(f" • Confusion Matrix:\n{metrics['Confusion_Matrix']}")
    print(f" • Trọng số (Weights): {clf.weights}, Bias: {clf.bias:.4f}")
    print("-" * 50)

    if show_plot:
        print("[*] Đang hiển thị đồ thị...")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(clf.loss_history, color='blue', lw=2)
        plt.title("Logistic Regression - Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.subplot(1, 2, 2)
        x0_vals = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
        x1_vals = -(clf.weights[0] * x0_vals + clf.bias) / clf.weights[1]
        plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0', alpha=0.6)
        plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='green', label='Class 1', alpha=0.6)
        plt.plot(x0_vals, x1_vals, color='black', linestyle='--', lw=2, label='Decision Boundary')
        plt.title("Logistic Regression - Decision Boundary")
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    return clf, metrics


def run_knn(show_plot=True, k=7, weights='distance'):
    print("\n" + "=" * 65)
    print(" [2] K-NEAREST NEIGHBORS (KNN) VIẾT TAY")
    print("=" * 65)

    np.random.seed(42)
    n_per_class = 150
    print(f"[*] Khởi tạo dữ liệu 3 lớp đa phân loại phi tuyến (n={n_per_class * 3})...")
    X0 = np.random.randn(n_per_class, 2) * 0.7 + np.array([-2.0, -1.0])
    y0 = np.zeros(n_per_class, dtype=int)
    X1 = np.random.randn(n_per_class, 2) * 0.7 + np.array([2.0, -1.0])
    y1 = np.ones(n_per_class, dtype=int)
    X2 = np.random.randn(n_per_class, 2) * 0.7 + np.array([0.0, 2.0])
    y2 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack((X0, X1, X2))
    y = np.concatenate((y0, y1, y2))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"[*] Huấn luyện KNN viết tay (k={k}, metric='euclidean', weights='{weights}')...")
    knn = RobustKNNClassifier(n_neighbors=k, metric='euclidean', weights=weights)
    knn.fit(X_train, y_train)

    y_test_pred = knn.predict(X_test)
    metrics = evaluate_multiclass_metrics(y_test, y_test_pred, knn.classes_)

    print("\n" + "-" * 50)
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f" • Accuracy        : {metrics['Accuracy'] * 100:.2f}%")
    print(f" • Confusion Matrix (3x3):\n{metrics['Confusion_Matrix']}")
    print("-" * 50)

    if show_plot:
        print("[*] Đang hiển thị đồ thị phân vùng phi tuyến...")
        plt.figure(figsize=(8, 6))
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
        plt.title(f"KNN Decision Boundary (k={knn.n_neighbors}, weights='{knn.weights}')")
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return knn, metrics


def run_hgb(show_plot=True, n_estimators=40, learning_rate=0.2, max_depth=4):
    print("\n" + "=" * 65)
    print(" [3] HISTOGRAM GRADIENT BOOSTING (HGB) VIẾT TAY")
    print("=" * 65)

    np.random.seed(42)
    n_samples = 600
    print(f"[*] Khởi tạo dữ liệu phi tuyến tính hình trăng khuyết (Moons shape, n={n_samples})...")
    t = np.linspace(0, np.pi, n_samples // 2)
    x1 = np.cos(t) + np.random.randn(n_samples // 2) * 0.15
    y1 = np.sin(t) + np.random.randn(n_samples // 2) * 0.15
    x2 = 1 - np.cos(t) + np.random.randn(n_samples // 2) * 0.15
    y2 = 0.5 - np.sin(t) + np.random.randn(n_samples // 2) * 0.15

    X0 = np.column_stack((x1, y1))
    X1 = np.column_stack((x2, y2))
    X = np.vstack((X0, X1))
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2))).astype(int)

    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    print(f"[*] Huấn luyện HGB viết tay ({n_estimators} stages, lr={learning_rate}, depth={max_depth})...")
    hgb = RobustHGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_bins=64,
        l2_regularization=1.5,
        verbose=True
    )
    hgb.fit(X_train, y_train)

    test_acc = hgb.score(X_test, y_test)
    print("\n" + "-" * 50)
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f" • Test Accuracy   : {test_acc * 100:.2f}%")
    print(f" • Tổng số cây     : {len(hgb.trees)}")
    print("-" * 50)

    if show_plot:
        print("[*] Đang hiển thị đồ thị...")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(hgb.loss_history, color='purple', lw=2, marker='o', markersize=3)
        plt.title("HGB Learning Curve (Cross-Entropy Loss)")
        plt.xlabel("Boosting Stages")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.subplot(1, 2, 2)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
        
        Z = hgb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0', alpha=0.7)
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1', alpha=0.7)
        plt.title(f"HGB Non-linear Boundary (Accuracy: {test_acc*100:.1f}%)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return hgb, {"Accuracy": test_acc, "n_trees": len(hgb.trees)}


def run_all_and_compare(show_plot=False):
    print("\n" + "#" * 70)
    print("     HUẤN LUYỆN VÀ SO SÁNH CẢ 3 MÔ HÌNH MACHINE LEARNING VIẾT TAY")
    print("#" * 70)

    # 1. Logistic Regression
    _, lr_metrics = run_logistic_regression(show_plot=show_plot)

    # 2. KNN
    _, knn_metrics = run_knn(show_plot=show_plot)

    # 3. HistGradientBoosting
    _, hgb_metrics = run_hgb(show_plot=show_plot)

    # Bảng tổng hợp so sánh
    print("\n" + "=" * 75)
    print("                    BẢNG TỔNG HỢP SO SÁNH 3 MÔ HÌNH")
    print("=" * 75)
    print(f"{'Mô hình':<30} | {'Độ chính xác (Accuracy)':<25} | {'Ghi chú':<15}")
    print("-" * 75)
    print(f"{'1. Logistic Regression':<30} | {lr_metrics['Accuracy'] * 100:>20.2f}% | {'Linear Baseline':<15}")
    print(f"{'2. K-Nearest Neighbors (KNN)':<30} | {knn_metrics['Accuracy'] * 100:>20.2f}% | {'Non-linear Multi':<15}")
    print(f"{'3. HistGradientBoosting (HGB)':<30} | {hgb_metrics['Accuracy'] * 100:>20.2f}% | {'Ensemble Boosting':<15}")
    print("=" * 75 + "\n")


def main_menu():
    while True:
        print("\n" + "=" * 65)
        print("    HỆ THỐNG MACHINE LEARNING VIẾT TAY THUẦN TÚY (NO SKLEARN)")
        print("=" * 65)
        print("1. Logistic Regression (Gradient Descent + L2 Regularization)")
        print("2. K-Nearest Neighbors (KNN Vectorized Distance + Weighted Voting)")
        print("3. HistGradientBoosting (Histogram Binning + Decision Trees Boosting)")
        print("4. Chạy toàn bộ & So sánh cả 3 mô hình")
        print("0. Thoát")
        print("=" * 65)

        choice = input("Vui lòng chọn chức năng (0-4): ").strip()

        if choice == "1":
            run_logistic_regression(show_plot=True)
        elif choice == "2":
            run_knn(show_plot=True)
        elif choice == "3":
            run_hgb(show_plot=True)
        elif choice == "4":
            run_all_and_compare(show_plot=False)
        elif choice == "0":
            print("\nCảm ơn bạn đã sử dụng hệ thống! Tạm biệt.\n")
            break
        else:
            print("[!] Lựa chọn không hợp lệ. Vui lòng nhập từ 0 đến 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning From Scratch - Pure Python & NumPy")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], help="Chạy trực tiếp chế độ (1-4)")
    parser.add_argument("--no-plot", action="store_true", help="Tắt hiển thị đồ thị Matplotlib (cho môi trường CLI/Headless)")

    args = parser.parse_args()
    show_plot = not args.no_plot

    if args.mode == 1:
        run_logistic_regression(show_plot=show_plot)
    elif args.mode == 2:
        run_knn(show_plot=show_plot)
    elif args.mode == 3:
        run_hgb(show_plot=show_plot)
    elif args.mode == 4:
        run_all_and_compare(show_plot=show_plot)
    else:
        main_menu()
