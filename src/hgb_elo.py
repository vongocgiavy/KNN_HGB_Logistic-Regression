import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. TIỀN XỬ LÝ: RỜI RẠC HÓA DỮ LIỆU THÀNH BINS (HISTOGRAM BINNING)
# ==========================================
class HistBinner:
    """Chuyển đổi dữ liệu số thực liên tục thành các thùng nguyên uint8 (0-255)."""
    def __init__(self, max_bins=256):
        self.max_bins = max_bins
        self.bin_thresholds_ = []

    def fit(self, X):
        self.bin_thresholds_ = []
        n_features = X.shape[1]
        for col in range(n_features):
            values = X[:, col]
            # Tìm các phân vị (quantiles) để tạo ngưỡng chia bin đều nhau
            quantiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
            thresholds = np.percentile(values, quantiles)
            # Loại bỏ các ngưỡng trùng lặp
            thresholds = np.unique(thresholds)
            self.bin_thresholds_.append(thresholds)
        return self

    def transform(self, X):
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for col in range(X.shape[1]):
            # np.digitize ánh xạ giá trị thực vào chỉ số bin (0 -> max_bins-1)
            X_binned[:, col] = np.digitize(X[:, col], self.bin_thresholds_[col])
        return X_binned

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==========================================
# 2. CÂY QUYẾT ĐỊNH DỰA TRÊN HISTOGRAM (HIST DECISION TREE)
# ==========================================
class HistNode:
    def __init__(self, feature=None, threshold_bin=None, left=None, right=None, value=None):
        self.feature = feature              # Chỉ số đặc trưng được chọn để chia
        self.threshold_bin = threshold_bin  # Ngưỡng bin (0 - 255) tại điểm phân tách
        self.left = left                    # Nhánh con trái
        self.right = right                  # Nhánh con phải
        self.value = value                  # Giá trị trả về nếu là node lá (Leaf value)

    @property
    def is_leaf(self):
        return self.value is not None


class HistDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=5, l2_regularization=1.0, max_bins=256):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.root = None

    def _compute_leaf_value(self, g, h):
        """Tính giá trị tối ưu của node lá: w = - sum(g) / (sum(h) + lambda)"""
        return -np.sum(g) / (np.sum(h) + self.l2_regularization)

    def _find_best_split(self, X_binned, g, h):
        best_gain = -1
        best_feat = None
        best_bin = None
        
        G_total, H_total = np.sum(g), np.sum(h)
        parent_score = (G_total ** 2) / (H_total + self.l2_regularization)
        
        n_samples, n_features = X_binned.shape

        for feat in range(n_features):
            feat_bins = X_binned[:, feat]
            
            # Xây dựng Histogram gom tổng Gradient và Hessian theo từng bin
            G_hist = np.bincount(feat_bins, weights=g, minlength=self.max_bins)
            H_hist = np.bincount(feat_bins, weights=h, minlength=self.max_bins)
            
            # Tính tổng tích lũy (Prefix sum) từ trái sang phải
            G_L = np.cumsum(G_hist)
            H_L = np.cumsum(H_hist)
            
            G_R = G_total - G_L
            H_R = H_total - H_L

            # Tránh chia cho 0 hoặc split rỗng
            valid = (H_L > 0) & (H_R > 0)
            if not np.any(valid):
                continue
            
            # Gain = 1/2 * [ (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - (G_total^2 / (H_total + lambda)) ]
            gain = 0.5 * (
                (G_L[valid] ** 2) / (H_L[valid] + self.l2_regularization) +
                (G_R[valid] ** 2) / (H_R[valid] + self.l2_regularization) -
                parent_score
            )
            
            max_gain_idx = np.argmax(gain)
            if gain[max_gain_idx] > best_gain:
                best_gain = gain[max_gain_idx]
                best_feat = feat
                # Tìm bin ngưỡng tương ứng
                valid_indices = np.where(valid)[0]
                best_bin = valid_indices[max_gain_idx]

        return best_feat, best_bin, best_gain

    def _build_tree(self, X_binned, g, h, depth=0):
        n_samples = X_binned.shape[0]

        # Điều kiện dừng
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return HistNode(value=self._compute_leaf_value(g, h))

        best_feat, best_bin, best_gain = self._find_best_split(X_binned, g, h)

        # Nếu không tìm thấy split giúp tăng Gain
        if best_gain <= 0 or best_feat is None:
            return HistNode(value=self._compute_leaf_value(g, h))

        # Chia tập dữ liệu
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


# ==========================================
# 3. THUẬT TOÁN HISTOGRAM GRADIENT BOOSTING CLASSIFIER
# ==========================================
class RobustHGBClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, max_bins=256, 
                 l2_regularization=1.0, min_samples_split=5, early_stopping_rounds=5, verbose=False):
        """
        Histogram-based Gradient Boosting Classifier.
        
        Parameters:
        -----------
        - n_estimators : int, Số lượng cây (boosting stages)
        - learning_rate: float, Hệ số co shrinkage (eta)
        - max_depth    : int, Độ sâu tối đa mỗi cây
        - max_bins     : int, Số lượng thùng histogram (mặc định 256)
        - l2_regularization : float, Hệ số phạt L2 cho trọng số lá cây
        """
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

    def _compute_loss(self, y_true, raw_predictions):
        p = self._sigmoid(raw_predictions)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # 1. Rời rạc hóa đặc trưng thành Bins
        self.binner = HistBinner(max_bins=self.max_bins)
        X_binned = self.binner.fit_transform(X)

        # 2. Khởi tạo dự đoán ban đầu bằng log-odds của lớp 1
        p1 = np.mean(y)
        self.base_pred = np.log(p1 / (1 - p1 + 1e-15))
        raw_predictions = np.full(n_samples, self.base_pred)

        self.trees = []
        self.loss_history = []
        best_loss = float('inf')
        rounds_without_improve = 0

        for i in range(self.n_estimators):
            # Xác suất dự đoán hiện tại
            p = self._sigmoid(raw_predictions)
            
            # 3. Tính Gradients (g) và Hessians (h) cho Binary Cross-Entropy
            g = p - y               # g = p_i - y_i
            h = p * (1.0 - p)       # h = p_i * (1 - p_i)

            # 4. Fit cây quyết định trên Histogram
            tree = HistDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                l2_regularization=self.l2_regularization,
                max_bins=self.max_bins
            )
            tree.fit(X_binned, g, h)

            # 5. Cập nhật raw predictions
            update = tree.predict(X_binned)
            raw_predictions += self.learning_rate * update
            self.trees.append(tree)

            # Tính Loss và kiểm tra Early Stopping
            current_loss = self._compute_loss(y, raw_predictions)
            self.loss_history.append(current_loss)

            if current_loss < best_loss - 1e-5:
                best_loss = current_loss
                rounds_without_improve = 0
            else:
                rounds_without_improve += 1

            if self.verbose and (i + 1) % (self.n_estimators // 5 or 1) == 0:
                print(f"Tree {i+1:3d}/{self.n_estimators} | Loss: {current_loss:.5f}")

            if rounds_without_improve >= self.early_stopping_rounds:
                if self.verbose:
                    print(f"-> Early stopping tại cây thứ {i+1}")
                break

        return self

    def predict_proba(self, X):
        X_binned = self.binner.transform(X)
        raw_predictions = np.full(X.shape[0], self.base_pred)
        
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X_binned)
            
        p1 = self._sigmoid(raw_predictions)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ==========================================
# 4. CHẠY THỬ NGHIỆM VÀ TRỰC QUAN HÓA
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Tạo tập dữ liệu phi tuyến (2 vòng cung lồng nhau - Moons shape)
    print("[1] Đang sinh tập dữ liệu phi tuyến tính...")
    n_samples = 600
    t = np.linspace(0, np.pi, n_samples // 2)
    # Moon 1
    x1 = np.cos(t) + np.random.randn(n_samples // 2) * 0.15
    y1 = np.sin(t) + np.random.randn(n_samples // 2) * 0.15
    # Moon 2
    x2 = 1 - np.cos(t) + np.random.randn(n_samples // 2) * 0.15
    y2 = 0.5 - np.sin(t) + np.random.randn(n_samples // 2) * 0.15

    X0 = np.column_stack((x1, y1))
    X1 = np.column_stack((x2, y2))
    X = np.vstack((X0, X1))
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2))).astype(int)

    # Xáo trộn
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # 2. Chia Train / Test (80% - 20%)
    train_sz = int(0.8 * len(y))
    X_train, X_test = X[:train_sz], X[train_sz:]
    y_train, y_test = y[:train_sz], y[train_sz:]

    # 3. Huấn luyện HGB Classifier
    print("\n[2] Huấn luyện Robust Histogram Gradient Boosting (HGB) Classifier...")
    hgb = RobustHGBClassifier(
        n_estimators=40,
        learning_rate=0.2,
        max_depth=4,
        max_bins=64,
        l2_regularization=1.5,
        verbose=True
    )
    hgb.fit(X_train, y_train)

    # 4. Đánh giá trên tập Test
    test_acc = hgb.score(X_test, y_test)
    print("\n[3] Kết quả kiểm thử:")
    print(f" • Test Accuracy: {test_acc * 100:.2f}%")
    print(f" • Tổng số cây được dựng: {len(hgb.trees)}")

    # 5. Vẽ biểu đồ
    print("\n[4] Đang hiển thị đồ thị trực quan hóa...")
    plt.figure(figsize=(12, 5))

    # Đồ thị 1: Learning Curve
    plt.subplot(1, 2, 1)
    plt.plot(hgb.loss_history, color='purple', lw=2, marker='o', markersize=3)
    plt.title("HGB Learning Curve (Cross-Entropy Loss)")
    plt.xlabel("Số lượng Cây (Boosting Stages)")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Đồ thị 2: Decision Boundary Phi Tuyến Phức Tạp
    plt.subplot(1, 2, 2)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    
    Z = hgb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0 (Test)', alpha=0.7)
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1 (Test)', alpha=0.7)
    plt.title(f"HGB Non-linear Decision Boundary (Accuracy: {test_acc*100:.1f}%)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
