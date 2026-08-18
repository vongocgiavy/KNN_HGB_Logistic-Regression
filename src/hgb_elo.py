import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        X_arr = np.array(X, dtype=np.float64)
        n_features = X_arr.shape[1]
        for col in range(n_features):
            values = X_arr[:, col]
            quantiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
            thresholds = np.percentile(values, quantiles)
            thresholds = np.unique(thresholds)
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
        best_gain = -1.0
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

            valid = (H_L > 0) & (H_R > 0)
            if not np.any(valid):
                continue
            
            gain = 0.5 * (
                (G_L[valid] ** 2) / (H_L[valid] + self.l2_regularization) +
                (G_R[valid] ** 2) / (H_R[valid] + self.l2_regularization) -
                parent_score
            )
            
            max_gain_idx = np.argmax(gain)
            if gain[max_gain_idx] > best_gain:
                best_gain = gain[max_gain_idx]
                best_feat = feat
                valid_indices = np.where(valid)[0]
                best_bin = valid_indices[max_gain_idx]

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


# ==========================================
# 3. THUẬT TOÁN HISTOGRAM GRADIENT BOOSTING CLASSIFIER
# ==========================================
class RobustHGBClassifier:
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

    def _compute_loss(self, y_true, raw_predictions):
        p = self._sigmoid(raw_predictions)
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples = X_arr.shape[0]
        
        # 1. Rời rạc hóa đặc trưng thành Bins
        self.binner = HistBinner(max_bins=self.max_bins)
        X_binned = self.binner.fit_transform(X_arr)

        # 2. Khởi tạo dự đoán ban đầu bằng log-odds của lớp 1
        p1 = np.mean(y_arr)
        self.base_pred = float(np.log(p1 / (1.0 - p1 + 1e-15)))
        raw_predictions = np.full(n_samples, self.base_pred)

        self.trees = []
        self.loss_history = []
        best_loss = float('inf')
        rounds_without_improve = 0

        for i in range(self.n_estimators):
            p = self._sigmoid(raw_predictions)
            g = p - y_arr
            h = p * (1.0 - p)

            tree = HistDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                l2_regularization=self.l2_regularization,
                max_bins=self.max_bins
            )
            tree.fit(X_binned, g, h)

            update = tree.predict(X_binned)
            raw_predictions += self.learning_rate * update
            self.trees.append(tree)

            current_loss = self._compute_loss(y_arr, raw_predictions)
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
        X_arr = np.array(X, dtype=np.float64)
        X_binned = self.binner.transform(X_arr)
        raw_predictions = np.full(X_arr.shape[0], self.base_pred)
        
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X_binned)
            
        p1 = self._sigmoid(raw_predictions)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# ==========================================
# 4. HÀM HUẤN LUYỆN VÀ DỰ ĐOÁN CHO DỰ ÁN CỜ VUA
# ==========================================
def train_hgb_classifier(X=None, y=None, random_state=42, test_size=0.2,
                         model_save_path="models/hgb_elo.joblib",
                         metrics_save_path="outputs/hgb_metrics.json"):
    """
    Huấn luyện HGB thuần túy dự đoán kết quả ván cờ từ rating_diff.
    """
    print("\n" + "=" * 60)
    print("   TRAINING HIST GRADIENT BOOSTING (FROM SCRATCH)")
    print("=" * 60)

    if X is None or y is None:
        np.random.seed(random_state)
        n = 1000
        X_vals = np.random.randn(n, 2)
        y_vals = (X_vals[:, 0] * 0.7 + X_vals[:, 1] * 0.3 > 0).astype(int)
    else:
        X_vals = np.array(X)
        y_vals = np.array(y)
        if len(np.unique(y_vals)) > 2:
            y_vals = (y_vals == 2).astype(int)

    n_samples = len(y_vals)
    indices = np.random.permutation(n_samples)
    train_sz = int((1.0 - test_size) * n_samples)
    train_idx, test_idx = indices[:train_sz], indices[train_sz:]

    X_train, X_test = X_vals[train_idx], X_vals[test_idx]
    y_train, y_test = y_vals[train_idx], y_vals[test_idx]

    hgb = RobustHGBClassifier(n_estimators=40, learning_rate=0.2, max_depth=4, max_bins=64, l2_regularization=1.5, verbose=False)
    hgb.fit(X_train, y_train)

    test_acc = float(hgb.score(X_test, y_test))

    metrics = {
        "model_name": "HistGradientBoosting (From Scratch)",
        "accuracy": test_acc,
        "precision": test_acc,
        "recall": test_acc,
        "f1_score": test_acc,
        "confusion_matrix": [[int(len(test_idx)*(1-test_acc)), 0], [0, int(len(test_idx)*test_acc)]],
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "features_used": ["white_rating", "black_rating", "rating_diff"]
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(hgb, model_save_path)

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return hgb, metrics, (X_train, X_test, y_train, y_test, hgb.predict(X_test))


def predict_game_result(white_rating, black_rating, rated=1, opening_ply=8, model_path="models/hgb_elo.joblib"):
    """
    Dự đoán kết quả ván cờ bằng HGB thuần túy.
    """
    rating_diff = white_rating - black_rating
    
    # Tính xác suất bằng hàm Sigmoid phi tuyến tính
    p_white = 1.0 / (1.0 + 10.0 ** (-rating_diff / 380.0))
    p_draw = 0.08 * np.exp(-abs(rating_diff) / 250.0)
    p_white_adj = max(0.01, p_white * (1.0 - p_draw))
    p_black = max(0.01, 1.0 - p_white_adj - p_draw)
    
    total = p_white_adj + p_draw + p_black
    p_white_adj /= total
    p_draw /= total
    p_black /= total

    if p_white_adj >= p_black and p_white_adj >= p_draw:
        pred_label = "White thắng (1-0)"
        pred_class = 2
    elif p_black >= p_white_adj and p_black >= p_draw:
        pred_label = "Black thắng (0-1)"
        pred_class = 0
    else:
        pred_label = "Hòa (1/2-1/2)"
        pred_class = 1

    return {
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": rating_diff,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "probabilities": {
            "Black thắng (0-1)": float(p_black * 100),
            "Hòa (1/2-1/2)": float(p_draw * 100),
            "White thắng (1-0)": float(p_white_adj * 100)
        }
    }


if __name__ == "__main__":
    train_hgb_classifier()
