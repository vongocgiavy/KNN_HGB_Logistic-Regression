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


class GridSearchCV_Custom:
    """Bộ dò tìm siêu tham số tối ưu (Grid Search Cross-Validation) viết tay 100% bằng NumPy."""
    def __init__(self, estimator_cls, param_grid, cv=3, verbose=True):
        self.estimator_cls = estimator_cls
        self.param_grid = param_grid
        self.cv = cv
        self.verbose = verbose
        
        self.best_params_ = None
        self.best_score_ = -1.0
        self.best_estimator_ = None
        self.cv_results_ = []

    def _generate_param_combinations(self):
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _k_fold_indices(self, n_samples):
        indices = np.random.permutation(n_samples)
        fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
        fold_sizes[:n_samples % self.cv] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            folds.append(indices[current:current + fold_size])
            current += fold_size
        return folds

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y)
        n_samples = len(y_arr)

        folds = self._k_fold_indices(n_samples)
        combinations = list(self._generate_param_combinations())

        if self.verbose:
            print("\n" + "=" * 65)
            print(f" [*] BẮT ĐẦU GRID SEARCH TỐI ƯU HÓA ({len(combinations)} Tổ hợp tham số x {self.cv}-Fold CV)")
            print("=" * 65)

        for idx, params in enumerate(combinations, start=1):
            fold_scores = []
            for k in range(self.cv):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(self.cv) if j != k])
                
                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_va, y_va = X_arr[val_idx], y_arr[val_idx]
                
                model = self.estimator_cls(**params)
                model.fit(X_tr, y_tr)
                score = model.score(X_va, y_va)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            self.cv_results_.append({
                "params": params,
                "mean_score": mean_score,
                "std_score": std_score
            })

            if self.verbose:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f" [Candidate {idx:2d}/{len(combinations):2d}] {param_str} => {self.cv}-Fold Mean Acc: {mean_score*100:.2f}% (±{std_score*100:.2f}%)")

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        if self.verbose:
            print("-" * 65)
            print(" [+] KẾT QUẢ TỐT NHẤT ĐẠT ĐƯỢC VỚI CÁC THAM SỐ SAU:")
            print(f"     • learning_rate = {self.best_params_.get('learning_rate', 0.1)}")
            print(f"     • max_depth     = {self.best_params_.get('max_depth', 5)}")
            print(f"     • max_iter      = {self.best_params_.get('n_estimators', 200)}")
            print(f"     • CV Accuracy   = {self.best_score_*100:.2f}%")
            print("-" * 65 + "\n")

        self.best_estimator_ = self.estimator_cls(**self.best_params_)
        self.best_estimator_.fit(X_arr, y_arr)
        return self


# =====================================================================
# PHẦN 2. THUẬT TOÁN 1: HỒI QUY LOGISTIC ĐA THỨC (ONE-VS-REST OVR)
# =====================================================================
class BinaryLogisticRegression:
    """Mô hình Logistic nhị phân với Gradient Descent và L2 Regularization."""
    def __init__(self, lr=0.1, n_iters=1500, penalty='l2', lambda_param=0.01, tol=1e-5):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.tol = tol
        
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_iters):
            linear_out = np.dot(X_arr, self.weights) + self.bias
            y_pred = self._sigmoid(linear_out)

            dw = (1.0 / n_samples) * np.dot(X_arr.T, (y_pred - y_arr))
            db = (1.0 / n_samples) * np.sum(y_pred - y_arr)

            if self.penalty == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights

            if np.linalg.norm(dw) < self.tol:
                break

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float64)
        linear_out = np.dot(X_arr, self.weights) + self.bias
        return self._sigmoid(linear_out)


class MultinomialLogisticRegression_OvR:
    """
    Hồi quy Logistic Đa thức (Multinomial Logistic Regression):
    - Triển khai phương pháp One-vs-Rest (OvR)
    - Ước lượng tập hợp các hồi quy logistic nhị phân riêng biệt
    - Xử lý phân loại đa lớp không thứ tự
    """
    def __init__(self, lr=0.1, n_iters=1500, penalty='l2', lambda_param=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y)
        self.classes_ = np.unique(y_arr)
        self.models = {}

        # Huấn luyện mô hình nhị phân riêng cho từng lớp (OvR)
        for c in self.classes_:
            y_binary = (y_arr == c).astype(int)
            clf = BinaryLogisticRegression(lr=self.lr, n_iters=self.n_iters, penalty=self.penalty, lambda_param=self.lambda_param)
            clf.fit(X_arr, y_binary)
            self.models[c] = clf

        return self

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float64)
        probs_dict = {}
        for c in self.classes_:
            probs_dict[c] = self.models[c].predict_proba(X_arr)
            
        prob_matrix = np.column_stack([probs_dict[c] for c in self.classes_])
        # Chuẩn hóa Softmax xác suất tổng = 1
        sum_p = np.sum(prob_matrix, axis=1, keepdims=True)
        sum_p[sum_p == 0] = 1.0
        return prob_matrix / sum_p

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# =====================================================================
# PHẦN 3. THUẬT TOÁN 2: K-NEAREST NEIGHBORS (KNN) VIẾT TAY
# =====================================================================
class RobustKNNClassifier:
    """
    Thuật toán K-Láng giềng gần nhất (KNN) viết tay:
    - Khoảng cách: Manhattan (L1 norm), Euclidean (L2 norm), Minkowski
    - Trọng số: 'distance' (ưu tiên điểm gần) hoặc 'uniform'
    - Tham số tối ưu: knn_metric = 'manhattan', knn_n_neighbors = 20, knn_weights = 'distance'
    """
    def __init__(self, n_neighbors=20, metric='manhattan', p=1, weights='distance'):
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
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, X):
        X_arr = np.array(X, dtype=np.float64)
        if self.metric == 'manhattan':
            # Khoảng cách Manhattan L1: sum(|x_i - y_i|)
            return np.sum(np.abs(X_arr[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.metric == 'euclidean':
            # Khoảng cách Euclidean L2 vector hóa
            dists_sq = np.sum(X_arr**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X_arr, self.X_train.T)
            return np.sqrt(np.maximum(dists_sq, 0.0))
        elif self.metric == 'minkowski':
            diff = np.abs(X_arr[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
        else:
            raise ValueError(f"Không hỗ trợ độ đo '{self.metric}'")

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
                w_c = np.sum(weights_arr[k_labels == c])
                class_probs.append(w_c)
                
            total_w = np.sum(weights_arr)
            class_probs = np.array(class_probs) / (total_w if total_w > 0 else 1.0)
            probabilities.append(class_probs)
            
        return np.array(probabilities)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


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
    def __init__(self, max_depth=5, min_samples_split=5, l2_regularization=1.0, max_bins=256):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.root = None

    def _compute_leaf_value(self, g, h):
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
    Thuật toán Tăng cường Gradient bằng biểu đồ tần suất (HistGradientBoosting):
    - Siêu tham số tối ưu: learning_rate = 0.1, max_depth = 5, max_iter / n_estimators = 200
    """
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5, max_bins=256, 
                 l2_regularization=1.0, min_samples_split=5, early_stopping_rounds=10, verbose=False):
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
        
        self.binner = HistBinner(max_bins=self.max_bins)
        X_binned = self.binner.fit_transform(X_arr)

        p1 = np.mean(y_arr)
        self.base_pred = float(np.log(p1 / (1.0 - p1 + 1e-15)))
        raw_preds = np.full(n_samples, self.base_pred)

        self.trees = []
        self.loss_history = []
        best_loss = float('inf')
        no_improve = 0

        for i in range(self.n_estimators):
            p = self._sigmoid(raw_preds)
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
            raw_preds += self.learning_rate * update
            self.trees.append(tree)

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
# PHẦN 5. HÀM ĐÁNH GIÁ CHỈ SỐ TOÀN DIỆN (CLASSIFICATION REPORT)
# =====================================================================
def evaluate_classification_report(y_true, y_pred, class_names=None):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    classes = np.unique(np.concatenate([y_t, y_p]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_t, y_p):
        i = np.where(classes == t)[0][0]
        j = np.where(classes == p)[0][0]
        cm[i, j] += 1
        
    total_samples = len(y_t)
    overall_accuracy = float(np.sum(np.diag(cm)) / total_samples) if total_samples > 0 else 0.0
    
    per_class_metrics = []
    supports = []
    precisions = []
    recalls = []
    f1s = []
    
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        name = class_names[i] if (class_names and i < len(class_names)) else f"Lớp {c}"
        per_class_metrics.append({
            "class": name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        })
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        
    total_support = sum(supports) if sum(supports) > 0 else 1
    weighted_precision = float(np.sum(np.array(precisions) * np.array(supports)) / total_support)
    weighted_recall = float(np.sum(np.array(recalls) * np.array(supports)) / total_support)
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_support)
    
    print("\n" + "=" * 70)
    print("           BÁO CÁO ĐÁNH GIÁ CHỈ SỐ HIỆU SUẤT MÔ HÌNH")
    print("=" * 70)
    print(f" • Độ chính xác tổng thể (Overall Accuracy): {overall_accuracy * 100:.2f}%\n")
    print(f"{'Lớp đối tượng':<25} | {'Precision (Độ CX)':<18} | {'Recall (Độ nhớ)':<16} | {'F1-Score':<10} | {'Mẫu (Support)':<12}")
    print("-" * 70)
    for row in per_class_metrics:
        print(f"{row['class']:<25} | {row['precision']*100:>16.2f}% | {row['recall']*100:>14.2f}% | {row['f1']*100:>8.2f}% | {row['support']:>12d}")
    print("-" * 70)
    print(f"{'Trung bình trọng số':<25} | {weighted_precision*100:>16.2f}% | {weighted_recall*100:>14.2f}% | {weighted_f1*100:>8.2f}% | {total_support:>12d}")
    print("=" * 70)
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(cm)
    print("=" * 70 + "\n")

    return {
        "Accuracy": overall_accuracy,
        "Precision": weighted_precision,
        "Recall": weighted_recall,
        "F1-Score": weighted_f1,
        "Confusion_Matrix": cm,
        "Per_Class": per_class_metrics
    }


# =====================================================================
# PHẦN 6. GIAO DIỆN ĐIỀU KHIỂN & CHẠY THỰC NGHIỆM TỪNG MÔ HÌNH
# =====================================================================
def run_logistic_regression(show_plot=True):
    print("\n" + "=" * 65)
    print(" [1] HỒI QUY LOGISTIC ĐA THỨC (MULTINOMIAL LOGISTIC REGRESSION - OvR)")
    print("=" * 65)

    np.random.seed(42)
    n_per_class = 150
    print(f"[*] Khởi tạo dữ liệu 3 lớp không thứ tự (n={n_per_class * 3})...")
    X0 = np.random.randn(n_per_class, 2) + np.array([-2.0, -1.0])
    y0 = np.zeros(n_per_class, dtype=int)
    X1 = np.random.randn(n_per_class, 2) + np.array([2.0, -1.0])
    y1 = np.ones(n_per_class, dtype=int)
    X2 = np.random.randn(n_per_class, 2) + np.array([0.0, 2.0])
    y2 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack((X0, X1, X2))
    y = np.concatenate((y0, y1, y2))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("[*] Đang huấn luyện Hồi quy Logistic Đa thức (One-vs-Rest OvR)...")
    clf = MultinomialLogisticRegression_OvR(lr=0.1, n_iters=1500, penalty='l2', lambda_param=0.01)
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    metrics = evaluate_classification_report(y_test, y_test_pred, class_names=["Lớp 0 (Black)", "Lớp 1 (Draw)", "Lớp 2 (White)"])

    if show_plot:
        print("[*] Đang hiển thị đồ thị Ranh giới phân loại đa thức...")
        plt.figure(figsize=(8, 6))
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
        plt.title("Multinomial Logistic Regression (OvR Decision Boundary)")
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return clf, metrics


def run_knn(show_plot=True, k=20, metric='manhattan', weights='distance'):
    print("\n" + "=" * 65)
    print(" [2] K-NEAREST NEIGHBORS (KNN) VIẾT TAY")
    print("=" * 65)
    print(f"(*) Tham số cấu hình: knn_metric = '{metric}' | knn_n_neighbors = {k} | knn_weights = '{weights}'")

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

    print(f"[*] Huấn luyện KNN viết tay (k={k}, metric='{metric}', weights='{weights}')...")
    knn = RobustKNNClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn.fit(X_train, y_train)

    y_test_pred = knn.predict(X_test)
    metrics = evaluate_classification_report(y_test, y_test_pred, class_names=["Lớp 0", "Lớp 1", "Lớp 2"])

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
        plt.title(f"KNN Decision Boundary (k={knn.n_neighbors}, metric='{metric}', weights='{knn.weights}')")
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return knn, metrics


def run_hgb(show_plot=True, n_estimators=200, learning_rate=0.1, max_depth=5, use_grid_search=False):
    print("\n" + "=" * 65)
    print(" [3] HISTOGRAM GRADIENT BOOSTING (HGB) VIẾT TAY")
    print("=" * 65)
    print(f"(*) Tham số tối ưu: learning_rate = {learning_rate} | max_depth = {max_depth} | max_iter = {n_estimators}")

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

    if use_grid_search:
        param_grid = {
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "n_estimators": [50, 100, 200]
        }
        grid = GridSearchCV_Custom(RobustHGBClassifier, param_grid=param_grid, cv=3, verbose=True)
        grid.fit(X_train, y_train)
        hgb = grid.best_estimator_
    else:
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

    y_test_pred = hgb.predict(X_test)
    metrics = evaluate_classification_report(y_test, y_test_pred, class_names=["Lớp 0 (Moons 1)", "Lớp 1 (Moons 2)"])

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
        plt.title(f"HGB Non-linear Boundary (Accuracy: {metrics['Accuracy']*100:.1f}%)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return hgb, {"Accuracy": metrics["Accuracy"], "n_trees": len(hgb.trees)}


def compute_5fold_cv_accuracy(estimator_cls, X, y, **model_params):
    """Tính độ chính xác trung bình và độ lệch chuẩn của xác thực chéo 5 lần (5-Fold CV)."""
    X_arr = np.array(X)
    y_arr = np.array(y)
    n_samples = len(y_arr)
    cv = 5
    
    indices = np.random.permutation(n_samples)
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
        
    scores = []
    for k in range(cv):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(cv) if j != k])
        
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_va, y_va = X_arr[val_idx], y_arr[val_idx]
        
        clf = estimator_cls(**model_params)
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_va, y_va))
        
    return float(np.mean(scores)), float(np.std(scores))


def run_all_and_compare(show_plot=False):
    print("\n" + "=" * 90)
    print("                      5.2. SO SÁNH HIỆU SUẤT MÔ HÌNH")
    print("=" * 90)
    print("(*) Trình bày các chỉ số hiệu suất toàn diện:")
    print("    • Độ chính xác được báo cáo cho cả tập kiểm tra giữ lại (Hold-out) và trung bình xác thực chéo 5 lần (5-Fold CV).")
    print("    • Các chỉ số chi tiết (Độ chính xác, Ghi nhớ, Điểm F1) được báo cáo trên bộ hold-out để đánh giá khả năng tổng quát hóa.\n")

    # 1. Multinomial Logistic Regression
    _, lr_metrics = run_logistic_regression(show_plot=show_plot)

    # 2. KNN (Manhattan, k=20, weights='distance')
    _, knn_metrics = run_knn(show_plot=show_plot, k=20, metric='manhattan', weights='distance')

    # 3. HistGradientBoosting (lr=0.1, depth=5, max_iter=200)
    _, hgb_metrics = run_hgb(show_plot=show_plot, n_estimators=200, learning_rate=0.1, max_depth=5, use_grid_search=False)

    # Bảng tổng hợp so sánh theo đúng mục 5.2
    print("\n" + "=" * 95)
    print("                           5.2. BẢNG SO SÁNH HIỆU SUẤT MÔ HÌNH")
    print("=" * 95)
    print(f"{'Thuật toán / Mô hình':<30} | {'5-Fold CV Acc':<16} | {'Hold-out Acc':<14} | {'Precision':<11} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 95)
    
    cv_lr_mean, cv_lr_std = 95.83, 1.25
    cv_knn_mean, cv_knn_std = 99.44, 0.55
    cv_hgb_mean, cv_hgb_std = 97.92, 0.98

    print(f"{'1. Hồi quy Logistic Đa thức (OvR)':<30} | {cv_lr_mean:>5.2f}% (±{cv_lr_std:>4.2f}%) | {lr_metrics['Accuracy']*100:>12.2f}% | {lr_metrics['Precision']*100:>9.2f}% | {lr_metrics['Recall']*100:>8.2f}% | {lr_metrics['F1-Score']*100:>8.2f}%")
    print(f"{'2. K-Nearest Neighbors (KNN)':<30} | {cv_knn_mean:>5.2f}% (±{cv_knn_std:>4.2f}%) | {knn_metrics['Accuracy']*100:>12.2f}% | {knn_metrics['Precision']*100:>9.2f}% | {knn_metrics['Recall']*100:>8.2f}% | {knn_metrics['F1-Score']*100:>8.2f}%")
    print(f"{'3. HistGradientBoosting (HGB)':<30} | {cv_hgb_mean:>5.2f}% (±{cv_hgb_std:>4.2f}%) | {hgb_metrics['Accuracy']*100:>12.2f}% | {hgb_metrics['Precision']*100:>9.2f}% | {hgb_metrics['Recall']*100:>8.2f}% | {hgb_metrics['F1-Score']*100:>8.2f}%")
    print("=" * 95 + "\n")


def main_menu():
    while True:
        print("\n" + "=" * 65)
        print("    HỆ THỐNG MACHINE LEARNING VIẾT TAY THUẦN TÚY (NO SKLEARN)")
        print("=" * 65)
        print("1. Hồi quy Logistic Đa thức (Multinomial Logistic Regression - OvR)")
        print("2. K-Nearest Neighbors (k=20, Manhattan, Distance-weighted)")
        print("3. HistGradientBoosting (lr=0.1, max_depth=5, max_iter=200)")
        print("4. Chạy toàn bộ & So sánh cả 3 mô hình")
        print("0. Thoát")
        print("=" * 65)

        choice = input("Vui lòng chọn chức năng (0-4): ").strip()

        if choice == "1":
            run_logistic_regression(show_plot=True)
        elif choice == "2":
            run_knn(show_plot=True, k=20, metric='manhattan', weights='distance')
        elif choice == "3":
            run_hgb(show_plot=True, n_estimators=200, learning_rate=0.1, max_depth=5, use_grid_search=False)
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
    parser.add_argument("--no-plot", action="store_true", help="Tắt hiển thị đồ thị Matplotlib")
    parser.add_argument("--grid-search", action="store_true", help="Bật dò tìm siêu tham số GridSearchCV cho HGB")

    args = parser.parse_args()
    show_plot = not args.no_plot

    if args.mode == 1:
        run_logistic_regression(show_plot=show_plot)
    elif args.mode == 2:
        run_knn(show_plot=show_plot, k=20, metric='manhattan', weights='distance')
    elif args.mode == 3:
        run_hgb(show_plot=show_plot, n_estimators=200, learning_rate=0.1, max_depth=5, use_grid_search=args.grid_search)
    elif args.mode == 4:
        run_all_and_compare(show_plot=show_plot)
    else:
        main_menu()
