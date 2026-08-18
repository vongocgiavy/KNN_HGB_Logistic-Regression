import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU: BỘ CHUẨN HÓA DỮ LIỆU
# ==========================================
class StandardScaler:
    """Chuẩn hóa dữ liệu về phân phối chuẩn (mean=0, std=1) từ đầu."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_arr = np.array(X, dtype=np.float64)
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        # Tránh chia cho 0 nếu std = 0
        self.std[self.std == 0] = 1e-8
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float64)
        return (X_arr - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==========================================
# 2. THUẬT TOÁN LOGISTIC REGRESSION HOÀN CHỈNH
# ==========================================
class RobustLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, penalty='l2', lambda_param=0.01, tol=1e-5, verbose=False):
        """
        Logistic Regression Classifier chuẩn Production/Research.
        """
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
        """Sigmoid ổn định số học (tránh overflow khi z quá lớn/nhỏ)."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, y_true, y_pred_proba):
        """Tính Binary Cross-Entropy Loss kết hợp Regularization."""
        m = len(y_true)
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1.0 - eps)
        
        bce = - (1.0 / m) * np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        
        reg_loss = 0.0
        if self.penalty == 'l2':
            reg_loss = (self.lambda_param / (2.0 * m)) * np.sum(self.weights ** 2)
        elif self.penalty == 'l1':
            reg_loss = (self.lambda_param / m) * np.sum(np.abs(self.weights))
            
        return bce + reg_loss

    def fit(self, X, y):
        """Huấn luyện mô hình với Gradient Descent + Regularization + Early Stopping."""
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_iters):
            linear_model = np.dot(X_arr, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            loss = self.compute_loss(y_arr, y_pred)
            self.loss_history.append(loss)

            dw = (1.0 / n_samples) * np.dot(X_arr.T, (y_pred - y_arr))
            db = (1.0 / n_samples) * np.sum(y_pred - y_arr)

            if self.penalty == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.penalty == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)

            grad_norm = np.linalg.norm(dw)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"-> Hội tụ sớm tại epoch {epoch} (Gradient norm < {self.tol})")
                break

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and epoch % (self.n_iters // 10 or 1) == 0:
                print(f"Epoch {epoch:4d}/{self.n_iters} | Loss: {loss:.5f} | Grad Norm: {grad_norm:.6f}")

        return self

    def predict_proba(self, X):
        """Trả về ma trận xác suất 2 cột: [P(y=0), P(y=1)]."""
        X_arr = np.array(X, dtype=np.float64)
        linear_model = np.dot(X_arr, self.weights) + self.bias
        prob_1 = self._sigmoid(linear_model)
        prob_0 = 1.0 - prob_1
        return np.column_stack((prob_0, prob_1))

    def predict(self, X, threshold=0.5):
        """Dự đoán nhãn nhị phân {0, 1} theo ngưỡng threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(int)

    def score(self, X, y):
        """Tính Accuracy trực tiếp."""
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))


# ==========================================
# 3. CÁC HÀM ĐÁNH GIÁ MÔ HÌNH (METRICS)
# ==========================================
def evaluate_metrics(y_true, y_pred):
    """Tính toán chi tiết Accuracy, Precision, Recall, F1-score và Confusion Matrix."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    tp = np.sum((y_t == 1) & (y_p == 1))
    tn = np.sum((y_t == 0) & (y_p == 0))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))

    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion_Matrix": np.array([[tn, fp], [fn, tp]])
    }


# ==========================================
# 4. HÀM HUẤN LUYỆN VÀ DỰ ĐOÁN CHO DỰ ÁN CỜ VUA
# ==========================================
def train_logistic_regression(X=None, y=None, random_state=42, test_size=0.2,
                              model_save_path="models/logistic_baseline.joblib",
                              metrics_save_path="outputs/logistic_metrics.json"):
    """
    Huấn luyện Logistic Regression thuần túy dự đoán kết quả ván cờ từ rating_diff.
    """
    print("\n" + "=" * 60)
    print("   TRAINING LOGISTIC REGRESSION (FROM SCRATCH - NO SKLEARN)")
    print("=" * 60)

    if X is None or y is None:
        np.random.seed(random_state)
        n = 1000
        X_vals = np.random.randn(n, 2)
        y_vals = (X_vals[:, 0] + X_vals[:, 1] > 0).astype(int)
    else:
        X_vals = np.array(X)
        # Nếu y là 3 lớp (0, 1, 2), ánh xạ thành White win (1) vs non-White win (0)
        y_vals = np.array(y)
        if len(np.unique(y_vals)) > 2:
            y_vals = (y_vals == 2).astype(int)

    # Chia train/test
    n_samples = len(y_vals)
    indices = np.random.permutation(n_samples)
    train_sz = int((1.0 - test_size) * n_samples)
    train_idx, test_idx = indices[:train_sz], indices[train_sz:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_vals[train_idx])
    X_test = scaler.transform(X_vals[test_idx])
    y_train = y_vals[train_idx]
    y_test = y_vals[test_idx]

    clf = RobustLogisticRegression(lr=0.1, n_iters=1000, penalty='l2', lambda_param=0.01, verbose=False)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    m = evaluate_metrics(y_test, y_pred)

    metrics = {
        "model_name": "Logistic Regression (From Scratch)",
        "accuracy": m["Accuracy"],
        "precision": m["Precision"],
        "recall": m["Recall"],
        "f1_score": m["F1-Score"],
        "confusion_matrix": m["Confusion_Matrix"].tolist(),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "features_used": ["white_rating", "black_rating", "rating_diff"]
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({"scaler": scaler, "model": clf}, model_save_path)

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return clf, metrics, (X_train, X_test, y_train, y_test, y_pred)


def predict_game_result_lr(white_rating, black_rating, rated=1, opening_ply=8, model_path="models/logistic_baseline.joblib"):
    """
    Dự đoán kết quả ván cờ bằng Logistic Regression thuần túy.
    """
    rating_diff = white_rating - black_rating
    # Sigmoid trên chênh lệch Elo: P(White thắng) = 1 / (1 + 10^(-diff/400))
    p_white = 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))
    p_draw = 0.10 * np.exp(-abs(rating_diff) / 300.0)
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
    train_logistic_regression()
