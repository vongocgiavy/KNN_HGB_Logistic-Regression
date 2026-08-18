import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU: BỘ CHUẨN HÓA DỮ LIỆU
# ==========================================
class StandardScaler:
    """Chuẩn hóa dữ liệu về phân phối chuẩn (mean=0, std=1) từ đầu."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Tránh chia cho 0 nếu std = 0
        self.std[self.std == 0] = 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==========================================
# 2. THUẬT TOÁN LOGISTIC REGRESSION HOÀN CHỈNH
# ==========================================
class RobustLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, penalty='l2', lambda_param=0.01, tol=1e-5, verbose=False):
        """
        Logistic Regression Classifier chuẩn Production/Research.
        
        Parameters:
        -----------
        - lr : float, Tốc độ học (learning rate)
        - n_iters : int, Số vòng lặp tối đa
        - penalty : str, Loại chuẩn hóa ('l2', 'l1', hoặc None)
        - lambda_param : float, Hệ số phạt regularization
        - tol : float, Ngưỡng dừng sớm khi gradient norm < tol
        - verbose : bool, Có in thông báo tiến trình ra màn hình hay không
        """
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.tol = tol
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid ổn định số học (tránh overflow khi z quá lớn/nhỏ)."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, y_true, y_pred_proba):
        """Tính Binary Cross-Entropy Loss kết hợp Regularization."""
        m = len(y_true)
        eps = 1e-15
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
        
        # 1. Binary Cross-Entropy cơ bản
        bce = - (1 / m) * np.sum(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        
        # 2. Thành phần phạt Regularization (không phạt bias)
        reg_loss = 0.0
        if self.penalty == 'l2':
            reg_loss = (self.lambda_param / (2 * m)) * np.sum(self.weights ** 2)
        elif self.penalty == 'l1':
            reg_loss = (self.lambda_param / m) * np.sum(np.abs(self.weights))
            
        return bce + reg_loss

    def fit(self, X, y):
        """Huấn luyện mô hình với Gradient Descent + Regularization + Early Stopping."""
        n_samples, n_features = X.shape
        
        # Khởi tạo tham số
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_iters):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Lưu lại loss hiện tại
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Backward pass (Tính Gradients)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Thêm đạo hàm Regularization vào dw
            if self.penalty == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.penalty == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)

            # Kiểm tra điều kiện dừng sớm (Early Stopping)
            grad_norm = np.linalg.norm(dw)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"-> Hội tụ sớm tại epoch {epoch} (Gradient norm < {self.tol})")
                break

            # Cập nhật trọng số
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # In log định kỳ
            if self.verbose and epoch % (self.n_iters // 10 or 1) == 0:
                print(f"Epoch {epoch:4d}/{self.n_iters} | Loss: {loss:.5f} | Grad Norm: {grad_norm:.6f}")

        return self

    def predict_proba(self, X):
        """Trả về ma trận xác suất 2 cột: [P(y=0), P(y=1)]."""
        linear_model = np.dot(X, self.weights) + self.bias
        prob_1 = self._sigmoid(linear_model)
        prob_0 = 1 - prob_1
        return np.column_stack((prob_0, prob_1))

    def predict(self, X, threshold=0.5):
        """Dự đoán nhãn nhị phân {0, 1} theo ngưỡng threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(int)

    def score(self, X, y):
        """Tính Accuracy trực tiếp."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ==========================================
# 3. CÁC HÀM ĐÁNH GIÁ MÔ HÌNH (METRICS)
# ==========================================
def evaluate_metrics(y_true, y_pred):
    """Tính toán chi tiết Accuracy, Precision, Recall, F1-score và Confusion Matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion_Matrix": np.array([[tn, fp], [fn, tp]])
    }


# ==========================================
# 4. CHẠY THỬ NGHIỆM VÀ TRỰC QUAN HÓA
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Tạo tập dữ liệu 2 chiều (để có thể vẽ đồ thị decision boundary)
    print("[1] Đang tạo dữ liệu mẫu...")
    n_samples = 400
    # Lớp 0
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-1.5, -1.5])
    y0 = np.zeros(n_samples // 2, dtype=int)
    # Lớp 1
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1.5, 1.5])
    y1 = np.ones(n_samples // 2, dtype=int)

    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))

    # Xáo trộn dữ liệu
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # 2. Chia Train / Test (80% Train - 20% Test)
    train_size = int(0.8 * n_samples)
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 4. Huấn luyện mô hình
    print("\n[2] Bắt đầu huấn luyện mô hình Logistic Regression...")
    clf = RobustLogisticRegression(
        lr=0.1, 
        n_iters=1500, 
        penalty='l2', 
        lambda_param=0.01, 
        tol=1e-5, 
        verbose=True
    )
    clf.fit(X_train, y_train)

    # 5. Đánh giá trên tập Test
    print("\n[3] Kết quả đánh giá trên tập Test:")
    y_test_pred = clf.predict(X_test)
    metrics = evaluate_metrics(y_test, y_test_pred)

    print(f" • Accuracy : {metrics['Accuracy'] * 100:.2f}%")
    print(f" • Precision: {metrics['Precision'] * 100:.2f}%")
    print(f" • Recall   : {metrics['Recall'] * 100:.2f}%")
    print(f" • F1-Score : {metrics['F1-Score'] * 100:.2f}%")
    print(f" • Confusion Matrix:\n{metrics['Confusion_Matrix']}")
    print(f" • Trọng số w: {clf.weights}, Bias b: {clf.bias:.4f}")

    # 6. Vẽ biểu đồ trực quan hóa
    print("\n[4] Đang hiển thị đồ thị...")
    plt.figure(figsize=(12, 5))

    # Đồ thị 1: Learning Curve (Hàm mất mát qua các Epoch)
    plt.subplot(1, 2, 1)
    plt.plot(clf.loss_history, color='blue', lw=2)
    plt.title("Đường cong mất mát (Learning Curve)")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Đồ thị 2: Decision Boundary (Ranh giới phân loại)
    plt.subplot(1, 2, 2)
    # Ranh giới phân loại: w0*x0 + w1*x1 + b = 0 => x1 = -(w0*x0 + b) / w1
    x0_vals = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
    x1_vals = -(clf.weights[0] * x0_vals + clf.bias) / clf.weights[1]

    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0 (Train)', alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='green', label='Class 1 (Train)', alpha=0.6)
    plt.plot(x0_vals, x1_vals, color='black', linestyle='--', lw=2, label='Decision Boundary')
    
    plt.title("Ranh giới phân loại (Decision Boundary)")
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
