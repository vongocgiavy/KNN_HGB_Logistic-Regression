import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure proper path resolution regardless of working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from-scratch models
try:
    from logistic_baseline import (
        StandardScaler as LRStandardScaler,
        RobustLogisticRegression,
        evaluate_metrics as evaluate_lr_metrics
    )
except ImportError:
    from src.logistic_baseline import (
        StandardScaler as LRStandardScaler,
        RobustLogisticRegression,
        evaluate_metrics as evaluate_lr_metrics
    )

try:
    from knn_result import (
        StandardScaler as KNNStandardScaler,
        RobustKNNClassifier,
        CustomPipeline,
        evaluate_multiclass_metrics as evaluate_knn_metrics,
        stratified_train_test_split as knn_train_test_split,
        predict_result_knn
    )
except ImportError:
    from src.knn_result import (
        StandardScaler as KNNStandardScaler,
        RobustKNNClassifier,
        CustomPipeline,
        evaluate_multiclass_metrics as evaluate_knn_metrics,
        stratified_train_test_split as knn_train_test_split,
        predict_result_knn
    )

try:
    from hgb_elo import (
        HistBinner,
        HistDecisionTree,
        RobustHGBClassifier
    )
except ImportError:
    from src.hgb_elo import (
        HistBinner,
        HistDecisionTree,
        RobustHGBClassifier
    )


def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    """Chia tập dữ liệu Train / Test thuần túy bằng NumPy."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    train_size = int((1.0 - test_size) * n_samples)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    if isinstance(X, pd.DataFrame):
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =====================================================================
# 1. CHỨC NĂNG 1: LOGISTIC REGRESSION (FROM SCRATCH)
# =====================================================================
def run_logistic_regression(show_plot=True):
    print("\n" + "=" * 65)
    print(" [1] LOGISTIC REGRESSION THUẦN TÚY (GRADIENT DESCENT + L2 REGULARIZATION)")
    print("=" * 65)

    np.random.seed(42)
    n_samples = 500
    print(f"[*] Đang khởi tạo tập dữ liệu phân loại 2 lớp mẫu (n={n_samples})...")
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-1.5, -1.5])
    y0 = np.zeros(n_samples // 2, dtype=int)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1.5, 1.5])
    y1 = np.ones(n_samples // 2, dtype=int)

    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = LRStandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("[*] Đang huấn luyện Logistic Regression với Early Stopping & L2 Penalty...")
    clf = RobustLogisticRegression(lr=0.1, n_iters=1500, penalty='l2', lambda_param=0.01, tol=1e-5, verbose=True)
    clf.fit(X_train, y_train)

    # Đánh giá
    y_test_pred = clf.predict(X_test)
    metrics = evaluate_lr_metrics(y_test, y_test_pred)

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
        print("[*] Đang hiển thị đồ thị trực quan hóa...")
        plt.figure(figsize=(12, 5))
        
        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(clf.loss_history, color='blue', lw=2)
        plt.title("Logistic Regression - Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Decision Boundary
        plt.subplot(1, 2, 2)
        x0_vals = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
        x1_vals = -(clf.weights[0] * x0_vals + clf.bias) / clf.weights[1]
        plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0 (Train)', alpha=0.6)
        plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='green', label='Class 1 (Train)', alpha=0.6)
        plt.plot(x0_vals, x1_vals, color='black', linestyle='--', lw=2, label='Decision Boundary')
        plt.title("Logistic Regression - Decision Boundary")
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    return clf, metrics


# =====================================================================
# 2. CHỨC NĂNG 2: K-NEAREST NEIGHBORS (FROM SCRATCH)
# =====================================================================
def run_knn(show_plot=True, k=7, weights='distance'):
    print("\n" + "=" * 65)
    print(" [2] K-NEAREST NEIGHBORS THUẦN TÚY (VECTORIZED DISTANCE + DISTANCE WEIGHTING)")
    print("=" * 65)

    np.random.seed(42)
    n_per_class = 150
    print(f"[*] Đang sinh tập dữ liệu 3 lớp đa phân loại phi tuyến (n={n_per_class * 3})...")
    X0 = np.random.randn(n_per_class, 2) * 0.7 + np.array([-2.0, -1.0])
    y0 = np.zeros(n_per_class, dtype=int)
    X1 = np.random.randn(n_per_class, 2) * 0.7 + np.array([2.0, -1.0])
    y1 = np.ones(n_per_class, dtype=int)
    X2 = np.random.randn(n_per_class, 2) * 0.7 + np.array([0.0, 2.0])
    y2 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack((X0, X1, X2))
    y = np.concatenate((y0, y1, y2))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = KNNStandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"[*] Đang chạy phân loại với Robust KNN (k={k}, metric='euclidean', weights='{weights}')...")
    knn = RobustKNNClassifier(n_neighbors=k, metric='euclidean', weights=weights)
    knn.fit(X_train, y_train)

    # Đánh giá
    y_test_pred = knn.predict(X_test)
    metrics = evaluate_knn_metrics(y_test, y_test_pred, knn.classes_)

    print("\n" + "-" * 50)
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f" • Accuracy        : {metrics['accuracy'] * 100:.2f}%")
    print(f" • Precision       : {metrics['precision'] * 100:.2f}% (weighted)")
    print(f" • Recall          : {metrics['recall'] * 100:.2f}% (weighted)")
    print(f" • F1-Score        : {metrics['f1_score'] * 100:.2f}% (weighted)")
    print(f" • Confusion Matrix (3x3):\n{np.array(metrics['confusion_matrix'])}")
    print("-" * 50)

    if show_plot:
        print("[*] Đang hiển thị biểu đồ phân vùng quyết định phi tuyến...")
        plt.figure(figsize=(8, 6))
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
        plt.title(f"KNN Non-linear Decision Boundary (k={knn.n_neighbors}, weights='{knn.weights}')")
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return knn, metrics


# =====================================================================
# 3. CHỨC NĂNG 3: HISTOGRAM GRADIENT BOOSTING (FROM SCRATCH)
# =====================================================================
def run_hgb(show_plot=True, n_estimators=40, learning_rate=0.2, max_depth=4):
    print("\n" + "=" * 65)
    print(" [3] HISTOGRAM GRADIENT BOOSTING THUẦN TÚY (HISTOGRAM BINNING + DECISION TREES)")
    print("=" * 65)

    np.random.seed(42)
    n_samples = 600
    print(f"[*] Đang sinh tập dữ liệu phi tuyến tính (Moons shape, n={n_samples})...")
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

    print(f"[*] Đang huấn luyện HGB ({n_estimators} stages, lr={learning_rate}, depth={max_depth})...")
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
        print("[*] Đang hiển thị biểu đồ Learning Curve và Ranh giới quyết định...")
        plt.figure(figsize=(12, 5))
        
        # Learning Curve
        plt.subplot(1, 2, 1)
        plt.plot(hgb.loss_history, color='purple', lw=2, marker='o', markersize=3)
        plt.title("HGB Learning Curve (Cross-Entropy Loss)")
        plt.xlabel("Boosting Stages")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Decision Boundary
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

    return hgb, {"accuracy": test_acc, "n_trees": len(hgb.trees)}


# =====================================================================
# 4. CHỨC NĂNG 4: SO SÁNH CẢ 3 MÔ HÌNH (FULL COMPARISON)
# =====================================================================
def run_all_and_compare(show_plot=False):
    print("\n" + "#" * 70)
    print("     HUẤN LUYỆN VÀ SO SÁNH CẢ 3 MÔ HÌNH MACHINE LEARNING (FROM SCRATCH)")
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
    print(f"{'2. K-Nearest Neighbors (KNN)':<30} | {knn_metrics['accuracy'] * 100:>20.2f}% | {'Non-linear Multi':<15}")
    print(f"{'3. HistGradientBoosting (HGB)':<30} | {hgb_metrics['accuracy'] * 100:>20.2f}% | {'Ensemble Boosting':<15}")
    print("=" * 75 + "\n")


# =====================================================================
# MENU GIAO DIỆN TƯƠNG TÁC CONSOLE
# =====================================================================
def main_menu():
    while True:
        print("\n" + "=" * 65)
        print("    HỆ THỐNG MACHINE LEARNING THUẦN TÚY (FROM SCRATCH - NO SKLEARN)")
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
            print("\nCảm ơn bạn đã sử dụng hệ thống Machine Learning! Tạm biệt.\n")
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
