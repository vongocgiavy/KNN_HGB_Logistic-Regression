# Lichess Chess Machine Learning Project (100% From Scratch)

Hệ thống Machine Learning phân tích, dự đoán và khai phá dữ liệu ván cờ Lichess với **3 thuật toán cốt lõi được xây dựng hoàn toàn từ đầu (From Scratch - Không sử dụng Scikit-Learn)**:

1. **Robust Logistic Regression**: Tối ưu Gradient Descent, L1/L2 Regularization, Early Stopping.
2. **Robust K-Nearest Neighbors (KNN)**: Vectorized Distance Matrix ($||A-B||^2 = ||A||^2 + ||B||^2 - 2AB^T$), Distance-weighted Voting.
3. **Robust Histogram-based Gradient Boosting (HGB)**: Histogram Quantile Binning (0-255 uint8), Prefix-sum Histogram Decision Trees, L2 Regularized Leaf Weights, Shrinkage & Early Stopping.

---

## 📌 Tổng quan 3 Thuật toán (From-Scratch)

| Thuật toán | Cơ chế & Tối ưu hóa thuật toán | Đầu ra / Đánh giá | Mục đích & Ứng dụng |
| :--- | :--- | :--- | :--- |
| **1. Logistic Regression** | Gradient Descent, Sigmoid tránh tràn số học, L2 Regularization, Early Stopping | Accuracy, Weighted Precision, Recall, F1-Score, Confusion Matrix, Learning Curve | **Baseline Model** phân loại tuyến tính |
| **2. K-Nearest Neighbors (KNN)** | Vectorized Euclidean/Manhattan/Minkowski Distance, $K$-láng giềng gần nhất, Distance-weighted | Accuracy, Multiclass Confusion Matrix, Non-linear Decision Boundary | **Non-linear Multi-class Classifier** phân vùng phi tuyến |
| **3. HistGradientBoosting (HGB)** | Quantile Binning, Histogram Gradient/Hessian cumsum, Cây quyết định chia nhánh theo Gain tối đa | Test Accuracy, Boosting Stages Loss Curve, Non-linear Complex Boundary | **State-of-the-Art Ensemble Boosting** xử lý dữ liệu phức tạp |

---

## 📁 Cấu trúc Thư mục

```
demo mh/
│
├── data/
│   ├── lichess_db_standard_rated_2015-08.pgn.zst   # File nén PGN gốc
│   ├── processed_games.csv                         # Dataset đã xử lý
│   └── filtered_processed_games.csv                # Dataset đã lọc
│
├── src/
│   ├── logistic_baseline.py                        # Logistic Regression thuần túy (From Scratch)
│   ├── knn_result.py                               # KNN Result Classifier thuần túy (From Scratch)
│   ├── knn_opening.py                              # KNN đa lớp & trực quan hóa phi tuyến (From Scratch)
│   ├── hgb_elo.py                                  # Histogram Gradient Boosting thuần túy (From Scratch)
│   ├── comparison.py                               # Báo cáo so sánh các mô hình
│   ├── data_loader.py                              # Đọc và stream dataset
│   ├── preprocessing.py                            # Tiền xử lý dữ liệu
│   ├── eda_results.py                              # Phân tích khám phá dữ liệu (EDA)
│   └── main.py                                     # Chương trình điều khiển chính (CLI & Menu)
│
├── outputs/                                        # Báo cáo metrics JSON, biểu đồ phân tích
├── app.py                                          # Web Dashboard Streamlit
├── requirements.txt                                # Danh sách thư viện
└── README.md                                       # Tài liệu hướng dẫn
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chạy Console Menu tương tác
```bash
python src/main.py
```

### 3. Chạy từng chế độ qua dòng lệnh (CLI)
- **Chạy Logistic Regression:**
  ```bash
  python src/main.py --mode 1
  ```
- **Chạy K-Nearest Neighbors (KNN):**
  ```bash
  python src/main.py --mode 2
  ```
- **Chạy Histogram Gradient Boosting (HGB):**
  ```bash
  python src/main.py --mode 3
  ```
- **Chạy so sánh cả 3 mô hình:**
  ```bash
  python src/main.py --mode 4
  ```
- **Tắt đồ thị popup (dành cho headless/terminal):**
  ```bash
  python src/main.py --mode 4 --no-plot
  ```

### 4. Khởi chạy giao diện Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
