# Lichess Chess Machine Learning Project

Dự án Machine Learning phân tích và dự đoán ván cờ từ tập dữ liệu cờ vua trực tiếp:
`lichess_db_standard_rated_2015-08.pgn.zst` (File PGN nén Zstandard).

---

## 📌 Tổng quan 3 Thuật toán và Nhiệm vụ riêng biệt

Dự án được xây dựng với **ĐÚNG 3 thuật toán**, mỗi thuật toán thực hiện một nhiệm vụ riêng biệt và độc lập:

| Thuật toán | Đầu vào (Features) | Đầu ra (Target / Goal) | Vai trò & Mục đích |
| :--- | :--- | :--- | :--- |
| **1. Logistic Regression** | `WhiteElo`, `BlackElo`, `EloDiff` | `Result` (0: Black, 1: Draw, 2: White) | **BASELINE** để so sánh hiệu năng dự đoán kết quả ván cờ |
| **2. KNN (K-Nearest Neighbors)** | `Moves` (Chuỗi nước đi) | Opening tương tự, ECO & Danh sách K ván gần nhất | **Tìm ván cờ & Opening tương tự** dựa trên độ tương đồng Cosine (Không dùng Elo, Không dùng cột Opening để tính khoảng cách) |
| **3. HistGradientBoosting (HGB)** | `WhiteElo`, `BlackElo`, `EloDiff` | `Result` (0: Black, 1: Draw, 2: White) | **Mô hình chính dự đoán kết quả ván cờ** dựa trên chênh lệch Elo |

> ⚠️ **LƯU Ý CỰC KỲ QUAN TRỌNG:**
> - KNN KHÔNG dùng Elo, chỉ dùng `Moves` để biểu diễn TF-IDF vector và tính Cosine Distance.
> - Logistic Regression và HGB sử dụng Elo để dự đoán `Result`.
> - Không trộn bài toán KNN và HGB thành một bài toán duy nhất.

---

## 📁 Cấu trúc Project

```
demo mh/
│
├── data/
│   ├── lichess_db_standard_rated_2015-08.pgn.zst   # File nén PGN gốc
│   └── processed_games.csv                         # Dataset trung gian đã parse & cache
│
├── src/
│   ├── data_loader.py                              # Đọc & stream dataset PGN.ZST, kiểm tra thống kê
│   ├── preprocessing.py                            # Tiền xử lý, tính EloDiff, mã hóa Result, làm sạch Moves
│   ├── logistic_baseline.py                        # Mô hình Logistic Regression (Baseline Elo -> Result)
│   ├── knn_opening.py                              # Mô hình KNN (Moves -> Opening tương tự)
│   ├── hgb_elo.py                                  # Mô hình HistGradientBoosting (Elo -> Result)
│   ├── comparison.py                               # Báo cáo so sánh Baseline vs HGB
│   └── main.py                                     # Giao diện điều khiển chính (CLI Interactive)
│
├── models/
│   ├── logistic_baseline.joblib                    # Pipeline Logistic Regression đã huấn luyện
│   ├── knn_opening.joblib                          # Search index & TF-IDF Vectorizer KNN
│   └── hgb_elo.joblib                              # Mô hình HistGradientBoosting đã huấn luyện
│
├── outputs/
│   ├── logistic_metrics.json                       # Báo cáo chỉ số Logistic Regression
│   ├── hgb_metrics.json                            # Báo cáo chỉ số HGB
│   └── model_comparison.txt                        # Báo cáo so sánh chi tiết
│
├── requirements.txt                                # Thư viện phụ thuộc
└── README.md                                       # Hướng dẫn chi tiết dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Project

### 1. Cài đặt môi trường
Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### 2. Chạy giao diện Menu tương tác (CLI)
Chạy file giao diện chính:
```bash
python src/main.py
```

Menu điều khiển sẽ xuất hiện:
```
============================================================
     HỆ THỐNG MACHINE LEARNING PHÂN TÍCH VÁN CỜ LICHESS
============================================================
1. Logistic Regression (BASELINE dự đoán Result từ Elo)
2. KNN tìm Opening tương tự (Dựa trên chuỗi nước đi Moves)
3. HistGradientBoosting (HGB dự đoán Result từ Elo)
4. Chạy toàn bộ (Full Pipeline & So sánh mô hình)
0. Thoát
============================================================
```

### 3. Chạy từng chế độ qua Dòng lệnh (Non-interactive Mode)
- **Chạy toàn bộ Pipeline & So sánh:**
  ```bash
  python src/main.py --mode 4
  ```
- **Chế độ KNN (Nhập nước đi):**
  ```bash
  python src/main.py --mode 2 --moves "1. e4 c5 2. Nf3 d6 3. d4"
  ```
- **Chế độ HGB (Nhập Elo):**
  ```bash
  python src/main.py --mode 3 --white-elo 1800 --black-elo 1500
  ```

---

## 📊 Chi tiết Tiền xử lý & Tính toán Feature

1. **Elo chênh lệch (`EloDiff`)**:
   $$\text{EloDiff} = \text{WhiteElo} - \text{BlackElo}$$
   *Ví dụ: WhiteElo = 1800, BlackElo = 1500 $\rightarrow$ EloDiff = 300.*

2. **Mã hóa Kết quả (`ResultEncoded`)**:
   - `0` = Black thắng (`0-1`)
   - `1` = Hòa (`1/2-1/2`)
   - `2` = White thắng (`1-0`)

3. **Biểu diễn Nước đi cho KNN (`Moves -> Vector`)**:
   - Chuỗi nước đi được làm sạch (loại bỏ số thứ tự nước đi, chú thích, kết quả).
   - Biến đổi thành n-gram (1-4 grams) bằng `TfidfVectorizer`.
   - Tính khoảng cách Cosine $\text{Distance} = 1 - \text{Cosine Similarity}$.
   - Chọn K ván gần nhất và lấy yếu tố mở đầu (Opening) xuất hiện nhiều nhất (Majority Voting).

---

## 📈 So sánh Mô hình Dự đoán Результат (Elo -> Result)

Mô hình **Logistic Regression (Baseline)** và **HistGradientBoosting (HGB)** được chia tập dữ liệu 80% Train / 20% Test (`random_state=42`).

Bảng so sánh kết quả kiểm thử tiêu chuẩn:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (BASELINE)** | **0.6440** | **0.6227** | **0.6440** | **0.6331** |
| **HistGradientBoosting (HGB)** | **0.6270** | **0.6070** | **0.6270** | **0.6168** |

*Đánh giá:* Mối quan hệ giữa chênh lệch điểm Elo và xác suất thắng/thua mang hình thái tuyến tính Sigmoid đặc trưng của hàm Logistic, giúp Logistic Regression hoạt động cực kỳ mượt mà và đóng vai trò làm Baseline lý tưởng để so sánh.
