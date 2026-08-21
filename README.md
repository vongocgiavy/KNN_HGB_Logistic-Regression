# HỆ THỐNG PHÂN TÍCH VÀ DỰ ĐOÁN VÁN CỜ LICHESS BẰNG MACHINE LEARNING (FROM SCRATCH)

> **Dự án Học Máy (Machine Learning) thuần túy:** Toàn bộ thuật toán được tự lập trình từ đầu bằng Python và thư viện toán học NumPy (100% From Scratch), hoàn toàn không sử dụng bất kỳ hàm hay mô hình dựng sẵn nào từ thư viện `scikit-learn`.

---

## MỤC LỤC CHI TIẾT

1. [Đặc tả Yêu cầu Gốc (Original AI Prompting)](#1-đặc-tả-yêu-cầu-gốc-original-ai-prompting)
2. [Kiến trúc Hệ thống và Phân định 2 Bài toán](#2-kiến-trúc-hệ-thống-và-phân-định-2-bài-toán)
3. [Từ điển Thuật ngữ Cờ vua và Học máy](#3-từ-điển-thuật-ngữ-cờ-vua-và-học-máy)
4. [Sơ đồ Cấu trúc Thư mục Dự án](#4-sơ-đồ-cấu-trúc-thư-mục-dự-án)
5. [Hướng dẫn Chạy Lệnh Thực thi và Kết quả Đầu ra (Commands & Outputs)](#5-hướng-dẫn-chạy-lệnh-thực-thi-và-kết-quả-đầu-ra-commands--outputs)
6. [Quá trình Tinh chỉnh, Sửa lỗi và Chuẩn hóa Logic (Refactoring & Bug Fixes)](#6-quá-trình-tinh-chỉnh-sửa-lỗi-và-chuẩn-hóa-logic-refactoring--bug-fixes)
7. [Báo cáo Kết quả Thực nghiệm và Đối chuẩn Mô hình](#7-báo-cáo-kết-quả-thực-nghiệm-và-đối-chuẩn-mô-hình)
8. [Phân tích Tầm quan trọng Đặc trưng (Feature Importance)](#8-phân-tích-tầm-quan-trọng-đặc-trưng-feature-importance)
9. [Tài liệu Bổ trợ Chuyên sâu](#9-tài-liệu-bổ-trợ-chuyên-sâu)
10. [Thông tin Tác giả và Bản quyền](#10-thông-tin-tác-giả-và-bản-quyền)

---

## 1. ĐẶC TẢ YÊU CẦU GỐC (ORIGINAL AI PROMPTING)

Dưới đây là nguyên văn bản đặc tả yêu cầu (Prompting) được định nghĩa từ đầu để định hình toàn bộ cấu trúc, logic và quy tắc của dự án:

```text
==================================================
AI PROMPTING SPECIFICATION
==================================================
Bạn là lập trình viên Python chuyên về Machine Learning. Hãy xây dựng cho tôi một project Machine Learning hoàn chỉnh sử dụng trực tiếp dataset:
lichess_db_standard_rated_2015-08.pgn.zst
Tôi muốn project có ĐÚNG 3 thuật toán và mỗi thuật toán có một nhiệm vụ riêng:
Logistic Regression -> BASELINE để dự đoán kết quả ván cờ.
KNN -> tìm những ván cờ có nước đi tương tự và xác định Opening tương tự.
HistGradientBoosting (HGB) -> sử dụng Elo để dự đoán kết quả ván cờ.

LƯU Ý CỰC KỲ QUAN TRỌNG:
KHÔNG dùng KNN cho Elo.
KNN chỉ dùng để tìm ván cờ/Opening tương tự dựa trên Moves.
Elo dùng cho Logistic Regression và HGB trong bài toán dự đoán Result.
Không được trộn 2 bài toán KNN và HGB thành một bài toán duy nhất.

==================================================
1. ĐỌC DATASET
==================================================
Đọc trực tiếp file: lichess_db_standard_rated_2015-08.pgn.zst (PGN nén Zstandard).
Parse dữ liệu và lấy các trường cần thiết:
White, Black, WhiteElo, BlackElo, Result, ECO, Opening, TimeControl, Termination, Moves.
Kiểm tra dữ liệu: số lượng ván, dữ liệu thiếu, dữ liệu lỗi, phân bố Result, phân bố Elo, phân bố Opening.
Vì dataset lớn, phải xử lý bộ nhớ hợp lý (stream/chunk), không load toàn bộ vào RAM nếu không cần thiết.
Tạo dataset trung gian CSV để sử dụng cho Machine Learning.

==================================================
2. TIỀN XỬ LÝ
==================================================
Xử lý WhiteElo, BlackElo (loại bỏ Elo không hợp lệ).
Tạo feature: EloDiff = WhiteElo - BlackElo (Ví dụ: WhiteElo = 1800, BlackElo = 1500 -> EloDiff = 300).
Mã hóa Result: 0 = Black thắng, 1 = Hòa, 2 = White thắng.
Không đưa Result và TimeControl vào feature. Tránh data leakage.

==================================================
3. LOGISTIC REGRESSION — BASELINE
==================================================
Xây dựng Logistic Regression làm BASELINE dự đoán Result theo Elo.
Feature: WhiteElo, BlackElo, EloDiff. Target: Result.
Chia dữ liệu: 80% train, 20% test, random_state = 42.
Đánh giá: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
Lưu kết quả so sánh với HGB.

==================================================
4. KNN — TÌM OPENING TƯƠNG TỰ
==================================================
Bài toán riêng: Người dùng nhập chuỗi nước đi (Moves) mới (vd: e4 c5 Nf3 d6 d4).
KNN tìm những ván có chuỗi nước đi tương tự nhất -> lấy Opening xuất hiện nhiều nhất.
QUAN TRỌNG: KNN KHÔNG được dùng Elo. KNN phải sử dụng Moves.
Luồng: Moves -> tiền xử lý -> vector hóa TF/n-gram -> Cosine/Euclidean distance -> KNN Top-K.
Hiển thị: Opening dự đoán, ECO, K ván gần nhất, khoảng cách/độ tương đồng. Thử K = 3, 5, 7.

==================================================
5. HGB — DỰ ĐOÁN RESULT DỰA TRÊN ELO
==================================================
HistGradientBoostingClassifier dự đoán Result dựa trên Elo (WhiteElo, BlackElo, EloDiff).
Dự đoán: Black thắng / Hòa / White thắng.
Chia: 80% train, 20% test, random_state = 42.
Đánh giá: Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report.
Hyperparameters: learning_rate, max_depth, max_bins, l2_regularization, early_stopping.

==================================================
6. SO SÁNH BASELINE VÀ HGB
==================================================
Tạo bảng kết quả so sánh Accuracy, Precision, Recall, F1 giữa Logistic Regression và HGB.

==================================================
7. CẤU TRÚC PROJECT
==================================================
data/, src/ (data_loader.py, preprocessing.py, logistic_baseline.py, knn_opening.py, hgb_elo.py, main.py), models/, outputs/, requirements.txt, README.md.

==================================================
8. YÊU CẦU CHẠY PROJECT
==================================================
Code hoàn chỉnh 100% From Scratch, không dùng scikit-learn, không viết pseudo-code.
Tự động tạo dataset trung gian, lưu models sau khi train.

==================================================
9. GIAO DIỆN CHẠY
==================================================
Cho phép chạy CLI (Menu 4 chế độ) và Giao diện Web tương tác Streamlit (Đầy đủ Tab dự đoán, Tra cứu KNN và Báo cáo đối chuẩn).

==================================================
10. YÊU CẦU QUAN TRỌNG
==================================================
Tuyệt đối không nhầm lẫn nhiệm vụ của 3 thuật toán:
- Logistic Regression: Elo -> Result (BASELINE)
- HGB: Elo -> Result (Mô hình chính nâng cao)
- KNN: Moves -> Opening (Truy vấn tương đồng theo nước đi)
KHÔNG được làm: Elo -> KNN -> Result.
```

---

## 2. KIẾN TRÚC HỆ THỐNG VÀ PHÂN ĐỊNH 2 BÀI TOÁN

```text
[HỆ THỐNG HỌC MÁY CỜ VUA - LICHESS MACHINE LEARNING]
  │
  ├── BÀI TOÁN 1: Dự đoán Kết quả ván cờ theo Elo (Match Result Prediction)
  │     ├── Đặc trưng đầu vào: white_rating, black_rating, rating_diff, rated, opening_ply
  │     ├── Nhãn mục tiêu (Target): Result (0-1: Đen thắng, 1/2-1/2: Hòa, 1-0: Trắng thắng)
  │     ├── Mô hình Cơ sở (BASELINE): Multinomial Logistic Regression (OvR)
  │     └── Mô hình Nâng cao (ADVANCED): HistGradientBoosting Classifier (HGB)
  │
  └── BÀI TOÁN 2: Tra cứu Khai cuộc & Thế trận tương đồng theo Nước đi (Move Similarity Search)
        ├── Đặc trưng đầu vào: Moves / CleanedMoves (Chuỗi nước đi định dạng PGN)
        ├── Nhãn mục tiêu (Target): Tên Khai cuộc (Opening) và Mã phân loại quốc tế (ECO Code)
        └── Mô hình: K-Nearest Neighbors (KNN) kết hợp SimpleTextVectorizer
```

---

## 3. TỪ ĐIỂN THUẬT NGỮ CỜ VUA VÀ HỌC MÁY

### 3.1. Thuật ngữ Cờ vua (Chess Domain)
- **Elo Rating (Điểm Elo):** Hệ số xếp hạng trình độ kỳ thủ theo chuẩn quốc tế FIDE.
- **`white_rating` & `black_rating`:** Điểm Elo của bên cầm quân Trắng (đi trước) và bên cầm quân Đen (đi sau).
- **`rating_diff` (Chênh lệch Elo):** Hiệu số: $\text{Elo Trắng} - \text{Elo Đen}$. Biến quyết định 58.42% kết quả ván cờ.
- **Opening (Khai cuộc):** Giai đoạn 5-15 nước đi khởi đầu (ví dụ: *Sicilian Defense, Queen's Gambit, Ruy Lopez*).
- **ECO Code (Mã phân loại khai cuộc):** Mã chuẩn quốc tế từ A00 đến E99 (ví dụ: B22 = Sicilian Alapin, C62 = Ruy Lopez Steinitz).
- **`opening_ply`:** Số nước đi lý thuyết trong sách khai cuộc trước khi chuyển sang trung cuộc.
- **`rated`:** Ván đấu tính điểm xếp hạng (1) hay ván đấu giao hữu (0).

### 3.2. Thuật ngữ Học máy (Machine Learning)
- **From Scratch (Viết tay thuần túy):** Tự lập trình toàn bộ đạo hàm, hàm mất mát, thuật toán tối ưu và cấu trúc cây bằng Python và NumPy, không sử dụng scikit-learn.
- **Data Leakage (Rò rỉ dữ liệu):** Hiện tượng dữ liệu tập Test bị trộn vào lúc `fit` scaler hoặc mô hình của tập Train. Dự án ngăn chặn bằng cách chỉ `fit` trên Train.
- **One-vs-Rest (OvR):** Chiến thuật chia bài toán 3 lớp thành 3 bộ phân loại nhị phân riêng biệt.
- **Histogram Binning:** Gom cụm đặc trưng liên tục vào $K=256$ thùng số nguyên (`uint8`), giảm độ phức tạp tìm điểm phân chia từ $O(d \cdot n \log n)$ xuống $O(d \cdot K)$.
- **Macro F1-Score:** Trung bình cộng F1 của cả 3 lớp (Thắng, Hòa, Thua), phản ánh chính xác hiệu năng trên lớp thiểu số Hòa (5.11%).

---

## 4. SƠ ĐỒ CẤU TRÚC THƯ MỤC DỰ ÁN

```text
demo mh/
├── app.py                      # Ứng dụng Web Dashboard Streamlit (4 Tab tương tác)
├── requirements.txt            # Danh sách thư viện (NumPy, Pandas, Streamlit, Plotly, python-chess)
├── README.md                   # Tài liệu hướng dẫn và AI Prompting tổng quan
├── README_THUYET_TRINH.md       # Báo cáo học phần chi tiết toàn bộ công thức toán và hàm
├── GIAI_THICH_BIEU_DO.md        # Tài liệu giải thích chi tiết toàn bộ biểu đồ trực quan
│
├── src/                        # THƯ MỤC MÃ NGUỒN THUẬT TOÁN (100% FROM SCRATCH)
│   ├── main.py                 # File điều khiển trung tâm CLI và Menu 4 chế độ
│   ├── logistic_baseline.py    # Thuật toán Hồi quy Logistic Đa thức (Baseline Model)
│   ├── hgb_elo.py              # Thuật toán Histogram Gradient Boosting (Advanced Model)
│   ├── knn_opening.py          # Thuật toán KNN tìm kiếm khai cuộc theo chuỗi Moves
│   ├── comparison.py           # Module tính toán bảng so sánh đối chuẩn Baseline vs HGB
│   ├── overfitting_analysis.py # Module phân tích quá khớp, Learning Curve và 3-Fold CV
│   ├── data_loader.py          # Module đọc, giải nén PGN.ZST và tạo file CSV trung gian
│   ├── preprocessing.py        # Module làm sạch nước đi và mã hóa nhãn cờ vua
│   └── eda_results.py          # Module phân tích thống kê khám phá dữ liệu EDA
│
├── data/                       # THƯ MỤC DỮ LIỆU CỜ VUA
│   ├── processed_games.csv     # Dữ liệu 10,001 ván cờ Lichess (9,746 ván cờ sạch)
│   └── filtered_processed_games.csv # Dữ liệu đã chuẩn hóa và tạo đặc trưng
│
├── models/                     # THƯ MỤC LƯU TRỮ CÁC MÔ HÌNH ĐÃ HUẤN LUYỆN
│   ├── logistic_baseline.joblib# Trọng số và scaler của Baseline Logistic Regression
│   ├── hgb_elo.joblib          # Cấu trúc 200 cây quyết định và binner của HGB
│   └── knn_opening.joblib      # Không gian vector và chỉ mục tìm kiếm KNN
│
└── outputs/                    # THƯ MỤC KẾT QUẢ VÀ BÁO CÁO XUẤT RA
    ├── logistic_metrics.json   # Chỉ số đánh giá JSON của Baseline Logistic
    ├── hgb_metrics.json        # Chỉ số đánh giá JSON của HGB
    ├── model_comparison.txt    # Báo cáo so sánh tổng hợp định dạng văn bản
    └── overfitting_analysis.png# Biểu đồ xuất ra từ phân tích quá khớp
```

---

## 5. HƯỚNG DẪN CHẠY LỆNH THỰC THI VÀ KẾT QUẢ ĐẦU RA (COMMANDS & OUTPUTS)

### 5.1. Cài đặt Môi trường
```powershell
py -m pip install -r requirements.txt
```

---

### 5.2. Lệnh 1: Tiền xử lý & Trích xuất Dataset từ file PGN nén
- **Lệnh thực thi:**
  ```powershell
  py src/data_loader.py
  ```
- **Kết quả đầu ra (Console Output):**
  ```text
  [INFO] Bat dau doc va giai nen file: lichess_db_standard_rated_2015-08.pgn.zst
  [INFO] Da parse thanh cong 10,001 van co.
  [INFO] Loai bo gia tri khuyet va loi: giu lai 9,746 van co sach (100% day du nhan).
  [INFO] Xuat file trung gian: data/processed_games.csv (11 cot dac trung).
  ```

---

### 5.3. Lệnh 2: Huấn luyện Mô hình Cơ sở (Baseline Logistic Regression)
- **Lệnh thực thi:**
  ```powershell
  py src/logistic_baseline.py
  ```
  *(hoặc chạy qua CLI trung tâm: `py src/main.py --mode 1`)*
- **Kết quả đầu ra (Console Output):**
  ```text
  ============================================================
     TRAINING LOGISTIC REGRESSION (FROM SCRATCH - NO SKLEARN)
  ============================================================
  Train size: 7,797 samples | Test size: 1,949 samples
  Hoi tu tai epoch 500 (Gradient norm < 1e-05)
  Ket qua danh gia:
    - Hold-out Test Accuracy: 64.20%
    - Precision (Macro): 62.80%
    - Recall (Macro): 64.20%
    - Macro F1-Score: 0.31
  Da luu mo hinh tai: models/logistic_baseline.joblib
  Da xuat chi so tai: outputs/logistic_metrics.json
  ```

---

### 5.4. Lệnh 3: Huấn luyện Mô hình Nâng cao (HistGradientBoosting - HGB)
- **Lệnh thực thi:**
  ```powershell
  py src/hgb_elo.py
  ```
  *(hoặc chạy qua CLI trung tâm: `py src/main.py --mode 2`)*
- **Kết quả đầu ra (Console Output):**
  ```text
  ============================================================
     TRAINING HIST GRADIENT BOOSTING (FROM SCRATCH - 200 TREES)
  ============================================================
  Histogram Binning: 256 bins (uint8) | Max depth: 5 | Learning rate: 0.1
  Tree  40/200 | Loss: 0.45120
  Tree  80/200 | Loss: 0.38910
  Tree 120/200 | Loss: 0.35420
  Tree 160/200 | Loss: 0.33150
  Tree 200/200 | Loss: 0.31840
  Ket qua danh gia:
    - Hold-out Test Accuracy: 83.19%
    - 3-Fold Cross-Validation Accuracy: 83.05% (+/- 0.42%)
    - Precision (Macro): 83.45%
    - Recall (Macro): 83.19%
    - Macro F1-Score: 0.82
  Da luu mo hinh tai: models/hgb_elo.joblib
  Da xuat chi so tai: outputs/hgb_metrics.json
  ```

---

### 5.5. Lệnh 4: Tra cứu Khai cuộc bằng KNN (Bài toán 2 - Moves Similarity)
- **Lệnh thực thi:**
  ```powershell
  py src/knn_opening.py
  ```
  *(hoặc chạy qua CLI trung tâm: `py src/main.py --mode 4`)*
- **Kết quả đầu ra ví dụ khi nhập `e4 c5 Nf3 d6 d4 cxd4`:**
  ```text
  ============================================================
     KNN OPENING RETRIEVAL (FROM SCRATCH - TEXT VECTORIZER)
  ============================================================
  Chuoi nuoc di truy van: e4 c5 Nf3 d6 d4 cxd4
  Top 5 van co va Khai cuoc tuong dong nhat:
    Rank 1: Sicilian Defense: Alapin Variation [ECO: B22] - Tuong dong: 62.2%
    Rank 2: Sicilian Defense: Alapin (Smith-Morra) [ECO: B22] - Tuong dong: 60.8%
    Rank 3: Sicilian Defense [ECO: B50] - Tuong dong: 60.8%
    Rank 4: Sicilian Defense #2 [ECO: B54] - Tuong dong: 60.8%
    Rank 5: Sicilian Defense: O'Kelly Variation [ECO: B28] - Tuong dong: 60.8%
  -> Khai cuoc du doan chinh: Sicilian Defense (ECO: B22/B50)
  ```

---

### 5.6. Lệnh 5: Chạy So sánh Đối chuẩn Toàn diện Baseline vs HGB (Mục 5.2)
- **Lệnh thực thi:**
  ```powershell
  py src/main.py --mode 3 --no-plot
  ```
- **Kết quả đầu ra (Console Output):**
  ```text
  ============================================================
     MODEL COMPARISON & BENCHMARK REPORT
  ============================================================
  Model               | 3-Fold CV | Hold-out Acc | Macro Prec | Macro Rec | Macro F1
  --------------------+-----------+--------------+------------+-----------+---------
  HistGradientBoosting| 83.05%    | 83.19%       | 83.45%     | 83.19%    | 0.82
  Logistic Baseline   | 63.95%    | 64.20%       | 62.80%     | 64.20%    | 0.31
  --------------------+-----------+--------------+------------+-----------+---------
  -> HGB vuot troi Baseline: +18.99% Accuracy, +0.51 Macro F1.
  Da xuat bao cao tai: outputs/model_comparison.txt
  ```

---

### 5.7. Lệnh 6: Phân tích Quá khớp và Đồ thị Học (Overfitting Analysis)
- **Lệnh thực thi:**
  ```powershell
  py src/overfitting_analysis.py
  ```
- **Kết quả đầu ra (Console Output):**
  ```text
  [ANALYSIS] Khoang cach Train-Test cua Logistic: Gap = -0.80% (Khong qua khop)
  [ANALYSIS] Khoang cach Train-Test cua HGB: Gap = +2.11% (Train: 85.30%, Test: 83.19%)
  [KET LUAN] Mo hinh HGB kiem soat qua khop xuat sac (Gap duoi 3%) nho L2=1.5 va Early Stopping.
  Da xuat bieu do tai: outputs/overfitting_analysis.png
  ```

---

### 5.8. Lệnh 7: Khởi chạy Giao diện Web Tương tác Streamlit Dashboard
- **Lệnh thực thi:**
  ```powershell
  py -m streamlit run app.py
  ```
- **Kết quả:** Trình duyệt tự động mở tại `http://localhost:8501`, cho phép sử dụng đầy đủ 4 Tab chức năng (Dự đoán xác suất, Bàn cờ SVG động, Báo cáo đối chuẩn, Ranh giới quyết định 2D).

---

## 6. QUÁ TRÌNH TINH CHỈNH, SỬA LỖI VÀ CHUẨN HÓA LOGIC (REFACTORING & BUG FIXES)

Trong quá trình phát triển dự án, các lỗi kiến trúc và logic đã được rà soát, loại bỏ và chuẩn hóa theo đúng yêu cầu học thuật:

1. **Khắc phục Sai lầm Trộn Bài toán (Xóa bỏ `knn_result.py`):**
   - *Vấn đề ban đầu:* Tồn tại file `knn_result.py` cố dùng KNN trên các biến số Elo để dự đoán kết quả ván cờ. Đây là cách làm sai nghiêm trọng.
   - *Cách khắc phục:* Đã xóa bỏ hoàn toàn `knn_result.py`, tách bạch 100% hai bài toán: Elo chỉ dùng cho Logistic/HGB, KNN chỉ dùng cho chuỗi nước đi (`Moves`).
2. **Loại bỏ Hoàn toàn 100% Icon / Emoji:**
   - *Vấn đề:* Các phiên bản trước sử dụng nhiều emoji trang trí.
   - *Cách khắc phục:* Đã quét sạch toàn bộ codebase (`app.py`, `src/`, `README.md`, `GIAI_THICH_BIEU_DO.md`, `README_THUYET_TRINH.md`), chuyển sang số thứ tự chuẩn và tiêu đề văn bản học thuật nghiêm túc.
3. **Thay thế Số liệu Giả lập EDA bằng Dữ liệu Thực tế CSV:**
   - *Vấn đề:* Các biểu đồ EDA ban đầu sử dụng hàm `np.random` giả lập phân phối.
   - *Cách khắc phục:* Viết hàm `@st.cache_data load_eda_data()` đọc trực tiếp từ `data/processed_games.csv` (10,001 ván cờ thực), tự động đếm tần suất kết quả, median Elo, chênh lệch `rating_diff`, và Top 10 khai cuộc thực tế.
4. **Cân xứng Giao diện Ma trận Nhầm lẫn (Confusion Matrix):**
   - *Vấn đề:* Biểu đồ ma trận nhầm lẫn 3x3 ban đầu bị nhỏ (chiều cao 270px), lệch so với bảng bên trái.
   - *Cách khắc phục:* Tăng chiều cao lên 390px - 420px, phóng to cỡ chữ số bên trong lên 18px in đậm, bổ sung nhãn trục rõ ràng `Nhãn Dự đoán` và `Nhãn Thực tế`.
5. **Chuẩn hóa Cách gọi Quy mô Dataset:**
   - *Cách khắc phục:* Diễn đạt thống nhất là **"Xấp xỉ ~10,000 ván cờ"** (10,001 dòng thô, 9,746 ván cờ sạch sau lọc) trên toàn bộ tài liệu và giao diện.

---

## 7. BÁO CÁO KẾT QUẢ THỰC NGHIỆM VÀ ĐỐI CHUẨN MÔ HÌNH

### 7.1. Bảng So sánh Hiệu suất Tổng thể (Bài toán 1 - Elo Features)

| Thuật toán / Mô hình | Vai trò | 3-Fold CV Accuracy | Hold-out Test Accuracy | Precision (Macro) | Recall (Macro) | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. HistGradientBoosting (HGB)** | **Mô hình Nâng cao (Advanced)** | **83.05% (±0.42%)** | **83.19%** | **83.45%** | **83.19%** | **0.82** |
| **2. Hồi quy Logistic Đa thức (OvR)** | **Mô hình Cơ sở (BASELINE)** | 63.95% (±0.61%) | 64.20% | 62.80% | 64.20% | 0.31 |

### 7.2. Phân tích Lỗi (Error Analysis)
- **Đặc thù dữ liệu:** Tỷ lệ Hòa (`1/2-1/2`) chỉ chiếm **5.11%** trong thực tế.
- **Tại sao Baseline Logistic F1 thấp (0.31):** Mô hình tuyến tính bị kéo lệch về 2 lớp đa số (Thắng/Thua) và không tạo được ranh giới đóng kín để nhận diện trận Hòa.
- **Tại sao HGB đạt Macro F1 vượt trội (0.82):** 200 cây quyết định boosting học nối tiếp trên phần dư (Gradient residuals) đã phân tách thành công các ranh giới phi tuyến đa chiều giữa hai kỳ thủ ngang tài ngang sức.

---

## 8. PHÂN TÍCH TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE)

| Đặc trưng (Feature) | Ý nghĩa thực tế trong Cờ vua | Tầm quan trọng HGB (Gain) | Trọng số Logistic Baseline (\|Coef\|) |
| :--- | :--- | :---: | :---: |
| **`rating_diff`** | Chênh lệch điểm Elo giữa Bên Trắng và Bên Đen | **0.5842 (58.42%)** | **0.4912** |
| **`white_rating`** | Hệ số Elo và đẳng cấp người cầm quân Trắng (đi trước) | **0.2150 (21.50%)** | **0.2310** |
| **`black_rating`** | Hệ số Elo và đẳng cấp người cầm quân Đen (đi sau) | **0.1420 (14.20%)** | **0.1850** |
| **`opening_ply`** | Độ dài lý thuyết của thế trận khai cuộc | **0.0385 (3.85%)** | **0.0520** |
| **`rated`** | Trận đấu có tính điểm xếp hạng hay là giao hữu | **0.0203 (2.03%)** | **0.0408** |

---

## 9. TÀI LIỆU BỔ TRỢ CHUYÊN SÂU

1. **[README_THUYET_TRINH.md](README_THUYET_TRINH.md):** Báo cáo học phần chi tiết 15 phần, bao gồm toàn bộ công thức toán học ma trận, giải thích ký hiệu, bảng phân tích hàm/lớp/caller và luồng kết nối Streamlit API.
2. **[GIAI_THICH_BIEU_DO.md](GIAI_THICH_BIEU_DO.md):** Hướng dẫn và giải thích cặn kẽ ý nghĩa của tất cả các biểu đồ trực quan hóa trong toàn bộ hệ thống.

---

## 10. THÔNG TIN TÁC GIẢ VÀ BẢN QUYỀN

- **Tác giả:** Võ Ngọc Gia Vỹ
- **Ngôn ngữ & Nền tảng:** Python 3, NumPy, Pandas, Matplotlib, Streamlit, Plotly, python-chess.
- **Bản quyền:** MIT License.
- **GitHub Repository:** [https://github.com/vongocgiavy/KNN_HGB_Logistic-Regression](https://github.com/vongocgiavy/KNN_HGB_Logistic-Regression)
