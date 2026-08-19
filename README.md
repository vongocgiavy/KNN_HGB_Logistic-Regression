# ♟️ HỆ THỐNG PHÂN TÍCH & DỰ ĐOÁN VÁN CỜ LICHESS BẰNG MACHINE LEARNING (FROM SCRATCH)

> **Dự án Học Máy (Machine Learning) thuần túy:** Toàn bộ thuật toán được **tự lập trình từ đầu bằng Python và thư viện toán học NumPy (100% From Scratch)**, hoàn toàn **không sử dụng** bất kỳ hàm hay mô hình dựng sẵn nào từ thư viện `scikit-learn`.

---

## 📖 MỤC LỤC
1. [Giới thiệu Dự án & Phân định Bài toán](#1-giới-thiệu-dự-án--phân-định-bài-toán)
2. [Từ điển Thuật ngữ cho Người mới bắt đầu](#2-từ-điển-thuật-ngữ-cho-người-mới-bắt-đầu)
3. [Phân công Thuật toán & Mô hình](#3-phân-công-thuật-toán--mô-hình)
4. [Sơ đồ Cấu trúc File Dự án](#4-sơ-đồ-cấu-trúc-file-dự-án)
5. [Hướng dẫn Cài đặt & Chạy chương trình](#5-hướng-dẫn-cài-đặt--chạy-chương-trình)
6. [Báo cáo Kết quả Thực nghiệm (Mục 5.2 & 5.3)](#6-báo-cáo-kết-quả-thực-nghiệm-mục-52--53)
7. [Tài liệu Giải thích Chi tiết Tất cả Biểu đồ (GIAI_THICH_BIEU_DO.md)](GIAI_THICH_BIEU_DO.md)

---

## 1. GIỚI THIỆU DỰ ÁN & PHÂN ĐỊNH BÀI TOÁN

Dự án này giải quyết bài toán phân tích và dự đoán dữ liệu ván cờ thực tế từ nền tảng **Lichess (Open Database)** với hàng nghìn trận đấu. Hệ thống phân định **rõ ràng thành 2 bài toán độc lập**:

### 🎯 Bài toán 1: Dự đoán Kết quả Ván cờ dựa trên Elo (Match Result Prediction)
- **Đặc trưng đầu vào (Features):** Điểm Elo người chơi (`white_rating`, `black_rating`), chênh lệch Elo (`rating_diff`), tính chất xếp hạng (`rated`), độ dài khai cuộc (`opening_ply`).
- **Mục tiêu (Target):** Dự đoán kết quả trận đấu (`1-0` Trắng thắng, `0-1` Đen thắng, `1/2-1/2` Hòa).
- **Mô hình sử dụng:**
  - **Logistic Regression (Multinomial OvR):** Làm mô hình cơ sở (**BASELINE**).
  - **HistGradientBoosting (HGB):** Mô hình nâng cao (**ADVANCED MODEL**) nắm bắt quan hệ phi tuyến.

### ♟️ Bài toán 2: Tìm Ván cờ & Khai cuộc Tương tự dựa trên Nước đi (Move Similarity & Opening Retrieval)
- **Đặc trưng đầu vào (Features):** Chuỗi nước đi cờ vua (`Moves` / `CleanedMoves`).
- **Mục tiêu (Target):** Tìm Top-K ván cờ lịch sử có thế trận tương đồng nhất và định danh tên Khai cuộc cùng mã ECO tương ứng.
- **Mô hình sử dụng:** **K-Nearest Neighbors (KNN)** dựa trên không gian khoảng cách vector nước đi (`Text Vectorizer`).

> [!IMPORTANT]
> **LƯU Ý CỰC KỲ QUAN TRỌNG:**
> - **KHÔNG** dùng KNN cho Elo để dự đoán kết quả ván cờ.
> - **KNN** chỉ dùng để tìm ván cờ/Opening tương tự dựa trên Moves.
> - **Elo** dùng cho Logistic Regression (Baseline) và HGB trong bài toán dự đoán Result.
> - **KHÔNG** trộn 2 bài toán KNN (Opening search) và HGB/Logistic (Result prediction) thành 1 bài toán duy nhất.

---

## 2. TỪ ĐIỂN THUẬT NGỮ CHO NGƯỜI MỚI BẮT ĐẦU

### ♟️ Thuật ngữ Cờ vua (Chess Domain):
- **Elo Rating (Điểm Elo):** Thước đo chuẩn quốc tế về trình độ của kỳ thủ. Điểm càng cao thì người chơi càng giỏi (ví dụ: người mới bắt đầu ~1000, kiện tướng ~2000+).
- **`white_rating` & `black_rating`:** Điểm Elo của người cầm quân Trắng và người cầm quân Đen.
- **`rating_diff` (Chênh lệch Elo):** Hiệu số: $\text{Elo Trắng} - \text{Elo Đen}$. Nếu số này dương lớn, Bên Trắng có trình độ vượt trội; nếu âm lớn, Bên Đen vượt trội.
- **Opening (Khai cuộc):** Giai đoạn khởi đầu ván cờ (khoảng 5-15 nước đầu tiên) nhằm triển khai quân và kiểm soát trung tâm (ví dụ: *Sicilian Defense, Ruy Lopez, French Defense*).
- **ECO Code (Mã phân loại khai cuộc):** Bảng mã chuẩn quốc tế gồm 1 chữ cái và 2 chữ số (từ A00 đến E99) dùng để định danh mọi thế cờ khai cuộc (ví dụ: `B20` là Sicilian Defense, `C50` là Italian Game).
- **`opening_ply`:** Số lượng nước đi lý thuyết thuộc sách giáo khoa khai cuộc trước khi người chơi tự đi theo tính toán cá nhân.
- **`rated`:** Ván cờ có tính điểm xếp hạng vào hệ thống Elo (1) hay chỉ là trận đấu giao hữu thử nghiệm (0).

---

### 🤖 Thuật ngữ Học máy (Machine Learning):
- **From Scratch (Viết tay thuần túy):** Tự mình lập trình toàn bộ công thức toán học, ma trận và thuật toán từ con số 0 bằng Python và NumPy, tuyệt đối không dùng thư viện ngoài làm hộ.
- **Data Leakage (Rò rỉ dữ liệu):** Sai lầm nghiêm trọng khi để tập dữ liệu kiểm thử (Test) bị lẫn vào quá trình học của tập huấn luyện (Train), khiến mô hình "học vẹt/gian lận".
- **StandardScaler (Chuẩn hóa dữ liệu):** Phép biến đổi đưa các đặc trưng về cùng thang đo chuẩn có giá trị trung bình $\text{Mean} = 0$ và độ lệch chuẩn $\text{Std} = 1$.
- **One-vs-Rest (OvR - Một đối Tất cả):** Chiến thuật giải quyết bài toán phân loại đa lớp (3 lớp: Thắng, Hòa, Thua). Mô hình sẽ tạo ra 3 bộ phân loại nhị phân riêng biệt và chọn lớp có xác suất cao nhất.
- **Histogram Binning (Gom cụm biểu đồ tần suất):** Kỹ thuật gom các giá trị số thực liên tục vào 256 thùng nguyên (0-255). Giúp tìm điểm chia nhánh cây với tốc độ siêu nhanh.
- **Cross-Validation (Xác thực chéo $K$-Fold):** Chia tập dữ liệu thành $K$ phần bằng nhau, lần lượt dùng 1 phần để kiểm tra và $K-1$ phần để học, xoay vòng $K$ lần để lấy điểm trung bình khách quan.
- **Hold-out Test (Tập kiểm tra giữ lại):** Tập dữ liệu (chiếm 20%) được cất riêng ra từ đầu và không bao giờ được chạm vào trong lúc huấn luyện.

---

## 3. PHÂN CÔNG THUẬT TOÁN & MÔ HÌNH

| Thuật toán | Vai trò trong Dự án | Bài toán áp dụng | Lý do lựa chọn |
| :--- | :--- | :--- | :--- |
| **1. Hồi quy Logistic Đa thức (Logistic Regression - OvR)** | **Baseline Model (Mô hình Cơ sở)** | Dự đoán Kết quả ván cờ theo Elo | Làm cột mốc đối chuẩn (Baseline). Xác suất thắng thua trong cờ vua chịu ảnh hưởng lớn từ chênh lệch Elo theo hàm Sigmoid chuẩn thống kê FIDE. |
| **2. HistGradientBoosting (HGB)** | **Advanced Model (Mô hình Nâng cao)** | Dự đoán Kết quả ván cờ theo Elo | Đạt độ chính xác và hiệu năng cao nhất. Có khả năng nắm bắt các ranh giới phi tuyến phức tạp (đặc biệt là các trường hợp ván cờ Hòa - Draw vốn rất khó phân biệt đối với mô hình tuyến tính). |
| **3. K-Nearest Neighbors (KNN)** | **Similarity Search Model** | Tra cứu Ván cờ & Khai cuộc theo Moves | Hoàn hảo cho bài toán **Truy vấn Khai cuộc**. Bằng cách so sánh khoảng cách giữa chuỗi nước đi người dùng nhập vào với các ván cờ có sẵn, KNN nhanh chóng tìm ra các ván cờ tương đồng nhất. |

---

## 4. SƠ ĐỒ CẤU TRÚC FILE DỰ ÁN

```text
demo mh/
├── app.py                      # Ứng dụng Web Dashboard giao diện Streamlit (3 Tab chức năng)
├── requirements.txt            # Danh sách thư viện cần thiết (Không có scikit-learn)
├── .gitignore                  # Cấu hình bỏ qua các file dung lượng lớn khi đẩy lên Git
├── README.md                   # File tài liệu hướng dẫn tổng quan toàn dự án
├── GIAI_THICH_BIEU_DO.md       # Tài liệu giải thích chi tiết ý nghĩa tất cả các biểu đồ
│
├── src/                        # THƯ MỤC CHỨA MÃ NGUỒN THUẬT TOÁN (TỰ VIẾT TAY 100%)
│   ├── main.py                 # File điều khiển trung tâm: Quản lý 2 bài toán & Menu Console CLI
│   ├── logistic_baseline.py    # Thuật toán Hồi quy Logistic Đa thức (Baseline Model)
│   ├── hgb_elo.py              # Thuật toán Histogram Gradient Boosting Classifier (Advanced Model)
│   ├── knn_opening.py          # Thuật toán KNN tìm kiếm ván cờ & nhận diện khai cuộc cờ vua
│   ├── comparison.py           # Module tính toán bảng so sánh Baseline vs Advanced HGB (5.2 & 5.3)
│   ├── overfitting_analysis.py # Module phân tích quá khớp & Learning Curve trên dữ liệu Elo
│   ├── data_loader.py          # Module đọc và giải nén dữ liệu PGN/CSV từ Lichess
│   ├── preprocessing.py        # Module làm sạch nước đi và mã hóa nhãn cờ vua
│   └── eda_results.py          # Module phân tích thống kê trực quan dữ liệu EDA
│
├── data/                       # THƯ MỤC DỮ LIỆU CỜ VUA THỰC TẾ
│   ├── processed_games.csv     # Dữ liệu ván cờ đã tiền xử lý
│   └── filtered_processed_games.csv # Dữ liệu cờ vua đã lọc sạch và tạo sẵn đặc trưng
│
├── models/                     # THƯ MỤC LƯU TRỮ CÁC MÔ HÌNH ĐÃ HUẤN LUYỆN
│   ├── logistic_baseline.joblib# Trọng số mô hình Baseline Logistic Regression
│   ├── hgb_elo.joblib          # Cấu trúc 200 cây quyết định của HGB
│   └── knn_opening.joblib      # Chỉ mục tìm kiếm khai cuộc KNN
│
└── outputs/                    # THƯ MỤC CHỨA KẾT QUẢ VÀ BÁO CÁO XUẤT RA
    ├── logistic_metrics.json   # Chỉ số đánh giá của mô hình Baseline Logistic
    ├── hgb_metrics.json        # Chỉ số đánh giá của mô hình HGB
    ├── model_comparison.txt    # Báo cáo so sánh tổng hợp lưu dưới dạng văn bản
    └── overfitting_analysis.png# Biểu đồ xuất ra từ phân tích quá khớp
```

---

## 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY CHƯƠNG TRÌNH

### 🛠️ Bước 1: Cài đặt môi trường
Mở Terminal / PowerShell tại thư mục dự án và chạy:
```powershell
py -m pip install -r requirements.txt
```

---

### 🚀 Bước 2: Các lệnh thực thi chương trình

#### 1. Khởi chạy Giao diện Web tương tác (Streamlit Dashboard):
```powershell
py -m streamlit run app.py
```
> *(Sau khi chạy lệnh, trình duyệt sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`)*

#### 2. Chạy So sánh Hiệu suất Bài toán Dự đoán Resultados Elo (Mục 5.2):
```powershell
py src/main.py --mode 3
```
*(Chạy trên Terminal không hiện popup đồ thị:)*
```powershell
py src/main.py --mode 3 --no-plot
```

#### 3. Chạy từng chế độ thực nghiệm:
- **Chế độ 1: Mô hình Cơ sở - Hồi quy Logistic Đa thức (Baseline OvR):**
  ```powershell
  py src/main.py --mode 1
  ```
- **Chế độ 2: Mô hình Nâng cao - Histogram Gradient Boosting (HGB):**
  ```powershell
  py src/main.py --mode 2
  ```
- **Chế độ 4: Bài toán 2 - KNN Tra cứu Khai cuộc & Nước đi tương tự:**
  ```powershell
  py src/main.py --mode 4
  ```

#### 4. Chạy Menu tương tác qua bàn phím:
```powershell
py src/main.py
```

---

## 6. BÁO CÁO KẾT QUẢ THỰC NGHIỆM (MỤC 5.2 & 5.3)

### 5.2. So sánh hiệu suất mô hình dự đoán kết quả (Elo Features)

Bảng dưới đây trình bày các chỉ số hiệu suất toàn diện giữa mô hình cơ sở (**Baseline Logistic Regression**) và mô hình nâng cao (**HistGradientBoosting**):

| Thuật toán / Mô hình | Vai trò trong Dự án | 3-Fold CV Accuracy | Hold-out Test Accuracy | Precision | Recall | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. HistGradientBoosting (HGB)** | **Mô hình Nâng cao (Advanced)** | **83.05% (±0.42%)** | **83.19%** | **83.45%** | **83.19%** | **0.82** |
| **2. Hồi quy Logistic Đa thức (OvR)** | **Mô hình Cơ sở (BASELINE)** | 63.95% (±0.61%) | 64.20% | 62.80% | 64.20% | 0.31 |

> **Nhận xét tổng quan:** Bộ phân loại **Tăng cường Gradient Biểu đồ Histogram (HistGradientBoosting)** đạt hiệu suất vượt trội so với mô hình Baseline, với độ chính xác giữ lại **83.19%** so với **64.20%**. Sự đồng bộ chặt chẽ giữa điểm số giữ lại và điểm xác thực chéo chứng minh khả năng tổng quát hóa vững chắc mà không bị quá khớp (Overfitting).

#### 🔍 Phân tích lỗi (Error Analysis):
Mặc dù mô hình Gradient Boosting đạt độ chính xác tổng thể cao, phân tích hiệu suất theo từng lớp cho thấy phần lớn lỗi phân loại xảy ra trong hạng mục **'Draw' (Hòa)**. Do sự mất cân bằng lớp cao (chỉ **5.11%** số lần hòa trong thực tế), mô hình tuyến tính cơ sở *Logistic Regression Baseline* gặp khó khăn trong việc phân biệt các trận hòa với các trận đấu quyết định kéo dài (Macro F1 = **0.31**). Trái lại, *HistGradientBoosting (HGB)* với 200 cây quyết định học nối tiếp đã nắm bắt thành công động lực phi tuyến phức tạp liên quan đến các trận hòa (Macro F1 = **0.82**).

---

### 5.3. Phân tích tầm quan trọng của tính năng (Feature Importance)

Các giá trị phân tích tầm quan trọng của tính năng cho thấy các mẫu nhất quán giữa các thuật toán:

| Tính năng (Feature) | Ý nghĩa thực tế trong Cờ vua | Tầm quan trọng HGB | Độ lớn trọng số Logistic Baseline (\|Coef\|) |
| :--- | :--- | :---: | :---: |
| **`rating_diff`** | Chênh lệch Elo giữa Bên Trắng và Bên Đen (Yếu tố quyết định cao nhất) | **0.5842** | **0.4912** |
| **`white_rating`** | Hệ số Elo và đẳng cấp người cầm quân Trắng | **0.2150** | **0.2310** |
| **`black_rating`** | Hệ số Elo và đẳng cấp người cầm quân Đen | **0.1420** | **0.1850** |
| **`opening_ply`** | Độ dài lý thuyết của thế trận khai cuộc trước khi vào trung cuộc | **0.0385** | **0.0520** |
| **`rated`** | Trận đấu có tính điểm xếp hạng hay chỉ là trận đấu giao hữu | **0.0203** | **0.0408** |

---

## 7. THÔNG TIN REPOSITORY & TÁC GIẢ

- **GitHub Repository:** [https://github.com/vongocgiavy/KNN_HGB_Logistic-Regression](https://github.com/vongocgiavy/KNN_HGB_Logistic-Regression)
- **Tác giả:** Võ Ngọc Gia Vỹ
- **Ngôn ngữ & Thư viện:** Python 3, NumPy, Pandas, Matplotlib, Streamlit, Plotly, Python-Chess.
- **Giấy phép:** MIT License.
