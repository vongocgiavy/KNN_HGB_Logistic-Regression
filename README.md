# ♟️ HỆ THỐNG PHÂN TÍCH & DỰ ĐOÁN VÁN CỜ LICHESS BẰNG MACHINE LEARNING (FROM SCRATCH)

> **Dự án Học Máy (Machine Learning) thuần túy:** Toàn bộ thuật toán được **tự lập trình từ đầu bằng Python và thư viện toán học NumPy (100% From Scratch)**, hoàn toàn **không sử dụng** bất kỳ hàm hay mô hình dựng sẵn nào từ thư viện `scikit-learn`.

---

## 📖 MỤC LỤC
1. [Giới thiệu Dự án](#1-giới-thiệu-dự-án)
2. [Từ điển Thuật ngữ cho Người mới bắt đầu](#2-từ-điển-thuật-ngữ-cho-người-mới-bắt-đầu)
   - [Thuật ngữ Cờ vua (Chess Domain)](#thuật-ngữ-cờ-vua-chess-domain)
   - [Thuật ngữ Học máy (Machine Learning)](#thuật-ngữ-học-máy-machine-learning)
3. [Tại sao lại lựa chọn 3 Thuật toán này?](#3-tại-sao-lại-lựa-chọn-3-thuật-toán-này)
4. [Giải thích Chi tiết Tác dụng của Từng File trong Dự án](#4-giải-thích-chi-tiết-tác-dụng-của-từng-file-trong-dự-án)
5. [Hướng dẫn Cài đặt & Chạy chương trình](#5-hướng-dẫn-cài-đặt--chạy-chương-trình)
6. [Báo cáo Kết quả Thực nghiệm (Mục 5.2 & 5.3)](#6-báo-cáo-kết-quả-thực-nghiệm-mục-52--53)

---

## 1. GIỚI THIỆU DỰ ÁN

Dự án này giải quyết bài toán phân tích và dự đoán dữ liệu ván cờ thực tế từ nền tảng **Lichess (Open Database)** với hơn hàng nghìn trận đấu. Hệ thống tập trung vào 2 mục tiêu cốt lõi:
1. **Dự đoán kết quả ván cờ (Game Result Prediction):** Dựa trên hệ số trình độ (Elo) của hai người chơi và các thông số trước trận đấu, mô hình sẽ tính toán xác suất thắng của Bên Trắng (`1-0`), Bên Đen (`0-1`) hoặc Hòa (`1/2-1/2`).
2. **Truy vấn & Nhận diện Khai cuộc (Opening Retrieval):** Người dùng nhập vào một chuỗi nước đi bất kỳ (ví dụ: `1. e4 c5 2. Nf3 d6`), mô hình sẽ tìm kiếm các ván cờ có thế trận tương đồng nhất trong quá khứ và định danh tên thế trận khai cuộc cùng mã ECO tương ứng.

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
- **Data Leakage (Rò rỉ dữ liệu):** Sai lầm nghiêm trọng khi để tập dữ liệu kiểm thử (Test) bị lẫn vào quá trình học của tập huấn luyện (Train), khiến mô hình "học vẹt/gian lận" dẫn đến điểm số cao ảo nhưng khi gặp dữ liệu thực tế thì thất bại.
- **StandardScaler (Chuẩn hóa dữ liệu):** Phép biến đổi đưa các đặc trưng về cùng thang đo chuẩn có giá trị trung bình $\text{Mean} = 0$ và độ lệch chuẩn $\text{Std} = 1$. Việc này giúp khoảng cách giữa các số hàng nghìn (như Elo 2000) không lấn át các số nhỏ (như opening_ply = 8).
- **One-vs-Rest (OvR - Một đối Tất cả):** Chiến thuật giải quyết bài toán phân loại đa lớp (3 lớp: Thắng, Hòa, Thua). Mô hình sẽ tạo ra 3 bộ phân loại nhị phân riêng biệt: *Trắng thắng vs Phần còn lại*, *Hòa vs Phần còn lại*, *Đen thắng vs Phần còn lại*, sau đó chọn lớp có xác suất cao nhất.
- **Histogram Binning (Gom cụm biểu đồ tần suất):** Kỹ thuật gom hàng triệu giá trị số thực liên tục vào một số lượng cố định các thùng nguyên (thường là 256 thùng - từ 0 đến 255). Giúp máy tính duyệt và tìm điểm chia nhánh cây với tốc độ siêu nhanh.
- **Cross-Validation (Xác thực chéo $K$-Fold):** Chia tập dữ liệu thành $K$ phần bằng nhau, lần lượt dùng 1 phần để kiểm tra và $K-1$ phần để học, xoay vòng $K$ lần để lấy điểm trung bình. Giúp đánh giá mô hình khách quan và trung thực nhất.
- **Hold-out Test (Tập kiểm tra giữ lại):** Tập dữ liệu (chiếm 20%) được cất riêng ra từ đầu và không bao giờ được chạm vào trong lúc huấn luyện, chỉ đem ra chấm điểm cuối cùng.
- **Precision (Độ chính xác theo lớp):** Trong những lần mô hình dự đoán là lớp X, có bao nhiêu phần trăm là đúng thật sự?
- **Recall (Khả năng ghi nhớ / Độ nhạy):** Trong tất cả các mẫu thực sự là lớp X ngoài đời, mô hình đã tìm ra được bao nhiêu phần trăm?
- **F1-Score:** Điểm số trung bình điều hòa giữa Precision và Recall. Nếu một trong hai chỉ số quá tệ thì F1-Score sẽ thấp ngay, giúp đánh giá chất lượng toàn diện.

---

## 3. TẠI SAO LẠI LỰA CHỌN 3 THUẬT TOÁN NÀY?

Dự án lựa chọn 3 thuật toán tiêu biểu đại diện cho 3 trường phái tư duy phân loại khác nhau:

| Thuật toán | Trường phái đại diện | Tại sao lại sử dụng trong Cờ vua? |
| :--- | :--- | :--- |
| **1. Hồi quy Logistic Đa thức (Multinomial Logistic - OvR)** | **Mô hình Tuyến tính & Xác suất cơ sở (Baseline)** | Làm cột mốc đối chuẩn (Baseline). Xác suất thắng thua trong cờ vua chịu ảnh hưởng lớn từ chênh lệch Elo theo hàm Sigmoid chuẩn thống kê FIDE. |
| **2. K-Nearest Neighbors (KNN)** | **Học dựa trên thể hiện & khoảng cách (Instance-Based / Lazy Learner)** | Hoàn hảo cho bài toán **Truy vấn Khai cuộc**. Bằng cách so sánh khoảng cách giữa chuỗi nước đi người dùng nhập vào với các ván cờ có sẵn, KNN nhanh chóng tìm ra các ván cờ tương đồng nhất mà không cần giả định trước phân phối. |
| **3. Histogram Gradient Boosting (HGB)** | **Học kết hợp Cây quyết định tăng cường (Ensemble Boosting)** | Đạt độ chính xác và hiệu năng cao nhất. Có khả năng nắm bắt các ranh giới phi tuyến phức tạp (đặc biệt là các trường hợp ván cờ Hòa - Draw vốn rất khó phân biệt đối với các mô hình tuyến tính). |

---

## 4. GIẢI THÍCH CHI TIẾT TÁC DỤNG CỦA TỪNG FILE TRONG DỰ ÁN

```text
demo mh/
├── app.py                      # Ứng dụng Web Dashboard giao diện Streamlit
├── requirements.txt            # Danh sách các thư viện cần thiết (Không có scikit-learn)
├── .gitignore                  # Cấu hình bỏ qua các file dung lượng lớn khi đẩy lên Git
├── README.md                   # File tài liệu hướng dẫn tổng quan toàn dự án
│
├── src/                        # THƯ MỤC CHỨA MÃ NGUỒN THUẬT TOÁN (TỰ VIẾT TAY 100%)
│   ├── main.py                 # File điều khiển trung tâm: Tự chứa đủ 3 thuật toán và Menu Console
│   ├── logistic_baseline.py    # Thuật toán Hồi quy Logistic Đa thức viết tay
│   ├── knn_result.py           # Thuật toán KNN phân loại kết quả ván cờ viết tay
│   ├── knn_opening.py          # Thuật toán KNN tìm kiếm & nhận diện khai cuộc cờ vua
│   ├── hgb_elo.py              # Thuật toán Histogram Gradient Boosting Classifier viết tay
│   ├── comparison.py           # Module tính toán bảng so sánh hiệu năng 5.2 và tầm quan trọng 5.3
│   ├── data_loader.py          # Module đọc và giải nén dữ liệu PGN/CSV từ Lichess
│   ├── preprocessing.py        # Module làm sạch nước đi và mã hóa nhãn cờ vua
│   └── eda_results.py          # Module phân tích thống kê trực quan dữ liệu EDA
│
├── data/                       # THƯ MỤC DỮ LIỆU CỜ VUA THỰC TẾ
│   ├── processed_games.csv     # Dữ liệu ván cờ đã tiền xử lý
│   └── filtered_processed_games.csv # Dữ liệu cờ vua đã lọc sạch và tạo sẵn đặc trưng
│
├── models/                     # THƯ MỤC LƯU TRỮ CÁC MÔ HÌNH ĐÃ HUẤN LUYỆN
│   ├── logistic_model.joblib   # File lưu trọng số mô hình Logistic Regression
│   ├── hgb_elo.joblib          # File lưu cấu trúc 200 cây quyết định của HGB
│   └── knn_opening.joblib      # File lưu chỉ mục tìm kiếm khai cuộc KNN
│
└── outputs/                    # THƯ MỤC CHỨA KẾT QUẢ VÀ BÁO CÁO XUẤT RA
    ├── logistic_metrics.json   # Chỉ số đánh giá của mô hình Logistic
    ├── hgb_metrics.json        # Chỉ số đánh giá của mô hình HGB
    └── model_comparison.txt    # Báo cáo so sánh tổng hợp lưu dưới dạng văn bản
```

### 📄 Chi tiết tác dụng từng file:
- [`src/main.py`](file:///d:/May_Hoc/demo%20mh/src/main.py): File trung tâm độc lập cao nhất của dự án. Chứa toàn bộ các lớp `StandardScaler`, `GridSearchCV_Custom`, `MultinomialLogisticRegression_OvR`, `RobustKNNClassifier`, `RobustHGBClassifier` và hệ thống Menu điều khiển tương tác.
- [`app.py`](file:///d:/May_Hoc/demo%20mh/app.py): Ứng dụng Web trực quan hóa bằng Streamlit với 3 Tab chức năng: Dự đoán xác suất kết quả cờ vua, Bàn cờ 2D tương tác tìm kiếm khai cuộc, và Bảng so sánh Benchmark các mô hình.
- [`src/logistic_baseline.py`](file:///d:/May_Hoc/demo%20mh/src/logistic_baseline.py): Độc lập xây dựng thuật toán hồi quy Logistic với thuật toán hạ gradient (Gradient Descent), hàm kích hoạt Sigmoid chống tràn số và bộ lọc nhiễu L2 Regularization.
- [`src/hgb_elo.py`](file:///d:/May_Hoc/demo%20mh/src/hgb_elo.py): Xây dựng thuật toán gom cụm Histogram 256 thùng, cây quyết định phân tách theo Gain của Gradient/Hessian và cơ chế Boosting 200 vòng lặp.
- [`src/knn_opening.py`](file:///d:/May_Hoc/demo%20mh/src/knn_opening.py): Tự xây dựng bộ vector hóa chuỗi nước đi (Text Vectorizer) và thuật toán khoảng cách Manhattan/Euclidean để truy xuất Top $K$ ván cờ tương đồng nhất.
- [`src/comparison.py`](file:///d:/May_Hoc/demo%20mh/src/comparison.py): Tự động tổng hợp các kết quả thực nghiệm, tính toán chỉ số và xuất bảng báo cáo Markdown/Text phục vụ nghiên cứu và thuyết trình.
- [`src/data_loader.py`](file:///d:/May_Hoc/demo%20mh/src/data_loader.py) & [`src/preprocessing.py`](file:///d:/May_Hoc/demo%20mh/src/preprocessing.py): Thực hiện đọc dữ liệu thô, lọc bỏ nước đi lỗi, mã hóa kết quả ván đấu và tạo các trường đặc trưng như `rating_diff`.

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

#### 2. Chạy So sánh toàn diện cả 3 mô hình (Mục 5.2):
```powershell
py src/main.py --mode 4
```
*(Nếu muốn chạy nhanh trên Terminal và tắt cửa sổ popup đồ thị:)*
```powershell
py src/main.py --mode 4 --no-plot
```

#### 3. Chạy kiểm thử từng thuật toán đơn lẻ:
- **Hồi quy Logistic Đa thức (Multinomial Logistic - OvR):**
  ```powershell
  py src/main.py --mode 1
  ```
- **K-Nearest Neighbors (KNN: $k=20$, Manhattan, Trọng số khoảng cách):**
  ```powershell
  py src/main.py --mode 2
  ```
- **Histogram Gradient Boosting (HGB: $lr=0.1$, $depth=5$, $iter=200$):**
  ```powershell
  py src/main.py --mode 3
  ```
- **HGB kết hợp dò tìm siêu tham số tự động (`GridSearchCV`):**
  ```powershell
  py src/main.py --mode 3 --grid-search
  ```

#### 4. Chạy Menu tương tác lựa chọn qua bàn phím:
```powershell
py src/main.py
```

---

## 6. BÁO CÁO KẾT QUẢ THỰC NGHIỆM (MỤC 5.2 & 5.3)

### 5.2. So sánh hiệu suất mô hình

Bảng dưới đây trình bày các chỉ số hiệu suất toàn diện. **Độ chính xác (Accuracy)** được báo cáo cho cả tập kiểm tra giữ lại (**Hold-out Test**) và trung bình của **xác thực chéo 3 lần (3-Fold CV)**. Các chỉ số chi tiết (**Độ chính xác - Precision, Ghi nhớ - Recall, Điểm Macro F1**) được báo cáo trên bộ hold-out để đánh giá khả năng tổng quát hóa:

| Thuật toán / Mô hình | 3-Fold CV Accuracy | Hold-out Test Accuracy | Precision | Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. HistGradientBoosting (HGB)** | **83.05% (±0.42%)** | **83.19%** | **83.45%** | **83.19%** | **0.82** |
| **2. Hồi quy Logistic Đa thức (OvR)** | 63.95% (±0.61%) | 64.20% | 62.80% | 64.20% | 0.31 |
| **3. K-Nearest Neighbors (KNN)** | 60.80% (±0.78%) | 61.50% | 59.90% | 61.50% | 0.28 |

> **Nhận xét tổng quan:** Bộ phân loại **Tăng cường Gradient Biểu đồ Histogram (HistGradientBoosting)** đạt hiệu suất vượt trội trên tất cả các chỉ số, với độ chính xác giữ lại **83.19%** và kết quả xác thực chéo nhất quán. Đáng chú ý, sự đồng bộ chặt chẽ giữa điểm số giữ lại và điểm xác thực chéo trên tất cả các mô hình cho thấy sự tổng quát hóa vững chắc mà không bị quá khớp (Overfitting).

#### 🔍 Phân tích lỗi (Error Analysis):
Mặc dù mô hình Gradient Boosting đạt độ chính xác tổng thể cao, phân tích hiệu suất theo từng lớp cho thấy phần lớn lỗi phân loại xảy ra trong hạng mục **'Draw' (Hòa)**. Do sự mất cân bằng lớp cao (chỉ **5.11%** số lần hòa trong thực tế), các mô hình đơn giản hơn như *K-Nearest Neighbors* và *Logistic Regression* gặp khó khăn trong việc phân biệt các trận hòa với các trận đấu quyết định kéo dài, dẫn đến điểm F1 trung bình vĩ mô thấp hơn (**0.28** và **0.31** tương ứng) so với *Gradient Boosting* (**0.82**), vốn đã thành công trong việc nắm bắt động lực phi tuyến đặc thù liên quan đến các trận hòa.

---

### 5.3. Phân tích tầm quan trọng của tính năng (Feature Importance)

Các giá trị phân tích tầm quan trọng của tính năng cho thấy các mẫu nhất quán giữa các thuật toán:

| Tính năng (Feature) | Ý nghĩa thực tế trong Cờ vua | Tầm quan trọng HGB | Độ lớn trọng số Logistic (\|Coef\|) |
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
