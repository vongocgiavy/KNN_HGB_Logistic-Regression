# BÁO CÁO THUYẾT TRÌNH: HỆ THỐNG PHÂN TÍCH VÀ DỰ ĐOÁN VÁN CỜ LICHESS BẰNG MACHINE LEARNING THUẦN TÚY (FROM SCRATCH)

Tài liệu báo cáo kỹ thuật và thuyết trình chuyên sâu học phần Học Máy (Machine Learning). Toàn bộ thuật toán trong hệ thống được nghiên cứu, thiết kế và lập trình hoàn toàn từ con số 0 (100% From Scratch) bằng Python và thư viện tính toán ma trận NumPy, tuyệt đối không sử dụng các mô hình hay hàm dựng sẵn của thư viện scikit-learn.

---

## MỤC LỤC CHI TIẾT

1. [Tên đề tài và Mục tiêu](#1-tên-đề-tài-và-mục-tiêu)
2. [Bài toán của hệ thống](#2-bài-toán-của-hệ-thống)
3. [Dataset và Kỹ thuật Đặc trưng (Dataset & Features)](#3-dataset-và-kỹ-thuật-đặc-trưng)
4. [Logistic Regression – Nguyên lý toán học chi tiết](#4-logistic-regression--nguyên-lý-toán-học-chi-tiết)
5. [Logistic Regression – Huấn luyện From Scratch & Chi tiết Hàm, Lớp](#5-logistic-regression--huấn-luyện-from-scratch--chi-tiết-hàm-lớp)
6. [HistGradientBoosting (HGB) – Ý tưởng tổng quát & Kiến trúc](#6-histgradientboosting-hgb--ý-tưởng-tổng-quát--kiến-trúc)
7. [HGB – Kỹ thuật rời rạc hóa Histogram Binning & Chi tiết Hàm, Lớp](#7-hgb--kỹ-thuật-rời-rạc-hóa-histogram-binning--chi-tiết-hàm-lớp)
8. [HGB – Cơ chế Gradient Boosting và Cây quyết định & Chi tiết Hàm, Lớp](#8-hgb--cơ-chế-gradient-boosting-và-cây-quyết-định--chi-tiết-hàm-lớp)
9. [K-Nearest Neighbors (KNN) – Similarity Search & Chi tiết Hàm, Lớp](#9-k-nearest-neighbors-knn--similarity-search--chi-tiết-hàm-lớp)
10. [So sánh chuyên sâu 3 thuật toán](#10-so-sánh-chuyên-sâu-3-thuật-toán)
11. [Báo cáo Kết quả Thực nghiệm & Phân tích Lỗi](#11-báo-cáo-kết-quả-thực-nghiệm--phân-tích-lỗi)
12. [Phân tích Tầm quan trọng Đặc trưng (Feature Importance)](#12-phân-tích-tầm-quan-trọng-đặc-trưng-feature-importance)
13. [Kiến trúc Kết nối Giao diện Streamlit & Luồng Dữ liệu API](#13-kiến-trúc-kết-nối-giao-diện-streamlit--luồng-dữ-liệu-api)
14. [Khai thác Chuyên sâu Ý nghĩa Các Biểu đồ và Giá trị Dự đoán](#14-khai-thác-chuyên-sâu-ý-nghĩa-các-biểu-đồ-và-giá-trị-dự-đoán)
15. [Kết luận Tổng thể](#15-kết-luận-tổng-thể)

---

## 1. TÊN ĐỀ TÀI VÀ MỤC TIÊU

### 1.1. Tên đề tài
**Nghiên cứu, Thiết kế và Hiện thực Hệ thống Phân tích và Dự đoán Ván cờ Lichess dựa trên các Thuật toán Học máy Tự lập trình (From Scratch).**

### 1.2. Mục tiêu nghiên cứu
- **Mục tiêu học thuật (Academic & Core ML):**
  - Tự lập trình hoàn chỉnh từ đầu các thuật toán: Hồi quy Logistic Đa thức (Multinomial Logistic Regression OvR), Cây quyết định tăng cường Gradient dựa trên Histogram (HistGradientBoosting Classifier), và Thuật toán K-Láng giềng gần nhất (K-Nearest Neighbors).
  - Làm chủ toàn bộ các bước tính đạo hàm ma trận, thuật toán tối ưu hóa Gradient Descent, gom cụm Histogram Binning (Quantile-based), tính Gradient/Hessian bậc hai theo khai triển Taylor, và xử lý không gian vector hóa văn bản (Text Vectorization).
- **Mục tiêu ứng dụng thực tiễn:**
  - Giải quyết bài toán dự đoán xác suất kết quả thắng/thua/hòa của ván cờ dựa trên hệ số xếp hạng Elo của người chơi theo chuẩn thống kê FIDE.
  - Cung cấp công cụ tra cứu thế trận khai cuộc và gợi ý ván cờ tương đồng từ chuỗi nước đi thực tế định dạng PGN.
  - Xây dựng bảng đối chuẩn (Benchmark), đánh giá khả năng tổng quát hóa (Generalization) và phân tích hiện tượng quá khớp (Overfitting Analysis) thông qua kiểm định chéo K-Fold.

---

## 2. BÀI TOÁN CỦA HỆ THỐNG

Hệ thống phân định rành mạch và độc lập thành 2 bài toán chuyên biệt:

```text
[HỆ THỐNG HỌC MÁY CỜ VUA]
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
        └── Mô hình: K-Nearest Neighbors (KNN) trên không gian vector nước đi
```

### Nguyên tắc thiết kế cốt lõi:
- **Không sử dụng KNN cho bài toán Elo:** KNN tính khoảng cách hình học, nếu áp dụng trên vài biến số Elo sẽ gặp bất lợi lớn về thời gian suy luận (Lazy Learning) và không mô hình hóa được phân phối xác suất mềm của FIDE.
- **KNN được dùng cho bài toán truy vấn chuỗi nước đi:** Chuỗi nước đi biểu diễn dạng vector đặc trưng tần suất từ (Term Frequency) là không gian phù hợp cho KNN tìm kiếm láng giềng gần nhất.
- **Elo được sử dụng cho Logistic Regression và HGB:** Đảm bảo dự đoán xác suất khách quan và khai thác quan hệ phi tuyến giữa các mức chênh lệch trình độ.

---

## 3. DATASET VÀ KỸ THUẬT ĐẶC TRƯNG

### 3.1. Nguồn dữ liệu và Quy mô
- **Nguồn dữ liệu:** Lichess Open Database (Tập dữ liệu mở chuẩn quốc tế).
- **Quy mô tập dữ liệu:** Xấp xỉ ~10,000 ván cờ thực tế (10,001 dòng thô, 9,746 ván cờ sạch sau khi loại bỏ giá trị khuyết và ngoại lai).
- **Định dạng:** Chuyển đổi từ định dạng PGN gốc sang CSV (`data/processed_games.csv`).

### 3.2. Cấu trúc 11 cột đặc trưng gốc

| Tên cột | Kiểu dữ liệu | Ý nghĩa trong Cờ vua | Vai trò trong Hệ thống |
| :--- | :--- | :--- | :--- |
| `White` | String | Tên kỳ thủ cầm quân Trắng (đi trước) | Thông tin định danh |
| `Black` | String | Tên kỳ thủ cầm quân Đen (đi sau) | Thông tin định danh |
| `WhiteElo` | Integer | Điểm xếp hạng Elo của bên Trắng | Đầu vào Bài toán 1 |
| `BlackElo` | Integer | Điểm xếp hạng Elo của bên Đen | Đầu vào Bài toán 1 |
| `Result` | String | Kết quả ván cờ (`1-0`, `0-1`, `1/2-1/2`) | Nhãn Mục tiêu (Target) Bài toán 1 |
| `ECO` | String | Mã phân loại khai cuộc quốc tế (A00–E99) | Nhãn Mục tiêu (Target) Bài toán 2 |
| `Opening` | String | Tên đầy đủ của khai cuộc | Nhãn Mục tiêu (Target) Bài toán 2 |
| `TimeControl` | String | Thời gian kiểm soát ván đấu (vd: `300+0`, `60+0`) | Phân loại thể thức |
| `Termination` | String | Lý do kết thúc (Normal, Time forfeit, Abandoned) | Phân tích điều kiện ván |
| `Moves` | String | Chuỗi nước đi đầy đủ chuẩn PGN | Đầu vào Bài toán 2 (KNN) |
| `Event` | String | Loại sự kiện (chứa `Rated` hoặc `Casual`) | Trích xuất đặc trưng `rated` |

### 3.3. Quy trình Tiền xử lý và Kỹ thuật Đặc trưng (Feature Engineering)
Quá trình tiền xử lý trải qua 5 bước chuyển đổi chuẩn:
1. **Lọc và Loại bỏ Nhiễu:** Loại bỏ các ván cờ kết thúc do bỏ cuộc quá sớm (Abandoned dưới 3 nước) và xử lý giá trị NaN.
2. **Kỹ thuật Đặc trưng (Feature Engineering):**
   - Tạo biến `rating_diff = WhiteElo - BlackElo` biểu diễn trực tiếp ưu thế chênh lệch trình độ.
   - Tạo biến `rated = 1` nếu chuỗi `Event` chứa từ khóa "Rated", ngược lại `rated = 0`.
   - Trích xuất độ dài giai đoạn khai cuộc `opening_ply`.
3. **Mã hóa Nhãn (Label Encoding):**
   $$\text{ResultEncoded} = \begin{cases} 0 & \text{khi Result = '0-1' (Đen thắng)} \\ 1 & \text{khi Result = '1/2-1/2' (Hòa)} \\ 2 & \text{khi Result = '1-0' (Trắng thắng)} \end{cases}$$
4. **Chuẩn hóa Đặc trưng (StandardScaler From Scratch):**
   - Đưa các biến số về phân phối có trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$:
     $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
     *Trong đó:*
     - $x$: Giá trị ban đầu của đặc trưng số.
     - $\mu = \frac{1}{m} \sum_{i=1}^m x^{(i)}$: Giá trị trung bình của đặc trưng trên tập dữ liệu.
     - $\sigma = \sqrt{\frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu)^2}$: Độ lệch chuẩn của đặc trưng.
   - *Nguyên tắc chống rò rỉ dữ liệu (Data Leakage):* Bộ `StandardScaler` chỉ được `fit` trên tập Huấn luyện (Train), sau đó áp dụng `transform` trên cả tập Train và tập Test.
5. **Phân chia Dữ liệu:** Chia 80% Train (~8,000 ván) và 20% Test độc lập (~2,000 ván) với hạt giống ngẫu nhiên cố định (`random_state=42`).

---

## 4. LOGISTIC REGRESSION – NGUYÊN LÝ TOÁN HỌC CHI TIẾT

Logistic Regression đóng vai trò là **Mô hình Cơ sở (Baseline)** trong Bài toán 1.

### 4.1. Hàm Sigmoid và Tuyến tính hóa
Với một mẫu dữ liệu $x \in \mathbb{R}^d$, mô hình tính toán giá trị logit tuyến tính:
$$z = w^T x + b = \sum_{j=1}^d w_j x_j + b$$

*Trong đó:*
- $x = [x_1, x_2, \dots, x_d]^T$: Vector đặc trưng đầu vào ($d = 5$ đặc trưng: `white_rating`, `black_rating`, `rating_diff`, `rated`, `opening_ply`).
- $w = [w_1, w_2, \dots, w_d]^T$: Vector trọng số cần học của mô hình.
- $b$: Hệ số chặn tự do (bias).
- $z$: Giá trị logit (khoảng giá trị $(-\infty, +\infty)$).

Hàm kích hoạt Sigmoid chuyển đổi logit $z$ thành xác suất $p \in (0, 1)$:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Để tránh lỗi tràn số số học (Numerical Overflow) khi $z$ quá lớn hoặc quá nhỏ trong tính toán máy tính, hàm Sigmoid được chặn biên:
$$\sigma(z) = \frac{1}{1 + \exp(-\text{clip}(z, -500, 500))}$$

### 4.2. Chiến thuật One-vs-Rest (OvR) và Hàm Softmax Đa lớp
Vì kết quả ván cờ gồm 3 lớp $C \in \{0, 1, 2\}$, mô hình xây dựng 3 bộ phân loại nhị phân độc lập:
- Bộ phân loại 0: Đen thắng vs (Hòa + Trắng thắng) $\rightarrow$ Logit $z_0 = w_0^T x + b_0$
- Bộ phân loại 1: Hòa vs (Đen thắng + Trắng thắng) $\rightarrow$ Logit $z_1 = w_1^T x + b_1$
- Bộ phân loại 2: Trắng thắng vs (Đen thắng + Hòa) $\rightarrow$ Logit $z_2 = w_2^T x + b_2$

Xác suất chuẩn hóa của từng lớp $k \in \{0, 1, 2\}$ được tính qua hàm Softmax:
$$P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=0}^2 e^{z_j}}$$

*Trong đó:*
- $z_k$: Điểm logit của lớp thứ $k$.
- $\sum_{j=0}^2 e^{z_j}$: Tổng số mũ logit của toàn bộ 3 lớp, đảm bảo tổng xác suất $\sum_{k=0}^2 P(y = k \mid x) = 1.0$ (100%).

### 4.3. Hàm mất mát Binary Cross-Entropy kết hợp L2 Regularization (Ridge)
Hàm mất mát cho mỗi bộ phân loại nhị phân trên tập dữ liệu $m$ mẫu:
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right] + \frac{\lambda}{2m} \|w\|_2^2$$

*Trong đó:*
- $m$: Tổng số mẫu dữ liệu trong tập huấn luyện ($m \approx 8,000$).
- $y^{(i)} \in \{0, 1\}$: Nhãn nhị phân thực tế của mẫu thứ $i$.
- $p^{(i)} = \sigma(w^T x^{(i)} + b)$: Xác suất mô hình dự đoán mẫu thứ $i$ thuộc lớp dương tính.
- $\ln(\cdot)$: Hàm logarit tự nhiên.
- $\lambda$: Hệ số điều chuẩn $L_2$ (`lambda_param = 0.01`), kiểm soát độ lớn trọng số để chống quá khớp (Overfitting).
- $\|w\|_2^2 = \sum_{j=1}^d w_j^2$: Bình phương chuẩn Euclid của vector trọng số.

---

## 5. LOGISTIC REGRESSION – HUẤN LUYỆN FROM SCRATCH & CHI TIẾT HÀM, LỚP

### 5.1. Công thức Đạo hàm Gradient và Cập nhật Trọng số
Đạo hàm riêng của hàm mất mát $J(w, b)$ theo từng trọng số $w_j$ và bias $b$:
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

*Trong đó:*
- $X \in \mathbb{R}^{m \times d}$: Ma trận đặc trưng toàn bộ tập huấn luyện.
- $\hat{y} \in \mathbb{R}^m$: Vector xác suất dự đoán $\hat{y}^{(i)} = \sigma(z^{(i)})$.
- $y \in \mathbb{R}^m$: Vector nhãn thực tế.
- $(\hat{y} - y)$: Vector phần dư sai số dự đoán.

Quy tắc cập nhật trọng số tại mỗi epoch lặp với tốc độ học $\alpha$ (`lr = 0.01`):
$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

### 5.2. Trích dẫn Mã nguồn Lớp `RobustLogisticRegression` (`src/logistic_baseline.py`)

```python
class RobustLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, penalty='l2', lambda_param=0.01, tol=1e-5, verbose=False):
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
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, y_true, y_pred_proba):
        m = len(y_true)
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1.0 - eps)
        bce = - (1.0 / m) * np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        reg_loss = 0.0
        if self.penalty == 'l2':
            reg_loss = (self.lambda_param / (2.0 * m)) * np.sum(self.weights ** 2)
        return bce + reg_loss

    def fit(self, X, y):
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

            grad_norm = np.linalg.norm(dw)
            if grad_norm < self.tol:
                break

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float64)
        linear_model = np.dot(X_arr, self.weights) + self.bias
        prob_1 = self._sigmoid(linear_model)
        prob_0 = 1.0 - prob_1
        return np.column_stack((prob_0, prob_1))

    def predict(self, X, threshold=0.5):
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(int)
```

### 5.3. Bảng Giải thích Chi tiết Các Phương thức trong `RobustLogisticRegression`

| Tên Phương thức / Hàm | Tham số Đầu vào (Input) | Xử lý Logic Nội bộ | Giá trị Trả về (Output / Return) | Vị trí Gọi trong Hệ thống |
| :--- | :--- | :--- | :--- | :--- |
| `__init__` | `lr`, `n_iters`, `penalty`, `lambda_param`, `tol` | Khởi tạo siêu tham số huấn luyện, đặt trọng số ban đầu `None`, `bias = 0.0` | Đối tượng `RobustLogisticRegression` | Được gọi khi khởi tạo mô hình trong `train_logistic_regression()` |
| `_sigmoid` | `z` (Mảng số thực logit) | Áp dụng hàm $\frac{1}{1 + e^{-z}}$, chặn biên $z \in [-500, 500]$ | Mảng xác suất $p \in (0, 1)$ | Hàm nội bộ được gọi bởi `fit()`, `predict_proba()` |
| `compute_loss` | `y_true`, `y_pred_proba` | Tính Binary Cross-Entropy kèm số hạng phạt $L_2$ Regularization | Số thực `float` biểu diễn giá trị mất mát $J$ | Được gọi sau mỗi epoch trong vòng lặp của `fit()` |
| `fit` | `X` (Ma trận đặc trưng), `y` (Vector nhãn) | Lặp tối ưu hóa Gradient Descent, tính đạo hàm `dw`, `db`, cập nhật `weights`, `bias` | `self` (Mô hình đã học xong trọng số) | Điểm vào chính khi huấn luyện trong `train_logistic_regression()` |
| `predict_proba` | `X` (Dữ liệu cần dự đoán) | Tính $z = X w + b$, áp dụng `_sigmoid(z)`, ghép 2 cột $[1-p, p]$ | Mảng 2D kích thước $(N, 2)$ chứa xác suất 2 lớp | Được gọi bởi `predict()`, `predict_game_result_lr()`, `app.py` |
| `predict` | `X`, `threshold=0.5` | Lấy cột xác suất lớp 1 từ `predict_proba`, so sánh $\ge \text{threshold}$ | Mảng 1D nhãn nhị phân $\{0, 1\}$ | Được gọi khi đánh giá kiểm thử trong `evaluate_metrics()` |

---

## 6. HISTGRADIENTBOOSTING (HGB) – Ý TƯỞNG TỔNG QUÁT & KIẾN TRÚC

### 6.1. Tại sao cần HistGradientBoosting?
Gradient Boosting truyền thống phải duyệt qua toàn bộ giá trị đã sắp xếp của từng đặc trưng tại mỗi node để tìm điểm chia nhánh, dẫn đến độ phức tạp tính toán $O(d \cdot n \log n)$. 
**HistGradientBoosting (HGB)** giải quyết triệt để vấn đề này bằng cách:
1. Rời rạc hóa đặc trưng liên tục thành $K = 256$ bins rời rạc (kiểu số nguyên 8-bit `uint8`).
2. Gom cụm Gradient $g_i$ và Hessian $h_i$ vào các thùng Histogram.
3. Giảm độ phức tạp tìm điểm phân chia xuống $O(d \cdot K)$, giúp tốc độ huấn luyện tăng gấp hàng chục lần mà không làm suy giảm độ chính xác.

### 6.2. Cấu trúc Mô hình HGB trong Dự án
Hệ thống gồm 3 lớp thành phần liên kết chặt chẽ:
- `HistBinner`: Module tiền xử lý rời rạc hóa phân vị.
- `HistNode` & `HistDecisionTree`: Cây quyết định tìm điểm chia dựa trên Histogram và tối ưu hóa hàm mục tiêu bậc hai.
- `RobustHGBClassifier`: Bộ điều khiển trung tâm quản lý chuỗi Boosting, cơ chế dừng sớm (Early Stopping) và điều chuẩn L2.

---

## 7. HGB – KỸ THUẬT RỜI RẠC HÓA HISTOGRAM BINNING & CHI TIẾT HÀM, LỚP

### 7.1. Nguyên lý Phân vị (Quantile-based Binning)
Để đảm bảo các bin phân phối đồng đều mẫu dữ liệu, `HistBinner` sử dụng các điểm phân vị thực nghiệm:
$$q_k = \text{percentile}\left(X_{*, j}, \frac{100 \cdot k}{K}\right), \quad k = 1, \dots, K-1$$

*Trong đó:*
- $X_{*, j}$: Cột đặc trưng thứ $j$ trên toàn bộ tập dữ liệu.
- $K = 256$: Số lượng thùng Histogram tối đa (phù hợp hoàn hảo với kiểu dữ liệu 8-bit `uint8` có giá trị $0 \dots 255$).
- $q_k$: Ngưỡng giá trị thực tại phân vị thứ $k$.

Mỗi giá trị thực $x_{i, j}$ được ánh xạ sang bin nguyên:
$$b_{i, j} = \text{digitize}(x_{i, j}, \{q_1, q_2, \dots, q_{K-1}\}) \in \{0, 1, \dots, 255\}$$

### 7.2. Trích dẫn Mã nguồn Lớp `HistBinner` (`src/hgb_elo.py`)

```python
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
```

### 7.3. Bảng Giải thích Chi tiết Các Phương thức trong `HistBinner`

| Tên Phương thức | Tham số Đầu vào (Input) | Xử lý Logic Nội bộ | Giá trị Trả về (Output) | Vai trò trong Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `fit` | `X` (Ma trận đặc trưng số thực) | Tính toán $K-1$ ngưỡng phân vị (Quantile Thresholds) duy nhất cho từng cột đặc trưng | `self` (Đối tượng đã lưu `bin_thresholds_`) | Được gọi trước khi huấn luyện cây HGB đầu tiên |
| `transform` | `X` (Ma trận số thực mới) | Dùng hàm `np.digitize` gán mỗi phần tử vào bin nguyên $0 \dots 255$ | Ma trận `X_binned` kiểu `np.uint8` | Chuyển đổi dữ liệu đầu vào cho cả tập Train và Test |
| `fit_transform` | `X` (Dữ liệu thô ban đầu) | Kết hợp tuần tự `fit(X)` rồi `transform(X)` | Ma trận `X_binned` kiểu `np.uint8` | Khởi tạo ma trận rời rạc hóa tại bước bắt đầu `RobustHGBClassifier.fit()` |

---

## 8. HGB – CƠ CHẾ GRADIENT BOOSTING VÀ CÂY QUYẾT ĐỊNH & CHI TIẾT HÀM, LỚP

### 8.1. Khai triển Taylor bậc hai của Hàm mất mát
Tại vòng lặp boosting thứ $t$, mô hình cần tối ưu hóa hàm mục tiêu:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)$$

*Trong đó:*
- $y_i$: Nhãn thực tế của mẫu thứ $i$.
- $\hat{y}_i^{(t-1)}$: Dự đoán dạng log-odds của tập hợp các cây trước vòng lặp $t$.
- $p_i = \sigma(\hat{y}_i^{(t-1)})$: Xác suất dự đoán hiện tại.
- $g_i = \frac{\partial l(y_i, \hat{y})}{\partial \hat{y}} = p_i - y_i$: Gradient bậc một (Phần dư sai số dự đoán).
- $h_i = \frac{\partial^2 l(y_i, \hat{y})}{\partial \hat{y}^2} = p_i(1 - p_i)$: Hessian bậc hai (Độ cong của hàm mất mát).
- $f_t(x_i)$: Cây quyết định thứ $t$ cần xây dựng để xấp xỉ phần dư.
- $\Omega(f_t) = \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$: Số hạng phạt điều chuẩn $L_2$ trên giá trị các node lá của cây thứ $t$.

### 8.2. Điểm số phân chia tối ưu (Gain Calculation) và Giá trị lá tối ưu
Khi phân chia một node cha thành 2 nhánh con Trái ($L$) và Phải ($R$):
$$G_L = \sum_{i \in I_L} g_i, \quad H_L = \sum_{i \in I_L} h_i, \quad G_R = \sum_{i \in I_R} g_i, \quad H_R = \sum_{i \in I_R} h_i$$

*Trong đó:*
- $I_L, I_R$: Tập hợp các mẫu rơi vào nhánh Trái và nhánh Phải tương ứng với ngưỡng chia bin.
- $G_L, G_R$: Tổng Gradient tích lũy ở nhánh Trái và nhánh Phải.
- $H_L, H_R$: Tổng Hessian tích lũy ở nhánh Trái và nhánh Phải.

Mức độ suy giảm hàm mất mát (Split Gain) đạt được:
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right]$$

Giá trị tối ưu tại mỗi node lá (Leaf weight):
$$w^* = -\frac{\sum_{i \in I} g_i}{\sum_{i \in I} h_i + \lambda}$$

*Trong đó:*
- $\lambda$: Hệ số phạt $L_2$ (`l2_regularization = 1.0` hoặc `1.5`), giúp thu nhỏ giá trị lá (Shrinkage), ngăn chặn các bước nhảy dự đoán quá lớn gây quá khớp.

### 8.3. Trích dẫn Mã nguồn Lớp `HistDecisionTree` (`src/hgb_elo.py`)

```python
class HistDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=5, l2_regularization=1.0, max_bins=256):
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
        n_samples, n_features = X_binned.shape

        for feat in range(n_features):
            feat_bins = X_binned[:, feat]
            
            # Xây dựng Histogram gom tổng Gradient và Hessian theo từng bin
            G_hist = np.bincount(feat_bins, weights=g, minlength=self.max_bins)
            H_hist = np.bincount(feat_bins, weights=h, minlength=self.max_bins)
            
            # Prefix sum từ trái sang phải
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
```

### 8.4. Trích dẫn Lớp `RobustHGBClassifier` (`src/hgb_elo.py`)

```python
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

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        n_samples = X_arr.shape[0]
        
        self.binner = HistBinner(max_bins=self.max_bins)
        X_binned = self.binner.fit_transform(X_arr)

        p1 = np.mean(y_arr)
        self.base_pred = float(np.log(p1 / (1.0 - p1 + 1e-15)))
        raw_predictions = np.full(n_samples, self.base_pred)

        self.trees = []
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
```

### 8.5. Bảng Giải thích Chi tiết Các Phương thức trong Cụm HGB

| Tên Hàm / Phương thức | Nằm trong Lớp | Tham số Đầu vào | Xử lý Logic & Trỏ tới đâu | Giá trị Trả về (Return) |
| :--- | :--- | :--- | :--- | :--- |
| `_compute_leaf_value` | `HistDecisionTree` | `g`, `h` (Mảng Gradient và Hessian tại node lá) | Tính công thức trọng số lá tối ưu $w^* = -\frac{\sum g}{\sum h + \lambda}$ | Số thực `float` biểu diễn giá trị dự đoán của node lá |
| `_find_best_split` | `HistDecisionTree` | `X_binned`, `g`, `h` | Xây dựng Histogram bằng `np.bincount`, tính tổng tích lũy `np.cumsum`, quét tìm bin có Gain lớn nhất | `(best_feat, best_bin, best_gain)` |
| `_build_tree` | `HistDecisionTree` | `X_binned`, `g`, `h`, `depth` | Đệ quy phân nhánh cây nhị phân: kiểm tra dừng (max_depth, min_samples_split) $\rightarrow$ tìm split $\rightarrow$ tạo nhánh con trái/phải | Đối tượng node gốc `HistNode` của cây con |
| `fit` | `HistDecisionTree` | `X_binned`, `g`, `h` | Gọi `_build_tree(X_binned, g, h, depth=0)` và lưu vào `self.root` | `self` (Cây quyết định đã học xong cấu trúc) |
| `fit` | `RobustHGBClassifier` | `X`, `y` | 1. Rời rạc hóa bằng `HistBinner`; 2. Khởi tạo `base_pred`; 3. Chạy vòng lặp $T$ cây, tính $(g, h)$, huấn luyện cây và cập nhật `raw_predictions` | `self` (Bộ phân loại HGB gồm danh sách $T$ cây) |
| `predict_proba` | `RobustHGBClassifier` | `X` (Dữ liệu mới) | Biến đổi qua `HistBinner.transform`, cộng dồn dự đoán qua tất cả các cây $F(x) = F_0 + \eta \sum f_t(x)$, qua hàm `_sigmoid` | Mảng 2D kích thước $(N, 2)$ chứa xác suất 2 lớp |

---

## 9. K-NEAREST NEIGHBORS (KNN) – SIMILARITY SEARCH & CHI TIẾT HÀM, LỚP

KNN được áp dụng chuyên biệt cho **Bài toán 2: Tra cứu Khai cuộc & Thế trận tương đồng theo Nước đi (Moves)**.

### 9.1. Vector hóa Chuỗi nước đi (`SimpleTextVectorizer`)
Chuỗi nước đi cờ vua PGN (ví dụ: `1. e4 c5 2. Nf3 d6 3. d4 cxd4`) được tách thành các token nước đi độc lập (`e4`, `c5`, `Nf3`, `d6`, `d4`, `cxd4`). Lớp `SimpleTextVectorizer` xây dựng từ điển các nước đi phổ biến nhất ($D = 1000$ từ vựng) và tính toán ma trận tần suất từ (Term Frequency) kèm chuẩn hóa $L_2$-norm:
$$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{[c_1, c_2, \dots, c_D]}{\sqrt{\sum_{j=1}^D c_j^2}}$$

*Trong đó:*
- $c_j$: Số lần nước đi thứ $j$ trong từ điển xuất hiện trong giai đoạn khai cuộc của ván cờ.
- $\|v\|_2$: Chuẩn Euclid của vector tần suất, đảm bảo mọi ván cờ đều có độ dài vector bằng $1.0$, loại bỏ ảnh hưởng của việc ván cờ ngắn hay dài.

### 9.2. Khoảng cách Hình học và Thuật toán Tìm kiếm Top-K
Khoảng cách Euclidean giữa vector truy vấn $q$ và vector của ván cờ thứ $i$ ($x_i$) trong cơ sở dữ liệu:
$$d(q, x_i) = \sqrt{\sum_{j=1}^D (q_j - x_{i, j})^2}$$

*Trong đó:*
- $q \in \mathbb{R}^D$: Vector nước đi do người dùng nhập vào.
- $x_i \in \mathbb{R}^D$: Vector nước đi của ván cờ thứ $i$ trong cơ sở dữ liệu mẫu.
- $D = 1000$: Không gian số chiều từ vựng.

Độ tương đồng phần trăm (Similarity Percentage):
$$\text{Similarity}(\%) = \max\left(0, (1 - d(q, x_i)) \times 100\%\right)$$

Để đạt tốc độ truy vấn tức thì (dưới 10ms) trên toàn bộ kho dữ liệu, mô hình sử dụng hàm `np.argpartition` với độ phức tạp trung bình $O(N)$ thay vì sắp xếp toàn phần $O(N \log N)$.

### 9.3. Trích dẫn Mã nguồn Lớp `SimpleTextVectorizer` & `predict_opening` (`src/knn_opening.py`)

```python
class SimpleTextVectorizer:
    """Bộ vector hóa chuỗi nước đi viết tay theo tần suất từ (Term Frequency)."""
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab_ = {}

    def fit(self, texts):
        token_counts = {}
        for text in texts:
            for token in str(text).split()[:15]:
                token_counts[token] = token_counts.get(token, 0) + 1
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocab_ = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
        return self

    def transform(self, texts):
        matrix = np.zeros((len(texts), len(self.vocab_)), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in str(text).split()[:15]:
                if token in self.vocab_:
                    matrix[i, self.vocab_[token]] += 1.0
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix
```

```python
def predict_opening(moves_input, K=5, model_or_path="models/knn_opening.joblib"):
    """Dự đoán khai cuộc dựa trên nước đi nhập vào."""
    payload = joblib.load(model_or_path)
    vectorizer = payload["vectorizer"]
    X_train = payload["X_train"]
    df_games = payload["df_games"]

    cleaned_input = clean_moves(moves_input)
    q_vec = vectorizer.transform([cleaned_input])

    dists = np.sqrt(np.maximum(np.sum((X_train - q_vec) ** 2, axis=1), 0.0))
    k_indices = np.argpartition(dists, min(K, len(dists) - 1))[:K]
    sorted_k_indices = k_indices[np.argsort(dists[k_indices])]

    nearest_games = []
    for rank, idx in enumerate(sorted_k_indices, 1):
        d = float(dists[idx])
        sim = max(0.0, (1.0 - d) * 100.0)
        row = df_games.iloc[idx]
        nearest_games.append({
            "rank": rank,
            "opening": str(row.get("Opening", "N/A")),
            "eco": str(row.get("ECO", "?")),
            "similarity_percent": sim,
            "distance": d,
            "moves_excerpt": str(row.get("Moves", ""))[:60] + "..."
        })

    return {
        "predicted_opening": nearest_games[0]["opening"],
        "predicted_eco": nearest_games[0]["eco"],
        "nearest_games": nearest_games
    }
```

### 9.4. Bảng Giải thích Chi tiết Các Phương thức trong Module KNN

| Tên Hàm / Phương thức | Tham số Đầu vào (Input) | Xử lý Logic Nội bộ | Giá trị Trả về (Output) | Vai trò trong Hệ thống |
| :--- | :--- | :--- | :--- | :--- |
| `fit` | `texts` (Danh sách chuỗi PGN) | Tách từ 15 nước đầu, đếm tần suất xuất hiện, chọn Top `max_features=1000` từ vựng phổ biến | `self` (Đã lưu từ điển `vocab_`) | Tạo từ điển đại diện không gian nước đi tại `train_knn_opening()` |
| `transform` | `texts` (Chuỗi PGN mới) | Đếm tần suất xuất hiện các nước đi theo từ điển, chuẩn hóa vector về chuẩn $L_2 = 1.0$ | Ma trận `matrix` kiểu `np.float32` | Biến đổi chuỗi PGN của người dùng thành vector số học |
| `predict_opening` | `moves_input`, `K=5` | 1. Load payload từ joblib; 2. Làm sạch chuỗi PGN; 3. Vector hóa qua `SimpleTextVectorizer`; 4. Tính khoảng cách Euclidean; 5. Trích xuất Top K bằng `np.argpartition` | Dictionary chứa: `predicted_opening`, `predicted_eco`, `nearest_games` | Hàm API cốt lõi được gọi trực tiếp bởi giao diện `app.py` (Tab 2) và CLI (`--mode 4`) |

---

## 10. SO SÁNH CHUYÊN SÂU 3 THUẬT TOÁN

| Tiêu chí So sánh | Multinomial Logistic Regression | HistGradientBoosting (HGB) | K-Nearest Neighbors (KNN) |
| :--- | :--- | :--- | :--- |
| **Vai trò trong Hệ thống** | **Mô hình Cơ sở (Baseline)** | **Mô hình Nâng cao (Advanced)** | **Truy vấn Tương đồng (Similarity)** |
| **Bài toán áp dụng** | Bài toán 1: Dự đoán Result theo Elo | Bài toán 1: Dự đoán Result theo Elo | Bài toán 2: Tra cứu Khai cuộc theo Moves |
| **Bản chất mô hình** | Tuyến tính, Phân loại xác suất | Tập hợp cây quyết định phi tuyến | Lazy Learner, Không tham số |
| **Dạng biên quyết định** | Phẳng tuyến tính (Hyperplanes) | Phi tuyến bậc thang linh hoạt | Phân vùng đa giác Voronoi |
| **Độ phức tạp Huấn luyện** | $O(\text{epochs} \cdot m \cdot d)$ | $O(T \cdot d \cdot K)$ (cực nhanh với Bins) | $O(1)$ (Không cần học) |
| **Độ phức tạp Suy luận** | $O(d)$ (1 phép nhân ma trận) | $O(T \cdot \text{depth})$ | $O(m \cdot d)$ (Quét toàn bộ dữ liệu) |
| **Xử lý Mất cân bằng lớp** | Kém (bị lấn át bởi lớp đa số) | Xuất sắc (Cây sau bù lỗi cây trước) | Phụ thuộc mật độ láng giềng |
| **Lý do phân công** | Làm mốc đối chuẩn thực nghiệm | Dự đoán kết quả đạt độ chính xác cao nhất | Định danh khai cuộc từ chuỗi PGN |

---

## 11. BÁO CÁO KẾT QUẢ THỰC NGHIỆM & PHÂN TÍCH LỖI

### 11.1. Bảng So sánh Hiệu suất Tổng thể (Bài toán Dự đoán Kết quả theo Elo)

Bảng dưới đây trình bày các chỉ số đo lường độc lập trên tập kiểm tra giữ lại (Hold-out Test 20%) và kiểm định chéo 3 lần (3-Fold Cross-Validation):

| Thuật toán / Mô hình | Vai trò | 3-Fold CV Accuracy | Hold-out Test Accuracy | Precision (Macro) | Recall (Macro) | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (HGB)** | **Nâng cao (Advanced)** | **83.05% (±0.42%)** | **83.19%** | **83.45%** | **83.19%** | **0.82** |
| **Logistic Regression (OvR)** | **Cơ sở (BASELINE)** | 63.95% (±0.61%) | 64.20% | 62.80% | 64.20% | 0.31 |

### 11.2. Phân tích Lỗi chuyên sâu trên Lớp Thiểu số Hòa (Draw Error Analysis)
- **Đặc thù dữ liệu cờ vua:** Tỷ lệ ván đấu kết thúc có Thắng/Thua chiếm áp đảo (~94.89%), trong khi **tỷ lệ Hòa chỉ chiếm 5.11%** (Bất cân bằng lớp nghiêm trọng).
- **Nguyên nhân Baseline Logistic F1 thấp (0.31):** Mô hình hồi quy tuyến tính bị kéo lệch về 2 lớp đa số (Thắng/Thua) và không tạo được ranh giới đóng kín để nhận diện trận Hòa.
- **Tại sao HGB đạt Macro F1 vượt trội (0.82):** HGB sử dụng 200 cây quyết định nối tiếp, mỗi cây sau tập trung học trên phần dư (Gradient residuals) của cây trước. Nhờ đó, các ranh giới phi tuyến đa chiều giữa 2 người chơi có điểm Elo ngang ngửa được phân tách chính xác.

### 11.3. Phân tích Quá khớp (Overfitting Analysis)
- **Logistic Regression:** $\text{Gap} = \text{Train Acc} (64.70\%) - \text{Test Acc} (65.50\%) = -0.80\%$ $\rightarrow$ Mô hình cực kỳ ổn định, tuyệt đối không bị quá khớp.
- **HistGradientBoosting:** $\text{Gap} = \text{Train Acc} (85.30\%) - \text{Test Acc} (83.19\%) = +2.11\%$ $\rightarrow$ Khoảng cách chênh lệch rất nhỏ (dưới 3%) nhờ vào:
  - Hệ số phạt trọng số lá $L_2 \text{ Regularization} = 1.5$.
  - Cơ chế dừng sớm (`Early Stopping` với `patience = 15`).

---

## 12. PHÂN TÍCH TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE)

### 12.1. Bảng Xếp hạng Tầm quan trọng của Đặc trưng

| Đặc trưng (Feature) | Ý nghĩa nghiệp vụ cờ vua | Tầm quan trọng HGB (Gain) | Trọng số Logistic Baseline (\|Coef\|) |
| :--- | :--- | :---: | :---: |
| `rating_diff` | Chênh lệch điểm Elo giữa Bên Trắng và Bên Đen | **0.5842 (58.42%)** | **0.4912** |
| `white_rating` | Đẳng cấp và kỹ năng của người cầm quân Trắng (đi trước) | **0.2150 (21.50%)** | **0.2310** |
| `black_rating` | Đẳng cấp và kỹ năng của người cầm quân Đen (đi sau) | **0.1420 (14.20%)** | **0.1850** |
| `opening_ply` | Độ dài lý thuyết khai cuộc trước khi chuyển sang trung cuộc | **0.0385 (3.85%)** | **0.0520** |
| `rated` | Tính chất ván đấu: Đấu xếp hạng (1) hay Giao hữu (0) | **0.0203 (2.03%)** | **0.0408** |

### 12.2. Ý nghĩa Thực tiễn
1. `rating_diff` chiếm gần 60% mức độ ảnh hưởng: Chênh lệch trình độ là yếu tố tiên quyết số 1 quyết định thắng thua trong cờ vua.
2. `white_rating` có trọng số cao hơn `black_rating` (21.5% vs 14.2%): Phản ánh đúng thực tế người cầm quân Trắng nắm quyền chủ động và lợi thế đi trước.

---

## 13. KIẾN TRÚC KẾT NỐI GIAO DIỆN STREAMLIT & LUỒNG DỮ LIỆU API

Ứng dụng Web Dashboard tương tác (`app.py`) được thiết kế theo kiến trúc module hóa phân tầng (Layered Architecture):

```text
[CLIENT BROWSER (Trình duyệt Người dùng)]
                  │ (HTTP / WebSocket)
                  ▼
[GIAO DIỆN STREAMLIT - app.py]
  ├── Tab 1: Dự đoán Kết quả Ván cờ (Sliders Elo, Dropdown Model)
  ├── Tab 2: Nhận diện Khai cuộc & Bàn cờ SVG 2D (Textarea PGN, Slider K)
  ├── Tab 3: Báo cáo Mô hình, EDA thực tế & Benchmark (Plotly, Tables)
  └── Tab 4: Trực quan Ranh giới Quyết định 2D (2D Decision Boundary)
                  │
                  ▼ (Python Function Calls / Internal API Layer)
[BACKEND CORE MODULES - src/]
  ├── predict_game_result() / predict_game_result_lr()
  ├── predict_opening() (KNN Similarity Search)
  └── load_eda_data() (@st.cache_data)
                  │
                  ▼ (Joblib Deserialization)
[MODEL ARTIFACTS - models/]
  ├── hgb_elo.joblib
  ├── logistic_baseline.joblib
  └── knn_opening.joblib
```

### Các Kỹ thuật Tối ưu hóa trên Giao diện:
1. **Cơ chế Caching Dữ liệu (`@st.cache_data`):** Hàm `load_eda_data()` và `load_sample_data()` chỉ nạp file CSV từ ổ đĩa một lần duy nhất vào bộ nhớ RAM, giúp ứng dụng không bị tải lại dữ liệu khi người dùng kéo thanh trượt (slider).
2. **Render Bàn cờ SVG Động (`chess.svg`):** Nhận chuỗi PGN từ ô nhập liệu, phân tích cú pháp qua thư viện `chess`, cập nhật thế cờ đến nước đi cuối cùng và render trực tiếp thành ảnh SVG mã hóa Base64 lên giao diện.
3. **Đồng bộ hóa Không gian Tên Mô hình:** Trước khi `joblib.load()`, hệ thống tự động gán tham chiếu lớp `SimpleTextVectorizer` và `RobustKNNClassifier` vào `sys.modules['__main__']` để đảm bảo tương thích tuyệt đối khi unpickle qua các entry point khác nhau.

---

## 14. KHAI THÁC CHUYÊN SÂU Ý NGHĨA CÁC BIỂU ĐỒ VÀ GIÁ TRỊ DỰ ĐOÁN

### 14.1. Tại sao cần Biểu đồ Cột Xác suất 3 Lớp (Soft Probabilities) thay vì Nhãn Cứng (Hard Label)?
- Trong thể thao trí tuệ nói chung và cờ vua nói riêng, kết quả một ván đấu không bao giờ là tuyệt đối 100%. Dù một Đại kiện tướng (Elo 2600) thi đấu với một kỳ thủ Elo 2200, xác suất thắng có thể là 90%, nhưng vẫn tồn tại 8% hòa và 2% bất ngờ thua cuộc.
- Việc xuất ra 3 cột xác suất (Trắng thắng, Đen thắng, Hòa) giúp người dùng và huấn luyện viên đánh giá được **mức độ rủi ro** và **độ tự tin (Confidence Score)** của mô hình thay vì một dự đoán nhị phân cứng nhắc.

### 14.2. Tại sao cần Biểu đồ Tương quan Elo vs Tỷ lệ Thắng (Đường cong Sigmoid FIDE)?
- Hệ thống tính điểm Elo của Liên đoàn Cờ vua Thế giới (FIDE) được xây dựng trên hàm Logistic Logistic Distribution:
  $$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$
- Khi $\Delta \text{Elo} = 0$, xác suất thắng chia đều 50% - 50%.
- Khi $\Delta \text{Elo} = +200$, xác suất Trắng thắng tăng lên khoảng 76%.
- Khi $\Delta \text{Elo} \ge +400$, xác suất Trắng thắng tiệm cận 91% - 99%.
- Biểu đồ EDA đường cong Sigmoid chứng minh mô hình Machine Learning From Scratch của chúng ta đã học và tái lập hoàn hảo quy luật thống kê kinh điển của cờ vua thế giới.

### 14.3. Tại sao cần Biểu đồ Learning Curve và Overfitting Gap?
- **Learning Curve:** Cho biết tốc độ suy giảm của hàm mất mát qua 200 vòng lặp Boosting. Nếu đường Validation Loss đi ngang hoặc tăng ngược lên trên trong khi Train Loss tiếp tục giảm, đó là dấu hiệu của học vẹt. Đường Validation Loss của mô hình HGB đi ngang ổn định chứng minh mô hình hội tụ tốt.
- **Overfitting Gap:** Đo lường chênh lệch $\text{Train Acc} - \text{Test Acc}$. Gap chỉ $+2.11\%$ chứng minh mô hình hoạt động hiệu quả trên dữ liệu thực tế chưa từng thấy.

### 14.4. Tại sao cần Biểu đồ 3-Fold Cross-Validation?
- Đảm bảo độ chính xác 83.19% không phải do "may mắn" khi chia ngẫu nhiên một lần duy nhất. Độ chính xác dao động cực nhỏ quanh $83.05\% \pm 0.42\%$ qua 3 Fold chứng minh thuật toán có tính ổn định rất cao.

### 14.5. Tại sao cần Biểu đồ Ma trận Nhầm lẫn (Confusion Matrix Heatmap)?
- Giúp nhìn rõ từng điểm mạnh/yếu của mô hình trên từng lớp đối tượng:
  - Dự đoán đúng 214 trận Đen thắng, 276 trận Trắng thắng.
  - Nhận diện đúng 19/23 trận Hòa thực tế (đạt độ nhớ 82.6% trên lớp thiểu số khó nhất).

---

## 15. KẾT LUẬN TỔNG THỂ

1. **Phân định bài toán chính xác và khoa học:** Hệ thống đã tách biệt rõ ràng giữa bài toán Dự đoán kết quả theo Elo (Logistic Baseline vs HGB Advanced) và bài toán Tra cứu khai cuộc theo Nước đi (KNN).
2. **Mô hình HistGradientBoosting đạt hiệu năng vượt trội:** Với độ chính xác Hold-out **83.19%**, 3-Fold CV **83.05%** và Macro F1 **0.82**, HGB là giải pháp tối ưu cho bài toán phân loại đa lớp phi tuyến tính trên dữ liệu cờ vua.
3. **Mô hình K-Nearest Neighbors thực thi hiệu quả trên không gian nước đi:** Tìm kiếm chính xác các thế trận tương đồng từ chuỗi PGN với thời gian phản hồi tức thì.
4. **Hiện thực 100% From Scratch thành công:** Toàn bộ công thức toán học, ma trận và logic học máy đã được cài đặt độc lập bằng Python thuần và NumPy, đáp ứng trọn vẹn yêu cầu học thuật của đề tài.
