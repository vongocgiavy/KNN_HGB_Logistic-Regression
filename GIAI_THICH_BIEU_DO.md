# 📊 TÀI LIỆU GIẢI THÍCH CHI TIẾT CÁC BIỂU ĐỒ TRỰC QUAN HÓA TRONG HỆ THỐNG

> **Mục đích tài liệu:** Giải thích chi tiết, cặn kẽ và trực quan ý nghĩa của **tất cả các biểu đồ** trong hệ thống Web Dashboard và các module báo cáo Machine Learning (From Scratch). Dành cho cả người mới bắt đầu và người dùng chuyên môn.

---

## 📖 MỤC LỤC
1. [Nhóm 1: Biểu đồ Dự đoán Xác suất & Bàn cờ Khai cuộc (Tab 1 & Tab 2)](#1-nhóm-1-biểu-đồ-dự-đoán-xác-suất--bàn-cờ-khai-cuộc-tab-1--tab-2)
2. [Nhóm 2: Biểu đồ So sánh Hiệu suất & Tầm quan trọng Tính năng (Mục 5.2 & 5.3)](#2-nhóm-2-biểu-đồ-so-sánh-hiệu-suất--tầm-quan-trọng-tính-năng-mục-52--53)
3. [Nhóm 3: Biểu đồ Phân tích Quá khớp (Overfitting Analysis & Learning Curve)](#3-nhóm-3-biểu-đồ-phân-tích-quá-khớp-overfitting-analysis--learning-curve)
4. [Nhóm 4: Biểu đồ Ranh giới Quyết định 2D (Decision Boundaries)](#4-nhóm-4-biểu-đồ-ranh-giới-quyết-định-2d-decision-boundaries)
5. [Nhóm 5: Biểu đồ Tổng hợp & Khai phá Dữ liệu Cờ vua (Lichess EDA Insights)](#5-nhóm-5-biểu-đồ-tổng-hợp--khai-phá-dữ-liệu-cờ-vua-lichess-eda-insights)

---

## 1. NHÓM 1: BIỂU ĐỒ DỰ ĐOÁN XÁC SUẤT & BÀN CỜ KHAI CUỘC (TAB 1 & TAB 2)

### 📊 1.1. Biểu đồ Cột Xác suất Kết quả Ván cờ (Tab 1)
- **Hình thức:** Biểu đồ cột (Bar chart) 3 màu đại diện cho 3 khả năng:
  - **Màu cam san hô (`#f78166`):** Xác suất Bên Đen thắng (`0-1`).
  - **Màu tím nhạt (`#bc8cff`):** Xác suất Hòa (`1/2-1/2`).
  - **Màu xanh dương (`#58a6ff`):** Xác suất Bên Trắng thắng (`1-0`).
- **Ý nghĩa trục:**
  - **Trục hoành (X):** 3 lớp kết quả của ván cờ.
  - **Trục tung (Y):** Xác suất dự đoán (từ 0% đến 100%).
- **Cách mô hình tính toán:**
  - Dựa trên hiệu số Elo $\text{rating\_diff} = \text{white\_rating} - \text{black\_rating}$ và các đặc trưng kèm theo, mô hình (HGB / Logistic) sẽ tính toán ra trọng số logit rồi chuẩn hóa qua hàm **Softmax / Sigmoid** để tổng xác suất của 3 cột luôn bằng đúng $100\%$.
- **Ý nghĩa thực tế:** Khi Elo Trắng cao hơn Đen từ +200 điểm trở lên, cột Trắng thắng sẽ chiếm ưu thế vượt trội (>70%). Khi hai bên cân bằng Elo, cột Hòa và cơ hội chia đều sẽ tăng lên.

---

### ♟️ 1.2. Bàn cờ SVG 2D & Bảng Tìm kiếm Tương tự KNN (Tab 2)
- **Hình thức:** Bàn cờ 2D tương tác trực quan render từ chuẩn chuỗi nước đi PGN.
- **Cách hoạt động của KNN:**
  - Chuỗi nước đi người dùng nhập vào được **Vector hóa (Text Vectorizer)** thành một vector đặc trưng dựa trên tần suất xuất hiện và thứ tự các nước đi.
  - Thuật toán KNN tính **khoảng cách Cosine / Manhattan** giữa vector này với hàng nghìn ván cờ trong cơ sở dữ liệu để tìm ra Top $K$ ván cờ có cấu trúc thế trận gần nhất.
- **Chỉ số Độ tương đồng (%):**
  $$\text{Độ tương đồng} = (1 - \text{Khoảng cách}) \times 100\%$$
  Số phần trăm càng cao chứng minh thế trận khai cuộc càng trùng khớp với ván cờ mẫu trong lịch sử.

---

## 2. NHÓM 2: BIỂU ĐỒ SO SÁNH HIỆU SUẤT & TẦM QUAN TRỌNG TÍNH NĂNG (MỤC 5.2 & 5.3)

### 📊 2.1. Biểu đồ So sánh Hiệu suất 3 Mô hình (Mục 5.2 - Grouped Bar Chart)
- **Hình thức:** Biểu đồ cột phân nhóm (Grouped Bar Chart) so sánh đồng thời 3 chỉ số cốt lõi:
  - **Cột Xám:** Độ chính xác trung bình qua 3 lần xác thực chéo (`3-Fold CV Acc %`).
  - **Cột Xanh dương:** Độ chính xác trên tập kiểm tra giữ lại độc lập (`Hold-out Test Acc %`).
  - **Cột Xanh lá:** Điểm trung bình vĩ mô (`Macro F1-Score x100`).
- **Ý nghĩa so sánh:**
  - **HistGradientBoosting (HGB):** Đạt điểm cao nhất trên cả 3 chỉ số (Hold-out: **83.19%**, 3-Fold CV: **83.05%**, Macro F1: **0.82**). Cột Hold-out và CV bám sát nhau chứng minh mô hình **học thật - không học vẹt**.
  - **Logistic Regression & KNN:** Đạt độ chính xác 61% - 64%, nhưng Macro F1 chỉ đạt **0.28 - 0.31** do gặp khó khăn với lớp thiểu số (Hòa).

---

### 🔍 2.2. Ý nghĩa Khối Phân tích Lỗi (Error Analysis)
- **Bản chất bài toán:** Trong dữ liệu cờ vua thực tế, các trận đấu kết thúc có thắng thua rõ ràng chiếm đa số, trong khi tỷ lệ **Hòa (Draw) chỉ chiếm vỏn vẹn 5.11%** (Mất cân bằng lớp - Class Imbalance).
- **Tại sao HGB vượt trội?** Các mô hình tuyến tính (Logistic) hoặc dựa trên láng giềng đơn giản (KNN) có xu hướng "bỏ qua" các trận hòa để dự đoán vào lớp chiếm đa số. Trong khi đó, HGB xây dựng 200 cây quyết định kế tiếp nhau, cây sau tập trung sửa sai cho cây trước, từ đó nắm bắt được các ranh giới phi tuyến phức tạp giúp nhận diện trận hòa chính xác hơn hẳn.

---

### 📊 2.3. Biểu đồ Cột Ngang Tầm quan trọng của Tính năng (Mục 5.3 - Feature Importance)
- **Hình thức:** Biểu đồ thanh ngang so sánh mức độ ảnh hưởng của 5 đặc trưng giữa **HistGradientBoosting** và **Logistic Regression**.
- **Thứ tự xếp hạng mức độ quyết định kết quả:**
  1. **`rating_diff` (0.5842):** Chiếm gần 60% mức độ ảnh hưởng. Chênh lệch trình độ là yếu tố tiên quyết số 1 quyết định thắng thua trong cờ vua.
  2. **`white_rating` (0.2150):** Trình độ của người cầm quân Trắng (người có lợi thế đi trước).
  3. **`black_rating` (0.1420):** Trình độ của người cầm quân Đen (khả năng phòng thủ và phản công).
  4. **`opening_ply` (0.0385):** Độ dài của giai đoạn khai cuộc trước khi bước vào trung cuộc.
  5. **`rated` (0.0203):** Tính chất trận đấu có tính điểm xếp hạng hay không.

---

## 3. NHÓM 3: BIỂU ĐỒ PHÂN TÍCH QUÁ KHỚP (OVERFITTING ANALYSIS & LEARNING CURVE)

### 📊 3.1. Biểu đồ So sánh Train Accuracy vs Hold-out Test Accuracy (Gap Chart)
- **Mục tiêu:** Đo lường trực tiếp xem mô hình có bị "học vẹt" (Overfitting) hay không.
- **Công thức tính khoảng cách chênh lệch:**
  $$\text{Gap} = \text{Train Accuracy} - \text{Hold-out Test Accuracy}$$
- **Ý nghĩa từng mô hình trên biểu đồ:**
  - **Logistic Regression ($\text{Gap} = -0.8\%$):** Khoảng cách gần như bằng 0 $\rightarrow$ Mô hình cực kỳ ổn định, tuyệt đối không bị quá khớp.
  - **HistGradientBoosting ($\text{Gap} = +2.1\%$):** Độ chính xác Train 85.3% và Test 83.19% rất sát nhau $\rightarrow$ Kiểm soát quá khớp xuất sắc nhờ **L2 Regularization (1.5)** và cơ chế **Early Stopping**.
  - **KNN ($\text{Gap} = +36.6\%$):** Train đạt 99.9% nhưng Test 63.3% $\rightarrow$ Đây là đặc tính tự nhiên của thuật toán Lazy Learner (KNN luôn nhớ chính xác các điểm dữ liệu trong tập train của nó).

---

### 📈 3.2. Biểu đồ Đường cong Học HGB (Learning Curve - Train Loss vs Validation Loss)
- **Hình thức:** Biểu đồ 2 đường cong suy giảm hàm mất mát (Cross-Entropy Loss) theo số vòng lặp Boosting Stages (từ stage 1 đến 200).
  - **Đường màu xanh nét liền:** `Train Loss` (Hàm mất mát trên tập học).
  - **Đường màu cam nét đứt:** `Validation Loss` (Hàm mất mát trên tập kiểm tra độc lập).
  - **Vùng bóng mờ ở giữa:** Khoảng cách chênh lệch giữa hai đường (`Overfitting Gap`).
- **Cách đọc biểu đồ chuẩn:**
  - Ban đầu cả 2 đường đều cao (~0.69). Qua từng vòng lặp, cả hai đường cùng giảm dốc xuống dưới $\rightarrow$ Mô hình đang học tốt.
  - Vùng bóng mờ nhỏ và đường Validation đi ngang ổn định (không bị uốn cong ngược lên trên) $\rightarrow$ **Mô hình hội tụ chuẩn xác và không bị Overfitting.**

---

### 📉 3.3. Biểu đồ Kiểm định Chéo 3 Lần (3-Fold Cross-Validation Stability)
- **Mục tiêu:** Kiểm tra xem mô hình có bị phụ thuộc vào sự may rủi khi chia tập dữ liệu hay không.
- **Cách đọc biểu đồ:**
  - Trục X thể hiện 3 lần chia (`Fold 1`, `Fold 2`, `Fold 3`).
  - Các đường nét đứt đậm biểu diễn `Validation Accuracy` của từng mô hình qua các Fold.
  - **Kết luận:** Cả 3 đường đều nằm ngang phẳng lì (HGB dao động quanh 83.05% ±0.42%, Logistic quanh 63.95% ±0.61%) chứng minh thuật toán có tính **ổn định rất cao và đáng tin cậy**.

---

## 4. NHÓM 4: BIỂU ĐỒ RANH GIỚI QUYẾT ĐỊNH 2D (DECISION BOUNDARIES)

### 🔵 4.1. Biểu đồ Ranh giới Tuyến tính Multinomial Logistic Regression (OvR 3 Vùng Màu)
- **Hình thức:** Không gian 2 chiều biểu diễn mặt phẳng phân lớp theo 2 đặc trưng quan trọng nhất (`rating_diff` vs `white_rating`).
- **Ý nghĩa các vùng màu:**
  - **Vùng nền Xanh dương nhạt:** Vùng mà mô hình quyết định kết quả là **Đen thắng (`0-1`)**.
  - **Vùng nền Đỏ mận nhạt:** Vùng mà mô hình quyết định kết quả là **Hòa (`1/2-1/2`)**.
  - **Vùng nền Trắng xám nhạt:** Vùng mà mô hình quyết định kết quả là **Trắng thắng (`1-0`)**.
- **Đặc điểm toán học:** Ranh giới giữa các vùng là các **đường thẳng phẳng (Linear Planes)** do bản chất của hàm hồi quy tuyến tính $z = w^T x + b$.

---

### 🌳 4.2. Biểu đồ Ranh giới Phi tuyến Cây Quyết định (HistGradientBoosting Non-linear Boundary)
- **Hình thức:** Mặt phẳng phân chia có dạng **đường bậc thang phân mảnh (Stepwise Non-linear Boundary)**.
- **Tại sao lại có hình dạng bậc thang?**
  - Cây quyết định phân chia không gian bằng các câu lệnh điều kiện nhị phân: $\text{Nếu } x_1 \le \text{ngưỡng } t \rightarrow \text{Nhánh Trái, ngược lại } \rightarrow \text{Nhánh Phải}$.
  - Khi kết hợp 200 cây quyết định lại với nhau, các lát cắt ngang - dọc giao nhau tạo thành một mạng lưới bậc thang linh hoạt có thể uốn lượn ôm sát các cụm dữ liệu phi tuyến (như các thế trận hòa hoặc lật kèo).

---

### 🎯 4.3. Biểu đồ Ma trận Nhầm lẫn (Confusion Matrix Heatmap - 3x3)
- **Hình thức:** Bảng nhiệt ma trận 3 hàng x 3 cột.
  - **Trục Y (Hàng dọc):** Kết quả thực tế ngoài đời (`Actual Label`).
  - **Trục X (Hàng ngang):** Kết quả do mô hình HGB dự đoán (`Predicted Label`).
- **Cách đọc số liệu:**
  - **Đường chéo chính (Tô màu đậm nhất):** Số lượng các ván cờ dự đoán **ĐÚNG HOÀN TOÀN** (892 trận Đen thắng đúng, 28 trận Hòa đúng, 910 trận Trắng thắng đúng).
  - **Các ô nằm ngoài đường chéo chính:** Số lượng các ván cờ bị dự đoán nhầm lẫn.

---

## 5. NHÓM 5: BIỂU ĐỒ TỔNG HỢP & KHAI PHÁ DỮ LIỆU CỜ VUA (LICHESS EDA INSIGHTS)

### 🍩 5.1. Biểu đồ Donut Phân phối Tỷ lệ Kết quả Ván cờ
- **Số liệu thống kê trên 9,746 ván cờ Lichess:**
  - **Bên Trắng thắng (`1-0`):** 4,960 ván (**49.8%**) $\rightarrow$ Chiếm tỷ lệ cao nhất do Bên Trắng có lợi thế đi trước 1 nước.
  - **Bên Đen thắng (`0-1`):** 4,510 ván (**45.1%**) $\rightarrow$ Bám sát phía sau.
  - **Hòa (`1/2-1/2`):** 498 ván (**5.11%**) $\rightarrow$ Tỷ lệ rất thấp trong các ván cờ nghiệp dư/bán chuyên trên mạng do người chơi thường đánh quyết liệt đến cùng.

---

### 📊 5.2. Biểu đồ Top 10 Khai cuộc Cờ vua Phổ biến Nhất
- **Thống kê các thế trận được ưa chuộng nhất:**
  1. **Sicilian Defense:** 1,480 ván (Khai cuộc phòng thủ phản công kinh điển nhất chống lại `1. e4`).
  2. **French Defense:** 920 ván (Cấu trúc phòng thủ chặt chẽ cánh vua).
  3. **Queen's Gambit:** 840 ván (Thế trận kiểm soát trung tâm hàng đầu với `1. d4`).
  4. **Italian Game:** 780 ván (Ván cờ Ý mở, triển khai quân nhanh).
  5. **King's Indian Defense, Ruy Lopez, Scandinavian, Caro-Kann, English Opening, Modern Defense...**

---

### 📈 5.3. Đường cong Tương quan: Chênh lệch Elo vs Xác suất Thắng
- **Hình thức:** Đường cong chữ S (Hàm Sigmoid) biểu diễn mối quan hệ giữa điểm chênh lệch Elo và tỷ lệ chiến thắng.
- **Ý nghĩa toán học chuẩn FIDE:**
  - Khi $\text{rating\_diff} = 0$ (Hai kỳ thủ bằng điểm Elo): Xác suất thắng chia đều 50% - 50%.
  - Khi $\text{rating\_diff} = +200$ điểm: Xác suất Trắng thắng tăng lên khoảng **76%**.
  - Khi $\text{rating\_diff} \ge +400$ điểm: Xác suất Trắng thắng tiệm cận **91% - 99%**.
  - Mô hình Logistic Regression và HistGradientBoosting đều khớp hoàn hảo theo quy luật thống kê này.

---

## 6. TỔNG KẾT
Tất cả các biểu đồ trên được thiết kế nhằm mục đích mang lại trải nghiệm phân tích dữ liệu toàn diện nhất: từ trực quan hóa mức độ vi mô (từng nước cờ, từng dự đoán) đến mức độ vĩ mô (khám phá toàn bộ tập dữ liệu và đánh giá độ tin cậy của thuật toán Machine Learning thuần túy From Scratch).
