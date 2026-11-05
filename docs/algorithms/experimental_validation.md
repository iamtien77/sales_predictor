# Kế Hoạch Kiểm Tra & Đánh Giá Mô Hình

Tài liệu này mô tả cách nhóm kiểm tra tính ổn định của các mô hình qua nhiều mốc thời gian khác nhau và phân tích các trường hợp mô hình có thể dự báo sai, giúp hiểu rõ hơn hoạt động của từng thuật toán trong bài lab.

---

## 1. Thiết kế Đánh giá Dự báo Tuần qua Nhiều Cửa sổ Thời gian (Rolling Temporal Validation)

Để kiểm chứng xem sự vượt trội của mô hình (đặc biệt là kiến trúc Two-Stage Hurdle) có tính ổn định theo thời gian hay chỉ là hiện tượng ngẫu nhiên trên một lát cắt dữ liệu, nhóm thiết kế giao thức kiểm tra qua 3 cửa sổ chuỗi thời gian liên tiếp:

### 1.1. Cấu trúc các cửa sổ kiểm tra (Expanding Window Split)
- **Nguyên tắc cửa sổ mở rộng:** Tập Train được mở rộng dần theo thời gian quá khứ, trong khi các cửa sổ Test là các khoảng thời gian kế tiếp không giao nhau.
- **Độc lập thời gian:** Mỗi fold thực hiện fit lại toàn bộ các bước tiền xử lý (Imputer, Scaler, Encoder) CHỈ trên tập Train tương ứng của fold đó, sau đó transform sang tập Test để loại bỏ triệt để rò rỉ dữ liệu.
- **Quy tắc tính toán đặc trưng trễ:** Các đặc trưng trễ (`last_week_sales`, `sales_lag_2_weeks`, `sales_lag_4_weeks`) và thống kê cuộn 4 tuần trong tập Test chỉ được sử dụng thông tin phát sinh trước thời điểm thứ Hai đầu tuần của từng tuần đích, mô phỏng đúng bài toán dự báo một tuần tiếp theo (Rolling One-week-ahead).

### 1.2. Mục tiêu so sánh đối chuẩn
- So sánh sai số RMSE và MAE trên đơn vị BRL giữa mô hình Hai giai đoạn Hurdle và Zero Baseline qua từng cửa sổ.
- Đánh giá tính ổn định của các chỉ số sai số của mô hình ứng viên so với các baseline chuẩn qua từng thời kỳ lịch sử.

---

## 2. Thiết kế Kế hoạch Kiểm chứng Cold-Start Proxy

### 2.1. Giao thức đánh giá qua nhiều giai đoạn
- Bài toán proxy ước lượng doanh thu 7 ngày kể từ giao dịch đầu tiên được kiểm chứng qua 3 cửa sổ thời gian khởi đầu độc lập.
- Các ứng viên tham gia đối chứng gồm:
  1. **Baseline Giá bán đơn vị (Price Baseline):** $\hat{s} = \text{first\_price}$.
  2. **Baseline Trung vị Danh mục (Category Median Baseline):** $\hat{s} = \text{median}(\text{category})$.
  3. **Cây Hồi quy Học trên Không gian Log (LogTree):** Cây quyết định học trên $\log(1 + \text{sales})$ từ thuộc tính sản phẩm và giá bán đầu tiên, sau đó chuyển ngược về BRL.
  4. **Cây Hồi quy Học Trực tiếp trên Doanh thu Gốc (RawRevenueTree):** Cây quyết định học trực tiếp trên thang đo tiền tệ BRL nhằm kiểm tra giả thuyết liệu việc biến đổi log có phải là nguyên nhân làm sai lệch phân phối hay không.

### 2.2. Phương pháp Phân tích Lỗi: Phân khúc Một đơn vị vs Nhiều đơn vị
Nhóm thiết kế kỹ thuật phân tích lỗi bóc tách (Error Decomposition) bằng cách chia tập dữ liệu thành hai phân khúc:
- **Phân khúc Một đơn vị (`one_unit`):** Các sản phẩm chỉ bán được đúng một đơn vị duy nhất trong vòng 7 ngày đầu.
- **Phân khúc Nhiều đơn vị (`multiple_units`):** Các sản phẩm phát sinh thêm giao dịch trong vòng 7 ngày đầu.

**Mục tiêu chẩn đoán:**
- Đo lường mức độ ảnh hưởng của cấu trúc biến mục tiêu (đã bao hàm giao dịch đầu tiên) lên sai số của Baseline Giá so với mô hình cây trên từng phân khúc sản phẩm.

---

## 3. Thiết kế Thí nghiệm Chẩn đoán Hiện tượng Nổ Số học của Linear Regression

### 3.1. Hiện tượng cần lưu ý khi làm bài
Mô hình hồi quy tuyến tính trong không gian logarit khi chuyển đổi ngược về thang tiền tệ BRL bằng hàm mũ ($\text{expm1}$) có nguy cơ bùng nổ số học do hiệu ứng đòn bẩy hàm mũ ở phần đuôi phân phối.

### 3.2. Thiết kế Thí nghiệm Đối chứng
Nhóm thử nghiệm với các yếu tố:
1. **Hệ số điều chuẩn Ridge ($\alpha_{\text{ridge}}$):** Khảo sát các mức $\alpha \in \{10^{-5}, 1.0, 100.0\}$ để kiểm tra xem liệu việc tăng độ phạt trọng số có hạn chế được giá trị log ngoại lai hay không.
2. **Quy mô tập huấn luyện:** Khảo sát trên tập mẫu con (20.000 dòng) và toàn bộ tập Train (hơn 655.000 dòng) để đánh giá ảnh hưởng của kích thước ma trận dữ liệu.
3. **Cơ chế Kẹp trần Thích ứng (Adaptive Clamping):** Thiết kế chốt chặn an toàn giới hạn dự báo log không vượt quá miền giá trị cực đại quan sát được trên tập Train:

$$
\hat{y}_{\text{clipped}} = \min(\hat{y}, y_{\max,\text{train}})
$$

### 3.3. Tiêu chí phân tích chẩn đoán
- Đo lường tỷ trọng của 1% số mẫu có sai số lớn nhất đóng góp vào tổng sai số bình phương (Top 1% SSE Share).
- Phân tích tương quan giữa số điều kiện của ma trận Gram (Condition Number), chuẩn trọng số $\|w\|_2$ và sai số BRL thực tế.
- Bài học rút ra: Cần thiết lập cơ chế kẹp biên an toàn (clamping/guardrails) khi chuyển đổi dự báo từ log sang hàm mũ để tránh số quá lớn làm sai lệch kết quả.
