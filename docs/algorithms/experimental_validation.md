# Kế Hoạch Kiểm Tra & Đánh Giá Mô Hình

Tài liệu đề xuất cách kiểm tra mô hình qua nhiều giai đoạn thời gian và tìm hiểu các trường hợp dự báo sai. Các thí nghiệm bổ sung phục vụ học tập, không giả định trước mô hình nào sẽ tốt hơn.

---

## 1. Thiết kế Đánh giá Dự báo Tuần qua Nhiều Cửa sổ Thời gian (Rolling Temporal Validation)

Nhóm dự kiến dùng ba cửa sổ thời gian để xem mức sai số có ổn định không và so sánh mô hình với baseline trên từng giai đoạn.

### 1.1. Cấu trúc các cửa sổ kiểm tra (Expanding Window Split)
- **Nguyên tắc cửa sổ mở rộng:** Tập Train được mở rộng dần theo thời gian quá khứ, trong khi các cửa sổ Test là các khoảng thời gian kế tiếp không giao nhau.
- **Độc lập thời gian:** Mỗi fold thực hiện học lại các tham số tiền xử lý chỉ trên tập Train tương ứng của fold đó, sau đó áp dụng cùng tham số cho tập Test để tránh nguy cơ rò rỉ dữ liệu.
- **Quy tắc tính toán đặc trưng trễ:** Các đặc trưng doanh thu trễ 1, 2, 4 tuần và thống kê trượt bốn tuần trong tập Test chỉ được sử dụng thông tin phát sinh trước thời điểm thứ Hai đầu tuần của từng tuần đích, mô phỏng đúng bài toán dự báo một tuần tiếp theo (Rolling One-week-ahead).

### 1.2. Mục tiêu so sánh qua ba cửa sổ

So sánh Hurdle với baseline Zero và Previous Week bằng MAE, RMSE và R² trên BRL. Chưa xác định phương pháp nào tốt hơn trước khi đánh giá.

| Cửa sổ | Train | Phần đánh giá | Nội dung cần ghi nhận |
| --- | --- | --- | --- |
| Thứ nhất | Giai đoạn lịch sử ban đầu | Giai đoạn tiếp theo | Sai số từng mô hình và số mẫu hợp lệ |
| Thứ hai | Mở rộng lịch sử đến trước giai đoạn mới | Giai đoạn sau cửa sổ thứ nhất | Thay đổi sai số theo thời gian |
| Thứ ba | Tiếp tục mở rộng lịch sử | Giai đoạn cuối được giữ lại | Mức ổn định của chênh lệch với baseline |

Các cửa sổ đánh giá không chồng lấn. Cấu hình phải được xác định trước hoặc chọn bằng phần Validation nằm trong lịch sử Train của từng cửa sổ; không dùng điểm đánh giá để điều chỉnh ngược lại.

---

## 2. Thiết kế Kế hoạch Kiểm chứng Cold-Start Proxy

### 2.1. Giao thức đánh giá qua nhiều giai đoạn

Proxy bảy ngày từ lần bán đầu dự kiến được so sánh qua ba cửa sổ theo ngày bắt đầu. Mỗi mẫu phải có đủ bảy ngày quan sát; mẫu Train có nhãn chồng lấn giai đoạn đánh giá cần được loại. Tham số mã hóa, điền thiếu và chuẩn hóa chỉ học từ Train tương ứng.

| Ứng viên | Cách dự báo | Câu hỏi cần kiểm tra |
| --- | --- | --- |
| Baseline một đơn vị | Giá giao dịch đầu đã biết | Mốc đơn giản này phù hợp đến đâu? |
| Baseline trung vị danh mục | Trung vị doanh thu danh mục trên Train | Thống kê danh mục có cung cấp mốc so sánh hữu ích không? |
| Cây học trên log | Học log doanh thu rồi chuyển về BRL | Sai số thay đổi thế nào sau phép nghịch đảo? |
| Cây học trực tiếp trên BRL | Học doanh thu gốc | Việc đổi thang mục tiêu có cải thiện sai số không? |

Đánh giá MAE và RMSE trên cùng mẫu cho mọi ứng viên. Đây là thí nghiệm đối chứng với cấu hình định trước, không chọn lại mô hình theo Test.

### 2.2. Phương pháp Phân tích Lỗi: Phân khúc Một đơn vị vs Nhiều đơn vị

Dự kiến chia các mẫu đánh giá thành hai nhóm để đọc sai số:
- **Chỉ bán một đơn vị:** target bằng giá giao dịch đầu đã biết, nên baseline một đơn vị có sai số bằng 0 theo định nghĩa bài toán. Đây là tính chất của target, không phải bằng chứng dự báo được sản phẩm chưa bán.
- **Bán nhiều đơn vị:** so sánh xem các cây có ước lượng được phần doanh thu phát sinh thêm tốt hơn baseline không.

Cần ghi số mẫu và tỷ lệ của mỗi nhóm khi phân tích. Không kết luận nguyên nhân chênh lệch toàn tập chỉ từ một chỉ số tổng hợp.

---

## 3. Thiết kế Thí nghiệm Chẩn đoán Hiện tượng Sai số Lớn khi Nghịch đảo Log của Linear Regression

### 3.1. Hiện tượng cần lưu ý khi làm bài
Mô hình hồi quy tuyến tính trong không gian logarit khi chuyển đổi ngược về thang tiền tệ BRL bằng hàm mũ ($\text{expm1}$) có nguy cơ phát sinh sai số rất lớn do hiệu ứng đòn bẩy hàm mũ ở phần đuôi phân phối.

### 3.2. Thiết kế Thí nghiệm Đối chứng
Nhóm dự kiến thử các yếu tố sau, giữ các điều kiện khác giống nhau:
1. **Hệ số điều chuẩn Ridge ($\alpha_{\text{ridge}}$):** Khảo sát các mức $\alpha \in \{10^{-5}, 1.0, 100.0\}$ để kiểm tra xem liệu việc tăng độ phạt trọng số có hạn chế được giá trị log ngoại lai hay không.
2. **Quy mô tập huấn luyện:** so sánh một mẫu con với toàn bộ Train, dùng cùng giai đoạn đánh giá để xem ảnh hưởng của lượng dữ liệu.
3. **Cơ chế Kẹp trần Thích ứng (Adaptive Clamping):** Thử nghiệm so sánh cơ chế kẹp trần an toàn giới hạn dự báo log không vượt quá miền giá trị cực đại quan sát được trên tập Train:

$$
\hat{y}_{\text{clipped}} = \min(\hat{y}, y_{\max,\text{train}})
$$

### 3.3. Tiêu chí phân tích và kế hoạch chẩn đoán

- So sánh sai số trên log và BRL để nhận biết ảnh hưởng của phép mũ.
- Theo dõi độ điều kiện của ma trận, miền dự báo log và tỷ trọng tổng sai số bình phương thuộc nhóm mẫu sai lớn nhất.
- Thay đổi từng yếu tố: hệ số Ridge, lượng dữ liệu hoặc biên chặn. Chọn các phương án cần so sánh trước khi xem Test.
- Chưa quy nguyên nhân cho suy biến ma trận, ngoại suy hoặc thiếu điều chuẩn nếu chưa có đối chứng. Nếu dùng thí nghiệm để thay đổi thiết kế, cần đánh giá lại trên giai đoạn chưa dùng để điều chỉnh.
