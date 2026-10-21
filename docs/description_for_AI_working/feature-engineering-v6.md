Bản v6. Kỹ Nghệ Biến Hành Vi Khách Hàng & Độ Phủ Địa Lý
Mục tiêu của bản này là nắm bắt thói quen chi tiêu và mức độ lan tỏa thị trường của nhóm khách hàng mua sản phẩm.

Bước 1. Kỹ nghệ biến chi tiêu quá khứ trung bình (customer_past_spend)
Tính tổng số tiền mà nhóm khách hàng đã mua sản phẩm ở tuần gần nhất đã từng chi tiêu trên sàn Olist trước tuần mua đó.
Đại diện cho phân khúc khách hàng: khách hàng có sức mua lớn (High-end) hay khách hàng phổ thông nhạy cảm về giá.

Bước 2. Kỹ nghệ biến tần suất mua hàng tích lũy (customer_past_orders)
Tính số lượng đơn hàng trung bình mà nhóm khách mua gần nhất đã thực hiện trong quá khứ.
Đo lường mức độ trung thành của tệp khách hàng tiếp cận sản phẩm.

Bước 3. Kỹ nghệ biến độ phủ địa lý (customer_state_count)
Đếm số lượng bang khác nhau tại Brazil mà sản phẩm đã tiếp cận được trong tuần giao dịch gần nhất.
Sản phẩm tiếp cận được nhiều bang (ví dụ: SP, RJ, MG, RS) chứng tỏ có mạng lưới phân phối rộng và sức hút thị trường cao.

Bước 4. Xử lý giá trị mặc định cho sản phẩm chưa có giao dịch
Đối với các tuần chưa có giao dịch gần nhất, gán giá trị 0.0 cho chi tiêu, số đơn và số bang.
Pipeline sẽ tự động áp dụng Imputer và Scaler đồng nhất trên tập dữ liệu.

Bước 5. Kiểm tra tương quan với doanh số
Xác nhận rằng customer_past_spend có tương quan dương với quy mô đơn hàng, hỗ trợ mô hình phân biệt giữa các giao dịch giá trị lớn và giao dịch thông thường.
