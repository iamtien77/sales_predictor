# Quy Chuẩn Dữ Liệu: Cold-Start Cho Sản Phẩm Mới Trước Lần Bán Đầu

Tài liệu này xác định các yêu cầu và đặc tả dữ liệu đầu vào cần thiết để thực hiện bài toán dự báo doanh thu cho sản phẩm mới trước thời điểm phát sinh giao dịch đầu tiên (Pre-launch Listing Cold-Start).

## 1. Bối cảnh & Giới hạn của Dữ liệu Hiện tại

Bộ dữ liệu Olist hiện tại không có thông tin về ngày niêm yết sản phẩm, giá niêm yết theo thời điểm, hay danh mục đầy đủ của các mặt hàng đã đưa lên sàn nhưng chưa từng bán được. Do không thể suy diễn các thông tin này từ thời điểm mua đầu tiên, nhóm xác định **chưa thể thực hiện kiểm chứng bài toán cold-start trước bán trên tập dữ liệu Olist gốc**.

Để có thể thử nghiệm bài toán này trong tương lai, cần bổ sung nguồn dữ liệu đáp ứng các quy chuẩn dưới đây.

## 2. Đặc tả Cấu trúc Dữ liệu Yêu cầu

### 2.1. Bảng Danh mục Sản phẩm
Mỗi dòng đại diện cho một sản phẩm duy nhất với khóa chính mã sản phẩm, bao gồm các trường:
- mã sản phẩm: Mã định danh của sản phẩm.
- ngày niêm yết: Ngày sản phẩm chính thức xuất hiện trên sàn.
- giá niêm yết ban đầu: Giá niêm yết ban đầu của sản phẩm khi ra mắt.
- thời điểm thuộc tính có sẵn: Mốc thời gian xác nhận các thông tin mô tả sản phẩm đã có sẵn trước hoặc đúng ngày niêm yết.
- danh mục sản phẩm: Ngành hàng của sản phẩm.
- Các thuộc tính mô tả: Khối lượng, số ảnh, độ dài tên và độ dài mô tả.

> **Yêu cầu quan trọng:** Bảng danh mục bắt buộc phải bao gồm cả những sản phẩm chưa từng phát sinh đơn hàng. Các thuộc tính mô tả phải là dữ liệu đã có sẵn tại thời điểm niêm yết, không lấy từ các cập nhật phát sinh sau đó để tránh rò rỉ thông tin tương lai.

### 2.2. Bảng Giao dịch
Ghi nhận toàn bộ các giao dịch mua hàng phát sinh:
- mã sản phẩm: Mã sản phẩm tương ứng.
- thời điểm mua: Thời điểm mua hàng thực tế.
- doanh thu: Doanh thu của giao dịch (đơn vị BRL).

### 2.3. Mốc Kết thúc Thu thập
- Mốc thời gian kết thúc thu thập dữ liệu phải được xác nhận bao phủ toàn bộ hệ thống; không lấy thời điểm bán cuối cùng của từng sản phẩm riêng lẻ làm mốc kết thúc nhằm tránh thiên lệch chọn mẫu.

## 3. Quy chuẩn Xử lý & Chống Rò rỉ Dữ liệu

1. **Gán nhãn 0 cho sản phẩm chưa bán:** Các sản phẩm trong danh mục không có đơn hàng nào trong vòng 7 ngày kể từ ngày niêm yết sẽ được gán nhãn doanh thu bằng **0 BRL**.
2. **Loại bỏ sản phẩm thiếu thời gian theo dõi:** Các sản phẩm có ngày niêm yết cách mốc kết thúc thu thập dưới 7 ngày sẽ bị loại bỏ để đảm bảo cửa sổ theo dõi luôn đủ 7 ngày.
3. **Loại bỏ đặc trưng tương lai:** Bất kỳ thuộc tính nào được cập nhật sau ngày niêm yết đều bị loại bỏ để tránh rò rỉ dữ liệu.
4. **Biến mục tiêu:** Tổng doanh thu phát sinh trong khoảng $[t_{\text{listing}}, t_{\text{listing}} + 7 \text{ ngày})$. Giá đầu vào là giá niêm yết ban đầu tại thời điểm niêm yết, không dùng giá bán trung bình trong tương lai.

## 4. Thiết kế Khung Đánh giá Mô hình

Khi có dữ liệu đáp ứng yêu cầu, nhóm dự kiến:
- Chia theo thời gian niêm yết; dùng nhiều cửa sổ nếu đủ dữ liệu và loại mẫu Train có cửa sổ nhãn chồng lấn phần đánh giá.
- So sánh cây hồi quy thuộc tính sản phẩm với baseline trung vị danh mục, Zero và một đơn vị theo giá niêm yết. Giá niêm yết không bảo đảm sản phẩm sẽ bán được một đơn vị.
- Học tham số tiền xử lý từ Train, xác định cấu hình trước khi đánh giá và dùng MAE, RMSE trên BRL để so sánh.

Dữ liệu giả lập có thể dùng để kiểm tra cách tạo nhãn, giữ sản phẩm chưa bán và xử lý cửa sổ thiếu ngày. Việc kiểm tra này không chứng minh khả năng dự báo trên dữ liệu thật. Phần trước niêm yết là hướng mở rộng, không phải yêu cầu hoàn thành của bài lab với Olist.
