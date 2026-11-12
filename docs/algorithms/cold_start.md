# Proxy Doanh thu Bảy Ngày Kể Từ Giao Dịch Đầu Tiên Quan Sát Được

Tài liệu đề xuất một thử nghiệm nhỏ trong bài lab: ước lượng doanh thu bảy ngày kể từ giao dịch đầu tiên quan sát được. Mốc này đã có một lần bán, nên không đại diện cho dự báo trước khi sản phẩm được niêm yết.

## 1. Quy ước bài toán dự báo

- **Mốc bắt đầu (Origin):** Thời điểm giao dịch đầu tiên quan sát được của sản phẩm trong tập dữ liệu.
- **Giá đầu vào:** Giá bán của giao dịch đầu tiên đã biết, tuyệt đối không dùng giá trung vị của các giao dịch trong tương lai.
- **Biến mục tiêu (Target):** Tổng doanh thu các mặt hàng của sản phẩm trong khoảng thời gian $[t_{\text{origin}}, t_{\text{origin}} + 7 \text{ ngày})$, bao gồm cả giao dịch đầu tiên đã biết.
- **Điều kiện hợp lệ của mẫu:** Loại bỏ các sản phẩm không có đủ 7 ngày quan sát sau mốc bắt đầu để tránh biến mục tiêu bị cắt ngắn.
- **Giới hạn dữ liệu:** Dữ liệu Olist không chứa các sản phẩm niêm yết nhưng chưa từng bán được hàng. Do đó, nhóm xác định thử nghiệm này chỉ đánh giá khả năng ước lượng quán tính ban đầu của các sản phẩm đã phát sinh giao dịch, chưa thể kết luận về các sản phẩm ra mắt không có người mua.

## 2. Thiết kế Huấn luyện và Đánh giá

Nhóm thiết kế quy trình phân chia dữ liệu theo trục thời gian mốc bắt đầu:
- **Tập Kiểm thử (Test Set):** Dành 20% số ngày bắt đầu phân biệt cuối cùng làm tập Test độc lập.
- **Cơ chế Purging chống rò rỉ nhãn:** Các mẫu trong tập Train có cửa sổ nhãn 7 ngày vượt qua ranh giới thời gian của tập Test đều bị loại bỏ hoàn toàn (purge) để tránh rò rỉ dữ liệu.
- **Tiền xử lý độc lập:** Các thống kê mã hóa danh mục theo nhãn, giá trị điền thiếu và hệ số chuẩn hóa chỉ được học từ Train. Đặc trưng số dự kiến gồm giá, khối lượng, số ảnh, độ dài tên và mô tả; có thể biến đổi log để giảm khoảng chênh lệch. Test dùng lại cùng tham số. Danh mục ít mẫu cần được làm trơn về thống kê chung của Train.

Các ứng viên dự kiến dùng để so sánh:
1. **Baseline trung vị danh mục:** dự báo trung vị doanh thu của danh mục tương ứng trên Train.
2. **Baseline một đơn vị:** dự báo bằng giá giao dịch đầu tiên, $\hat{s}=p_0$.
3. **Cây hồi quy thuộc tính sản phẩm:** học trên nhãn log doanh thu, sau đó chuyển về BRL. Độ sâu và số mẫu tối thiểu tại lá cần được xác định trước khi đánh giá Test; không thay đổi theo điểm Test.

### Vì sao baseline một đơn vị là đối chứng quan trọng?
- Theo thiết kế bài toán, biến mục tiêu 7 ngày đã bao hàm cả giao dịch đầu tiên.
- Đối với các sản phẩm có sức mua chậm (chỉ bán được đúng 1 đơn vị duy nhất trong vòng 7 ngày đầu), biến mục tiêu thực tế sẽ bằng đúng giá bán của giao dịch đó. Do đó, Baseline Một đơn vị sẽ đạt sai số bằng 0 trên phân khúc sản phẩm này.
- Mô hình cây quyết định dự báo trung bình theo vùng lá trong không gian logarit, nên không tự nhiên gán $\hat{s} = p_0$. Việc so sánh với Baseline Một đơn vị giúp nhóm đánh giá xem liệu các thuộc tính sản phẩm có thực sự bổ sung thông tin hữu ích hay mô hình đang bị nhiễu so với một quy tắc suy luận cơ bản.

## 3. Thiết kế Kiểm chứng và Chống rò rỉ

Nhóm dự kiến ghi lại mốc chia, số mẫu hợp lệ, số mẫu loại vì thiếu thời gian quan sát hoặc chồng lấn nhãn, cùng tỷ lệ sản phẩm chỉ bán một đơn vị. Các thông tin này giúp giải thích phạm vi đánh giá.

| Mô hình | Chỉ số dự kiến | Phạm vi so sánh | Câu hỏi cần kiểm tra |
| --- | --- | --- | --- |
| Baseline một đơn vị | MAE, RMSE, R² trên BRL | Toàn tập và từng nhóm sản lượng | Quy tắc dựa vào giao dịch đã biết có đủ làm mốc tham chiếu không? |
| Cây hồi quy | MAE, RMSE, R² trên BRL | Cùng mẫu với hai baseline | Thuộc tính sản phẩm có giúp giảm sai số không? |
| Baseline trung vị danh mục | MAE, RMSE, R² trên BRL | Cùng mẫu với cây hồi quy | Thống kê danh mục đóng góp bao nhiêu so với giá đầu tiên? |

Điểm Test dùng để mô tả và so sánh các ứng viên đã xác định trước, không dùng để chọn lại tham số hoặc mặc định cây là phương án tốt nhất.

## 4. Cơ chế Dự phòng và Giới hạn Áp dụng

- **Cơ chế dự phòng:** Trong trường hợp gặp sản phẩm mới thiếu thông tin đặc trưng hoặc danh mục chưa từng xuất hiện, dự kiến dùng thống kê từ Train để điền thiếu và mã hóa danh mục mới. Nếu không có mô hình phù hợp, có thể dùng quy tắc một đơn vị theo giá đầu tiên hợp lệ và ghi rõ đây là dự phòng. Không suy ra nhu cầu trước bán từ quy tắc này.
- **Giới hạn bài toán:** Nhóm lưu ý rằng kết quả proxy 7 ngày sau lần bán đầu không đại diện cho bài toán dự báo trước khi sản phẩm ra mắt. Để giải quyết bài toán trước niêm yết, cần có dữ liệu catalog thực tế như quy định trong [Quy chuẩn Dữ liệu Niêm yết](listing_data_contract.md).
