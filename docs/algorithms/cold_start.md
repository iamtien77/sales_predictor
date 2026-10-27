# Proxy doanh thu bảy ngày kể từ giao dịch đầu tiên quan sát được

> **Snapshot hiện tại:** artifact ghi `2026-10-27 16:26:38`, chế độ `full`, có tuning, 655.066 dòng Train. Nguồn: [Test](../../results/metrics/model_comparison.csv), [Validation](../../results/metrics/validation_comparison.csv), [tuning](../../results/metrics/tuning_history.csv). Thời gian fit thuộc lần chạy này. Kết quả dùng snapshot `delivered` cuối kỳ, chưa bảo đảm khả dụng tại forecast origin.

Tên lớp `ColdStartProductRegressor` và tên trường `launch_sales_brl` được giữ lại để tương thích với mã nguồn cũ. Các tên này **không có nghĩa** mô hình đã được kiểm chứng khả năng dự báo cold-start từ ngày niêm yết sản phẩm.

## Quy ước bài toán dự báo

- **Mốc bắt đầu (origin):** thời điểm mua đầu tiên quan sát được của sản phẩm trong các đơn có trạng thái cuối kỳ là `delivered`.
- **Giá:** giá giao dịch đầu tiên, không dùng trung vị giá của các giao dịch tương lai.
- **Biến mục tiêu (target):** tổng doanh thu mặt hàng trong khoảng `[origin, origin + 7 ngày)`, bao gồm giao dịch đầu tiên đã biết.
- Loại các sản phẩm không có đủ bảy ngày theo dõi sau mốc bắt đầu.
- Dữ liệu không có sản phẩm chưa bán được hàng; chưa đưa ra kết luận về các trường hợp ra mắt không phát sinh doanh thu.
- Trạng thái `delivered` và metadata sản phẩm là snapshot hồi cứu của Olist; chưa thể xác minh chúng đã khả dụng tại thời điểm dự báo. Đây là thí nghiệm proxy ngoại tuyến.

## Huấn luyện và đánh giá

`src/evaluation/cold_start.py` dành 20% số ngày bắt đầu phân biệt cuối cùng làm tập Test. Các mẫu Train có cửa sổ nhãn bảy ngày vượt qua ranh giới Test được loại bỏ (purge). Bộ mã hóa danh mục và bộ điền giá trị số khuyết thiếu chỉ được fit trên Train.

Cây được xác định trước với độ sâu 6, số mẫu tối thiểu để chia nút là 20 và số mẫu tối thiểu tại lá là 10. Không dùng Test để chọn mô hình hoặc siêu tham số.

Tất cả ứng viên được đánh giá trên cùng các dòng Test:

1. Baseline trung vị doanh thu theo danh mục, fit trên Train.
2. Baseline một đơn vị: dự báo doanh thu bằng giá giao dịch đầu tiên đã biết.
3. Cây dựa trên thông tin sản phẩm: năm đặc trưng số và target encoding theo danh mục có làm trơn; học trên `log1p` doanh thu rồi chuyển đổi ngược để tính metrics.

Dự báo sau khi chuyển đổi ngược từ log không mặc nhiên là kỳ vọng số học có điều kiện hoặc trung vị có điều kiện. Cần so sánh cả MAE và RMSE để đánh giá chất lượng.

### Vì sao baseline một đơn vị là đối chứng mạnh?

Target bao gồm giao dịch đầu tiên, với giá đã có trong đầu vào. Khi sản phẩm chỉ bán một đơn vị trong cửa sổ bảy ngày, target bằng đúng giá đó, nên baseline một đơn vị có sai số bằng 0 trên mẫu này. Manifest của thí nghiệm đã lưu ghi nhận **85,47% sản phẩm Test chỉ bán một đơn vị**; CSV đánh giá cho thấy baseline này có MAE và RMSE thấp hơn cây dựa trên thông tin sản phẩm. Đây là bằng chứng trên holdout đã lưu, không bảo đảm baseline sẽ thắng trên các bộ dữ liệu khác.

Mỗi lá cây gộp nhiều sản phẩm và dự báo giá trị trung bình trong không gian log, nên không bắt buộc giữ quan hệ `sales = first_price`, kể cả với sản phẩm chỉ bán một đơn vị. Chuyển đổi ngược giá trị trung bình này cũng không trực tiếp tối thiểu hóa sai số bình phương theo BRL. Các cơ chế đó có thể góp phần tạo ra chênh lệch quan sát được; muốn xác định nguyên nhân cần thí nghiệm đối chứng, không chỉ dựa vào metrics tổng hợp. Baseline một đơn vị vẫn có thể dự báo thiếu đối với sản phẩm phát sinh thêm giao dịch.

Bộ lọc `delivered` hồi cứu cũng ảnh hưởng đến dữ liệu dự báo tuần: các đơn hàng được nhóm theo thời điểm mua nhưng sử dụng kết quả giao hàng cuối cùng. Fit preprocessing trên Train và temporal split không khiến thông tin tương lai này trở nên khả dụng tại một mốc dự báo trước đó. Vì vậy, kết quả tuần vẫn là backtest ngoại tuyến trên snapshot cho đến khi tái dựng được trạng thái đơn hàng theo thời gian và xác định rõ target là doanh thu đặt hàng hay doanh thu đã hoàn tất.

## Tái lập thí nghiệm

```powershell
.venv/Scripts/python.exe -m src.evaluation.cold_start
```

Lệnh trên chạy lại thí nghiệm, xuất `results/metrics/cold_start_evaluation.csv` và `cold_start_manifest.json`, thay thành phần cold-start trong gói triển khai tuần hiện có, đồng thời cập nhật phần kết quả trong README. Cần có dữ liệu Olist gốc và gói mô hình tuần đã lưu. Luồng đầy đủ trong `main.py` sử dụng cùng hàm đánh giá này.

Manifest ghi lại các mốc ngày, số lượng mẫu, số mẫu bị purge, dấu vân tay dữ liệu và các giả định. Metadata của artifact chứa hash liên kết các tệp đánh giá với mô hình đã fit. Xem mục 4 được sinh tự động trong README để lấy số liệu hiện tại. Cây được đóng gói là ứng viên thực nghiệm, không đồng nghĩa với việc nó tốt hơn baseline giá.

## Giao diện và cơ chế dự phòng

Giao diện nạp mô hình đã fit từ gói artifact và hiển thị baseline một đơn vị để đối chiếu. Khi không có mô hình, hàm hỗ trợ trả kết quả theo quy tắc giả định bán một đơn vị và ghi rõ đây là phương án dự phòng, chưa phải dự báo ML.

Marketing là mô phỏng riêng dựa trên giả định; chưa phải tác động nhân quả được học từ dữ liệu.
