Bản v4. Mô Phỏng Ma Trận Tổn Thất Chuỗi Cung Ứng (Supply Chain Cost)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là mô hình hóa bài toán kinh tế thực tế về sự bất đối xứng giữa chi phí lưu kho khi dư hàng và tổn thất cơ hội khi đứt hàng.

Bước 1. Thiết lập ma trận tổn thất tài chính bất đối xứng
Triển khai hàm compute_supply_chain_cost() trong src/evaluation/metrics.py:
- Khi dự báo cao hơn thực tế (y_hat > y): Phát sinh chi phí lưu kho và tồn đọng vốn (Overstock holding cost = 2.0 BRL / đơn vị dư thừa).
- Khi dự báo thấp hơn thực tế (y_hat < y): Phát sinh tổn thất mất khách hàng và đứt hàng (Understock stockout penalty = 10.0 BRL / đơn vị thiếu hụt).
Chi phí phạt thiếu hàng nặng gấp 5 lần chi phí thừa hàng, phản ánh đúng thực tế bán lẻ.

Bước 2. Tính toán tổn thất chuỗi cung ứng cho từng mô hình
Tính tổng chi phí tổn thất và quy về mức chi phí trung bình trên mỗi sản phẩm (Cost_Per_Item_BRL):
- **Tuned Model đạt chi phí thấp nhất toàn bảng:** **41,07 BRL / sản phẩm**.
- Decision Tree đạt: 41,66 BRL / sản phẩm.
- Previous Week Baseline đạt: 42,45 BRL / sản phẩm.
- Random Forest đạt: 42,67 BRL / sản phẩm.
- Two-Stage Hurdle đạt: 45,48 BRL / sản phẩm.
- Revenue Mean Baseline đạt: 56,35 BRL / sản phẩm.

Bước 3. Phân tích nguyên nhân Tuned Model tối ưu chi phí nhất
Mặc dù Two-Stage Hurdle có RMSE thấp hơn nhưng Tuned Model cân bằng tốt hơn giữa sai số dương và sai số âm, giúp giảm thiểu đáng kể các trường hợp phạt nặng do đứt hàng.

Bước 4. Đưa Cost_Per_Item_BRL vào bảng so sánh tổng thể
Tích hợp chỉ số tài chính này vào kết quả nghiệm thu chính thức của đồ án.
