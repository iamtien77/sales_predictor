Bản v1. Quy Trình Nghiệm Thu Độc Lập Trên Test Set & Nghịch Đảo BRL

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là thiết lập giao thức kiểm định khách quan trên 293.717 dòng Test Set theo thang đo tiền tệ thực tế BRL.

Bước 1. Tách biệt hoàn toàn tập kiểm tra độc lập (Test Set)
Tập Test gồm 293.717 dòng Sản phẩm – Tuần nằm ở các tuần cuối cùng của chuỗi thời gian Olist.
Tập Test hoàn toàn chưa từng được nhìn thấy bởi bất kỳ mô hình nào trong quá trình huấn luyện hay tinh chỉnh siêu tham số.

Bước 2. Nguyên tắc nghịch đảo hàm mũ an toàn
Dự báo của toàn bộ các mô hình trong không gian log (y_pred_log) được chuyển đổi ngược về thang đo tiền tệ thực tế:
y_pred_money = max(0.0, exp(y_pred_log) - 1).
Nhãn thực tế y_test cũng được quy đổi ngược: y_true_money = max(0.0, exp(y_test_log) - 1).

Bước 3. Khóa thang đo so sánh Target_Scale = 'BRL'
Toàn bộ sai số và chỉ số đo lường bắt buộc phải tính toán trên y_true_money và y_pred_money.
Nghiêm cấm báo cáo sai số trên thang log rồi hiểu nhầm là tiền tệ BRL.

Bước 4. Duyệt tuần tự qua toàn bộ danh mục mô hình
Vòng lặp tính toán sai số cho 10 mô hình và baseline:
Two-Stage Hurdle, Tuned Model, Decision Tree, Random Forest, Gradient Boosting, Revenue Mean, Log Mean, Zero Baseline, Previous Week, Linear Regression.

Bước 5. Lưu trữ kết quả dự báo thô
Lưu mảng dự báo y_pred của từng mô hình vào cấu trúc results_eval sẵn sàng cho bước phân tích chuyên sâu.
