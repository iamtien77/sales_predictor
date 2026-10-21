"""
test_weekly_forecasting.py - Kiểm thử toàn vẹn chuỗi thời gian tuần và hoàn thiện Zero-Sales
=============================================================================================
- test_weekly_panel_aggregation(): Kiểm tra tổng hợp giao dịch từ cấp item lên cấp (product_id, year_week)
- test_zero_sales_completion(): Kiểm tra điền đầy đủ các tuần không có giao dịch với số lượng bán bằng 0 cho từng sản phẩm
- test_temporal_continuity(): Kiểm tra tính liên tục của trục thời gian tuần (không khuyết thiếu tuần từ min_date đến max_date)
- test_lag_features_integrity(): Kiểm tra tạo các biến trễ lag_1_sales, lag_2_sales, lag_4_sales đúng tuần và không dịch chuyển sai lệch
- test_rolling_window_statistics(): Kiểm tra rolling mean và rolling std 4 tuần được tính toán đúng trên chuỗi tuần đã điền 0
- test_no_future_information_in_lags(): Đảm bảo các đặc trưng tuần tuyệt đối không sử dụng thông tin của tuần hiện tại hoặc tương lai
"""
