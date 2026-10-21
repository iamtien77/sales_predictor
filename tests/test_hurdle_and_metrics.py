"""
test_hurdle_and_metrics.py - Kiểm thử mô hình Two-Stage Hurdle và hàm độ đo WAPE
================================================================================
- test_hurdle_classifier_stage(): Kiểm tra Stage 1 phân loại nhị phân xác suất phát sinh đơn hàng (zero vs non-zero sales probability)
- test_hurdle_regressor_stage(): Kiểm tra Stage 2 hồi quy dự đoán doanh số có điều kiện (conditional positive sales)
- test_hurdle_combined_prediction(): Kiểm tra công thức kết hợp E[Y] = P(Y > 0) * E[Y | Y > 0]
- test_wape_metric_calculation(): Kiểm tra hàm độ đo WAPE (Weighted Absolute Percentage Error) tính đúng theo công thức chuẩn
- test_wape_zero_division_handling(): Kiểm tra xử lý trường hợp tổng actual sales bằng 0 không bị lỗi ZeroDivisionError
- test_hurdle_non_negative_output(): Kiểm tra đầu ra của mô hình Hurdle luôn đảm bảo tính chất không âm (sales >= 0)
"""
