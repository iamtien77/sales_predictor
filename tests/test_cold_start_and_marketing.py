"""
test_cold_start_and_marketing.py - Kiểm thử kịch bản Cold Start và Chiến dịch Marketing
========================================================================================
- test_cold_start_new_product_inference(): Kiểm tra suy luận doanh số cho sản phẩm hoàn toàn mới chưa có lịch sử bán hàng (lag=0, rolling=0)
- test_cold_start_fallback_to_category(): Kiểm tra cơ chế dự phòng (fallback) theo mức trung bình của phân loại ngành hàng khi thiếu dữ liệu cá thể
- test_marketing_discount_sensitivity(): Kiểm tra tính nhạy cảm doanh số khi thay đổi tỷ lệ chiết khấu (discount_rate) và giá bán sau giảm
- test_marketing_ad_spend_impact(): Kiểm tra tác động của việc gia tăng ngân sách quảng cáo/tiếp thị lên doanh số kỳ vọng
- test_non_negative_predictions_under_extreme_scenarios(): Đảm bảo dự báo không âm (sales >= 0) ngay cả dưới các kịch bản kích cầu cực đoan
"""
