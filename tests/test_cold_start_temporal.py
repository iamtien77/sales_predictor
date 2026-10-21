"""
test_cold_start_temporal.py - Kiểm thử phân chia dữ liệu thời gian cho sản phẩm Cold-Start
==========================================================================================
- test_temporal_cold_start_split(): Kiểm tra phân chia tập dữ liệu tách biệt theo thời gian và đánh dấu chính xác sản phẩm mới ra mắt
- test_no_future_leakage_in_cold_start(): Đảm bảo tuyệt đối không rò rỉ dữ liệu tương lai (look-ahead bias) vào tập train của sản phẩm cold-start
- test_cold_start_evaluation_metrics(): Kiểm tra tính toán riêng biệt các độ đo lỗi (WAPE, MAE) cho tập sản phẩm cold-start so với sản phẩm truyền thống
- test_cold_start_transition_to_warm(): Kiểm tra quá trình chuyển dịch trạng thái từ cold-start sang warm sau khi sản phẩm tích lũy đủ các tuần lịch sử
"""
