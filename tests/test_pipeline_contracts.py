"""
test_pipeline_contracts.py - Kiểm thử hợp đồng ràng buộc đầu vào/ra của Pipeline
=================================================================================
- test_input_schema_validation(): Kiểm tra bắt lỗi khi thiếu các cột bắt buộc trong DataFrame đầu vào
- test_invalid_data_types(): Kiểm tra cảnh báo hoặc raise ngoại lệ khi gặp kiểu dữ liệu không hợp lệ (chuỗi thay vì số, ngày tháng sai format)
- test_unseen_categorical_levels(): Kiểm tra cơ chế xử lý các giá trị phân loại mới (unseen categories) chưa từng xuất hiện trong tập huấn luyện
- test_output_prediction_contract(): Kiểm tra định dạng vector đầu ra (NumPy 1D array/Series float, không chứa NaN/Inf, độ dài khớp số mẫu)
- test_pipeline_immutability(): Kiểm tra pipeline không làm thay đổi (mutate in-place) DataFrame gốc đầu vào
"""
