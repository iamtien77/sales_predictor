"""
test_preprocessing.py - Kiểm thử module tiền xử lý dữ liệu
===========================================================
- test_handle_missing_values(): Kiểm tra không còn giá trị NaN sau khi xử lý
- test_handle_outliers(): Kiểm tra giá trị nằm trong ngưỡng IQR
- test_encode_categorical(): Kiểm tra các cột dạng chuỗi được chuyển thành số
- test_split_data_random(): Kiểm tra kích thước 3 tập train, val và test theo tỷ lệ 8:1:1 khi random split
- test_split_data_temporal(): Kiểm tra thứ tự thời gian của 3 tập (8:1:1), tránh rò rỉ dữ liệu (look-ahead bias)
"""
