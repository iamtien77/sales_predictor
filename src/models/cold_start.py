"""
cold_start.py - Mô hình và chiến lược dự đoán cho sản phẩm mới (Cold Start)
============================================================================
Xử lý bài toán suy luận doanh số cho sản phẩm chưa có lịch sử bán hàng (không có lag/rolling):

Classes:
- CategoryAverageFallback: Cơ chế dự phòng tính doanh số kỳ vọng dựa trên phân loại ngành hàng và khoảng giá
- ColdStartRegressor: Mô hình kết hợp đặc trưng tĩnh (thuộc tính vật lý, danh mục, giá bán) với cơ chế fallback
"""
