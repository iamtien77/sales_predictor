"""
linear.py - Mô hình hồi quy tuyến tính From-Scratch
===================================================
Thuật toán hồi quy tuyến tính viết thuần bằng ma trận NumPy:

Classes:
- LinearRegressionScratch:
  * Huấn luyện bằng phương trình pháp tuyến (Normal Equation: beta = (X^T X)^(-1) X^T y) kèm Regularization Ridge (L2)
  * Tùy chọn tối ưu bằng Gradient Descent theo batch/mini-batch
  * Cơ chế Adaptive Clamping: Tự động chặn dưới giá trị dự báo không âm (sales >= 0) phù hợp thực tế kinh doanh
"""
