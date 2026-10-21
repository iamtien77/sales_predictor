"""
baselines.py - Các mô hình Baseline đối chuẩn (Heuristics & Simple Models)
==========================================================================
4 mô hình đối chuẩn tối thiểu bắt buộc để làm mốc so sánh hiệu năng:

Classes:
- ZeroRegressor: Luôn dự đoán doanh số = 0 (đối chuẩn cơ sở cho tập dữ liệu zero-inflated)
- MeanRegressor: Luôn dự đoán giá trị trung bình doanh số tập huấn luyện
- LogMeanRegressor: Dự đoán trung bình số học trong không gian log (log1p transform) rồi inverse transform
- PreviousWeekRegressor: Dự đoán doanh số tuần hiện tại bằng đúng doanh số tuần liền trước (lag 1)
"""
