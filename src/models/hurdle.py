"""
hurdle.py - Mô hình hồi quy hai giai đoạn Two-Stage Hurdle Model
================================================================
Giải pháp chuyên biệt xử lý tập dữ liệu thưa có tỷ lệ tuần không phát sinh doanh số cao (Zero-inflated):

Classes:
- TwoStageHurdleRegressor:
  * Giai đoạn 1 (Stage 1 - Classifier): Mô hình phân loại nhị phân dự đoán xác suất P(Y > 0)
  * Giai đoạn 2 (Stage 2 - Regressor): Mô hình hồi quy huấn luyện trên tập mẫu dương (Y > 0) dự đoán E[Y | Y > 0]
  * Dự báo kết hợp: E[Y] = P(Y > 0) * E[Y | Y > 0]
"""
