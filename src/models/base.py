"""
base.py - Lớp cơ sở trừu tượng cho các mô hình học máy From-Scratch
===================================================================
Định nghĩa giao diện chuẩn mực (API contract) thuần NumPy tương thích scikit-learn:

Classes:
- BaseEstimator: Lớp cha quản lý siêu tham số (get_params, set_params)
- BaseRegressor(BaseEstimator): Lớp cơ sở trừu tượng cho mô hình hồi quy (fit, predict, score)
- BaseClassifier(BaseEstimator): Lớp cơ sở trừu tượng cho mô hình phân loại (fit, predict, predict_proba)
"""
