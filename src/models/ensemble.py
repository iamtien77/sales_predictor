"""
ensemble.py - Các mô hình tổ hợp From-Scratch (Ensemble Methods)
=================================================================
Hiện thực thuần NumPy không phụ thuộc thư viện ngoài:

Classes:
- RandomForestRegressorScratch: Mô hình rừng ngẫu nhiên sử dụng kỹ thuật Bagging (Bootstrap Aggregating)
  và lấy mẫu ngẫu nhiên không gian đặc trưng (Feature Subsampling / max_features)
- GradientBoostingRegressorScratch: Mô hình tăng cường độ dốc (Gradient Boosting) huấn luyện tuần tự
  các cây quyết định nông để khớp phần dư (residuals) theo hàm mất mát Bình phương Tối thiểu (MSE Loss)
"""
