"""
main.py - Entry point, điều phối toàn bộ workflow
===================================================
Thứ tự chạy:
  1. src.data.loader           → Tải dataset (Kaggle e-commerce / Online Retail)
  2. src.data.preprocessing    → Xử lý missing values, outliers, encode categorical, split train/test
  3. src.data.feature_engineering → Tạo features mới (product popularity, CLV, seasonal trends, marketing)
  4. src.evaluation.visualization → EDA biểu đồ phân tích
  5. src.models.regressors     → Huấn luyện (Linear Reg, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
  6. src.evaluation.metrics    → Đánh giá (MSE, MAE, RMSE, R²)
  7. src.models.tuning         → Tối ưu hyperparameter (GridSearchCV)
  8. src.evaluation.error_analysis → Phân tích lỗi và các trường hợp sai lệch lớn nhất
  9. src.models.persistence   → Lưu model / pipeline
  10. src.deployment.predictor → Demo dự đoán sales theo kịch bản
"""
