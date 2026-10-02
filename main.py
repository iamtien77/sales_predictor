"""
main.py - Entry point, điều phối toàn bộ workflow
===================================================
Thứ tự chạy:
  1. data.loader           → Tải dataset (Kaggle e-commerce / Online Retail)
  2. data.preprocessing    → Xử lý missing values, outliers, encode categorical
  3. data.feature_engineering → Tạo features mới (popularity, CLV, seasonal trends)
  4. evaluation.visualization → EDA
  5. models.regressors     → Huấn luyện (Linear Reg, Decision Tree, Random Forest, Gradient Boosting)
  6. evaluation.metrics    → Đánh giá (MSE, MAE, RMSE, R²)
  7. models.tuning         → Tối ưu hyperparameter (GridSearchCV)
  8. evaluation.visualization → Biểu đồ so sánh
  9. deployment.predictor  → Demo dự đoán sales cho sản phẩm mới
"""
