"""
test_models.py - Kiểm thử module mô hình học máy
=================================================
- test_get_models(): Kiểm tra danh sách 4 thuật toán hồi quy + XGBoost
- test_train_and_evaluate(): Kiểm tra tính toán metrics (MSE, MAE, RMSE, R²)
- test_hyperparameter_tuning(): Kiểm tra chạy GridSearchCV trả về best estimator
- test_save_and_load_model(): Kiểm tra tính toàn vẹn của model/pipeline sau khi serialize bằng joblib
"""
