"""
preprocessing.py - Tiền xử lý dữ liệu
=======================================
- handle_missing_values(df): Xử lý missing values (imputation / drop)
- handle_outliers(df, columns): Phát hiện và xử lý outliers (IQR / Z-score)
- encode_categorical(df, columns): Chuyển categorical → numerical (Label / One-Hot Encoding)
- split_data(df, target): Tách features/target, chia train/test
- scale_features(X_train, X_test): Chuẩn hóa bằng StandardScaler
"""
