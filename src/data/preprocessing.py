"""
preprocessing.py - Tiền xử lý dữ liệu
=======================================
- handle_missing_values(df): Xử lý missing values (imputation / drop)
- handle_outliers(df, columns): Phát hiện và xử lý outliers (IQR / Z-score capping)
- encode_categorical(df, columns): Chuyển categorical → numerical (One-Hot / Ordinal Encoding)
- split_data(df, target, temporal=True): Phân chia train/test (hỗ trợ cả random split và temporal split)
- scale_features(X_train, X_test): Chuẩn hóa bằng StandardScaler / ColumnTransformer
"""
