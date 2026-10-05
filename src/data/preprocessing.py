"""
preprocessing.py - Tiền xử lý dữ liệu
=======================================
- handle_missing_values(df): Xử lý missing values (imputation / drop)
- handle_outliers(df, columns): Phát hiện và xử lý outliers (IQR / Z-score capping)
- encode_categorical(df, columns): Chuyển categorical → numerical (One-Hot / Ordinal Encoding)
- split_data(df, target, train_size=0.8, val_size=0.1, test_size=0.1, temporal=True): Phân chia train/val/test theo tỷ lệ 8:1:1 (hỗ trợ cả random split và temporal split)
- scale_features(X_train, X_val, X_test): Chuẩn hóa bằng StandardScaler / ColumnTransformer (fit trên train, transform trên val và test)
"""
