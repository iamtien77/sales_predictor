# 📊 Predicting Product Sales

## 1. Mô tả bài toán

Một công ty e-commerce muốn **dự đoán doanh số bán hàng** (sales) của sản phẩm nhằm **tối ưu hóa quản lý kho** và **chiến lược marketing**.

Đây là bài toán **hồi quy (regression)** — dự đoán một biến liên tục (sales) dựa trên nhiều đặc trưng đầu vào.

### Dữ liệu đầu vào
- **Dataset**: E-commerce / Online Retail dataset từ Kaggle
- Các nhóm đặc trưng:
  - **Product information**: Danh mục sản phẩm (category), giá (price), mô tả (description)
  - **Customer demographics**: Tuổi (age), giới tính (gender), vị trí (location)
  - **Purchase history**: Lịch sử mua (previous purchases), tần suất mua (purchase frequency)
  - **Marketing campaign data**: Chiến dịch chạy (campaigns run), chi phí quảng cáo (ad spend)
  - **Seasonal & holiday**: Thông tin mùa vụ, ngày lễ

### Biến mục tiêu (Target Variable)
- **Product sales** — biến số liên tục (numerical)

### Đầu ra mong muốn
- Giá trị dự đoán sales cho từng sản phẩm
- Bảng so sánh hiệu suất giữa các mô hình hồi quy
- Biểu đồ trực quan (Actual vs Predicted, Residuals, Feature Importance)
- Dự đoán theo nhiều kịch bản (new product launch, marketing campaign, seasonal promotion)

---

## 2. Xác định yêu cầu

### Yêu cầu chức năng
| # | Yêu cầu | Mô tả |
|---|---------|-------|
| F1 | Tải dữ liệu | Tải dataset e-commerce từ Kaggle hoặc file local |
| F2 | Tiền xử lý | Xử lý missing values, outliers, encode categorical, chia train/test, scale |
| F3 | Feature Engineering | Tạo features mới: product popularity, CLV, seasonal trends, purchase frequency |
| F4 | Huấn luyện mô hình | Train 4 mô hình: Linear Regression, Decision Tree, Random Forest, Gradient Boosting |
| F5 | Đánh giá | Tính MSE, MAE, RMSE, R² |
| F6 | Hyperparameter Tuning | Tối ưu tham số bằng GridSearchCV |
| F7 | Phân tích lỗi | Phân tích residuals, tìm mẫu có sai số lớn nhất |
| F8 | Trực quan hóa | EDA, Actual vs Predicted, Residuals, Feature Importance, so sánh mô hình |
| F9 | Deployment | Lưu model, dự đoán sales cho sản phẩm mới, dự đoán theo kịch bản |

### Yêu cầu phi chức năng
- Code có cấu trúc module rõ ràng, dễ bảo trì
- Logging theo dõi quá trình chạy
- Kết quả (biểu đồ, bảng) được lưu ra thư mục `results/`
- Có file `requirements.txt` để tái tạo môi trường

---

## 3. Phân tích nhiệm vụ từng module

### 3.1. `config.py` — Cấu hình project
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| Đường dẫn dữ liệu | URL dataset Kaggle, đường dẫn file local, thư mục output |
| Tên features | Danh sách features theo nhóm (product, customer, marketing, seasonal) |
| Tham số chung | `test_size=0.2`, `random_state=42`, `cv_folds=5` |
| Hyperparameter grids | Dict tham số GridSearchCV cho Decision Tree, Random Forest, Gradient Boosting |

### 3.2. `data/loader.py` — Tải dữ liệu
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `load_data()` | Tải dataset e-commerce từ Kaggle hoặc file CSV local, trả về DataFrame |
| `explore_data(df)` | In shape, kiểu dữ liệu, missing values, thống kê mô tả, phân bố target |

### 3.3. `data/preprocessing.py` — Tiền xử lý dữ liệu
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `handle_missing_values(df)` | Xử lý giá trị thiếu bằng imputation (mean/median) hoặc drop |
| `handle_outliers(df, columns)` | Phát hiện outliers bằng IQR hoặc Z-score, xử lý (clip/remove) |
| `encode_categorical(df, columns)` | Chuyển biến categorical → numerical (Label Encoding / One-Hot Encoding) |
| `split_data(df, target)` | Tách features (X) và target (y), chia train/test |
| `scale_features(X_train, X_test)` | Chuẩn hóa bằng StandardScaler (fit trên train, transform trên test) |

### 3.4. `data/feature_engineering.py` — Tạo features mới
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `add_product_popularity(df)` | Tạo feature popularity dựa trên số lượng reviews, ratings |
| `add_customer_lifetime_value(df)` | Tính Customer Lifetime Value (CLV) từ lịch sử mua hàng |
| `add_seasonal_trends(df)` | Thêm features mùa vụ: month, quarter, is_holiday |
| `add_purchase_frequency(df)` | Tính tần suất mua hàng của khách |
| `add_marketing_features(df)` | Tính ROI marketing, tỷ lệ ad spend / revenue |

### 3.5. `models/regressors.py` — 4 mô hình hồi quy
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `get_models()` | Trả về dict `{name: model}` cho Linear Regression, Decision Tree, Random Forest, Gradient Boosting |
| `train_and_evaluate(models, X_train, X_test, y_train, y_test)` | Huấn luyện từng model, tính metrics (MSE, MAE, RMSE, R²) |

### 3.6. `models/tuning.py` — Tối ưu Hyperparameter
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `tune_decision_tree(X, y)` | GridSearchCV cho Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf` |
| `tune_random_forest(X, y)` | GridSearchCV cho Random Forest: `n_estimators`, `max_depth`, `max_features` |
| `tune_gradient_boosting(X, y)` | GridSearchCV cho Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth` |
| `tune_all(X, y)` | Gọi tất cả, trả về dict `{name: best_estimator}` |

### 3.7. `models/persistence.py` — Lưu/tải model
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `save_model(model, scaler, path)` | Serialize model + scaler bằng joblib |
| `load_model(path)` | Deserialize model đã lưu |
| `save_pipeline(pipeline, path)` | Lưu toàn bộ pipeline (scaler → encoder → model) |
| `load_pipeline(path)` | Tải pipeline đã lưu |

### 3.8. `evaluation/metrics.py` — Tính toán metrics hồi quy
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `compute_metrics(y_test, y_pred)` | Tính MSE, MAE, RMSE, R² |
| `print_regression_reports(results)` | In báo cáo metrics cho từng mô hình |
| `build_summary_table(all_results)` | Tạo DataFrame tổng hợp, sắp xếp theo RMSE |

### 3.9. `evaluation/visualization.py` — Trực quan hóa
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `plot_eda(df)` | EDA: phân bố sales, correlation heatmap, boxplot outliers |
| `plot_actual_vs_predicted(results, y_test)` | Scatter plot Actual vs Predicted cho từng model |
| `plot_residuals(results, y_test)` | Biểu đồ residuals (residual vs predicted, histogram) |
| `plot_feature_importance(model, names)` | Bar chart Feature Importance (Random Forest / Gradient Boosting) |
| `plot_model_comparison(summary_df)` | Biểu đồ cột grouped so sánh MSE/MAE/RMSE/R² |
| `plot_seasonal_trends(df)` | Biểu đồ xu hướng sales theo mùa (line chart theo month/quarter) |

### 3.10. `evaluation/error_analysis.py` — Phân tích lỗi dự đoán
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `analyze_residuals(X_test, y_test, y_pred, names)` | Phân tích residuals theo từng feature |
| `find_worst_predictions(X_test, y_test, y_pred, n)` | Tìm n mẫu có sai số lớn nhất, xem đặc trưng |
| `plot_error_distribution(y_test, y_pred)` | Histogram phân bố sai số |
| `compare_error_patterns(results, X_test, y_test)` | So sánh pattern lỗi giữa các mô hình |

### 3.11. `deployment/predictor.py` — Triển khai dự đoán
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `predict_sales(features, model, scaler)` | Nhận vector features → trả về giá trị sales dự đoán |
| `demo_prediction(model, scaler, X_test, y_test)` | Lấy ngẫu nhiên 1 mẫu test, dự đoán và so sánh với actual |
| `predict_scenarios(model, scaler, scenarios)` | Dự đoán nhiều kịch bản (new product, marketing campaign, seasonal promo) |

### 3.12. `utils/logger.py` — Logging và tiện ích
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `setup_logger(name, log_file)` | Tạo logger ghi ra console + file |
| `log_step(step_name)` | Decorator đo thời gian chạy từng bước |
| `print_section_header(title)` | In header đẹp phân cách các section |

### 3.13. `main.py` — Entry point
| Bước | Gọi module | Mô tả |
|------|-----------|-------|
| 1 | `data.loader` | Tải + khám phá dữ liệu |
| 2 | `data.preprocessing` | Xử lý missing, outliers, encode, split, scale |
| 3 | `data.feature_engineering` | Tạo features mới (popularity, CLV, seasonal, ...) |
| 4 | `evaluation.visualization` | EDA biểu đồ |
| 5 | `models.regressors` | Huấn luyện 4 mô hình hồi quy |
| 6 | `evaluation.metrics` | Đánh giá metrics |
| 7 | `models.tuning` | Tối ưu hyperparameter |
| 8 | `evaluation.error_analysis` | Phân tích lỗi dự đoán |
| 9 | `evaluation.visualization` | Biểu đồ so sánh tổng hợp |
| 10 | `models.persistence` | Lưu model tốt nhất |
| 11 | `deployment.predictor` | Demo dự đoán sales + kịch bản |

---

## 4. Công nghệ sử dụng

| Thư viện | Mục đích |
|----------|----------|
| `pandas` | Xử lý dữ liệu dạng bảng |
| `numpy` | Tính toán số học |
| `scikit-learn` | Mô hình ML, tiền xử lý, đánh giá |
| `matplotlib` | Vẽ biểu đồ cơ bản |
| `seaborn` | Biểu đồ thống kê nâng cao |
| `joblib` | Lưu/tải model |

---

## 5. Cấu trúc thư mục

```
sales_predictor/
├── main.py                         # Entry point, điều phối workflow
├── config.py                       # Cấu hình (paths, params, grids)
├── requirements.txt                # Dependencies
├── README.md                       # Tài liệu phân tích bài toán
├── data/
│   ├── loader.py                   # Tải dữ liệu
│   ├── preprocessing.py            # Tiền xử lý (missing, outlier, encode, split, scale)
│   └── feature_engineering.py      # Tạo features mới (popularity, CLV, seasonal)
├── models/
│   ├── regressors.py               # 4 mô hình: LR, DT, RF, GB
│   ├── tuning.py                   # GridSearchCV hyperparameter tuning
│   └── persistence.py              # Lưu/tải model (joblib)
├── evaluation/
│   ├── metrics.py                  # MSE, MAE, RMSE, R²
│   ├── visualization.py            # EDA, Actual vs Predicted, Residuals, so sánh
│   └── error_analysis.py           # Phân tích lỗi dự đoán
├── deployment/
│   └── predictor.py                # Dự đoán sales, demo kịch bản
├── results/                        # Output: biểu đồ (.png), bảng (.csv)
└── utils/
    └── logger.py                   # Logging và tiện ích
```
