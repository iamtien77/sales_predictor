"""
config.py - Cấu hình trung tâm cho toàn bộ dự án Dự đoán Doanh số (Sales Predictor)
Tuân thủ tiêu chuẩn From-Scratch (NumPy thuần túy) và Temporal Split chống Data Leakage
"""
from pathlib import Path

# ==========================================
# 1. ĐƯỜNG DẪN HỆ THỐNG (PATHS)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Đường dẫn kết quả thực nghiệm dành riêng cho Jupyter Notebooks
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOK_RESULTS_DIR = NOTEBOOKS_DIR / "results"
NOTEBOOK_SAVED_MODELS_DIR = NOTEBOOK_RESULTS_DIR / "saved_models"
NOTEBOOK_FIGURES_DIR = NOTEBOOK_RESULTS_DIR / "figures"
NOTEBOOK_METRICS_DIR = NOTEBOOK_RESULTS_DIR / "metrics"

# Cấu hình trực quan hóa & độ phân giải xuất bản biểu đồ
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Đảm bảo các thư mục luôn tồn tại
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    NOTEBOOK_RESULTS_DIR,
    NOTEBOOK_SAVED_MODELS_DIR,
    NOTEBOOK_FIGURES_DIR,
    NOTEBOOK_METRICS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Tên file dataset mặc định
RAW_DATA_FILE = RAW_DATA_DIR / "olist_merged_sales_dataset.csv"
RAW_CUSTOMERS_FILE = RAW_DATA_DIR / "olist_customers_dataset.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_sales_data.csv"

# Đường dẫn cache ma trận đặc trưng From-Scratch
CACHE_X_TRAIN = PROCESSED_DATA_DIR / "X_train.npz"
CACHE_Y_TRAIN = PROCESSED_DATA_DIR / "y_train.npy"
CACHE_X_VAL = PROCESSED_DATA_DIR / "X_val.npz"
CACHE_Y_VAL = PROCESSED_DATA_DIR / "y_val.npy"
CACHE_X_TEST = PROCESSED_DATA_DIR / "X_test.npz"
CACHE_Y_TEST = PROCESSED_DATA_DIR / "y_test.npy"
CACHE_FEATURE_NAMES = PROCESSED_DATA_DIR / "feature_names.json"
CACHE_PIPELINE = PROCESSED_DATA_DIR / "preprocessor_pipeline.pkl"

# ==========================================
# 2. TOÀN BỘ ĐẶC TRƯNG (FULL FEATURE TAXONOMY)
# ==========================================
TARGET_COLUMN = "sales"
LOG_TARGET_COLUMN = "log_sales"
TIMESTAMP_COLUMN = "order_purchase_timestamp"

# 1. Product Information & Popularity
PRODUCT_BASE_COLS = ["category", "price", "description"]
PRODUCT_POPULARITY_COLS = ["rating", "num_reviews", "popularity_score", "price_relative_to_cat_avg"]
PRODUCT_TEXT_COLS = ["description_length", "description_word_count"]

# 2. Customer Demographics
CUSTOMER_DEMOGRAPHIC_COLS = ["age", "gender", "location", "age_group"]

# 3. Purchase History & CLV
PURCHASE_HISTORY_COLS = [
    "previous_purchases",
    "purchase_frequency",
    "recency_days",
    "avg_order_value",
    "customer_lifetime_value"
]

# 4. Marketing Campaign Data
MARKETING_COLS = [
    "campaigns_run",
    "ad_spend",
    "campaign_channel",
    "discount_rate",
    "discounted_price",
    "ad_spend_to_price_ratio"
]

# 5. Seasonal & Holiday Information
SEASONAL_CALENDAR_COLS = [
    "month",
    "quarter",
    "day_of_week",
    "is_weekend",
    "season",
    "is_holiday",
    "is_peak_shopping_season"
]

TEMPORAL_LAG_COLS = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_mean_7",
    "sales_rolling_std_7"
]

# ==========================================
# 6. ĐẶC TRƯNG MỞ RỘNG GIÁ TRỊ CAO (HIGH-IMPACT ADVANCED FEATURES)
# ==========================================
PRICING_FRICTION_COLS = [
    "freight_ratio",              # freight_value / (price + freight_value)
    "discount_amount",            # price * discount_rate
    "price_zscore_in_cat"         # (price - mean_cat) / (std_cat + eps)
]

LOGISTICS_COLS = [
    "is_interstate",              # 1 nếu customer_state != seller_state
    "estimated_delivery_days"     # số ngày giao dự kiến
]

CYCLICAL_TEMPORAL_COLS = [
    "month_sin", "month_cos",     # sin/cos biến đổi chu kỳ tháng (1-12)
    "dow_sin", "dow_cos"          # sin/cos biến đổi chu kỳ ngày trong tuần (0-6)
]

TRUST_BEHAVIOR_COLS = [
    "bayesian_popularity",        # rating * log1p(num_reviews)
    "is_repeat_customer"          # 1 nếu previous_purchases > 0
]

# Danh sách biến phân loại cần mã hóa (Categorical)
CATEGORICAL_COLS = [
    "category",
    "gender",
    "location",
    "age_group",
    "campaign_channel",
    "season"
]

# Danh sách biến số học cần chuẩn hóa (Numerical Features đầy đủ)
NUMERICAL_COLS = [
    # 1. Giá & Kinh tế đơn vị
    "price",
    "discount_rate",
    "discounted_price",
    "discount_amount",
    "discount_depth",
    "freight_ratio",
    "estimated_net_revenue",
    "price_to_payment_ratio",
    "price_relative_to_cat_avg",
    "price_zscore_in_cat",
    "category_sales_share",

    # 2. Vật lý 3D & Ma sát Logistics
    "product_volume_cm3",
    "product_density_g_cm3",
    "freight_cost_per_kg",
    "is_interstate",
    "estimated_delivery_days",

    # 3. Chuỗi thời gian trễ & Động lượng
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_mean_7",
    "sales_rolling_std_7",
    "sales_momentum_ratio",

    # 4. Chu kỳ lịch & Mùa vụ
    "month",
    "quarter",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_peak_shopping_season",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",

    # 5. Khách hàng RFM & Tiếp thị
    "age",
    "previous_purchases",
    "purchase_frequency",
    "recency_days",
    "avg_order_value",
    "customer_lifetime_value",
    "is_repeat_customer",
    "campaigns_run",
    "ad_spend",
    "ad_spend_to_price_ratio",
    "ad_spend_efficiency",

    # 6. Đánh giá & Tín nhiệm xã hội
    "description_length",
    "description_word_count",
    "rating",
    "num_reviews",
    "popularity_score",
    "bayesian_popularity"
]

# ==========================================
# 3. THAM SỐ HUẤN LUYỆN (MODEL PARAMETERS)
# ==========================================
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 42
CV_FOLDS = 5
USE_TEMPORAL_SPLIT = True  # True: chia theo mốc thời gian, False: random split

# Dự báo theo tuần: các đầu vào đã có trước khi tuần target bắt đầu.
WEEKLY_NUMERICAL_COLS = [
    "past_review_count", "past_review_rating", "customer_past_spend",
    "customer_past_orders", "customer_state_count",
    "price", "last_week_sales", "sales_lag_2_weeks", "sales_lag_4_weeks",
    "sales_mean_4_weeks", "sales_std_4_weeks", "last_week_units",
    "weeks_since_first_sale", "month_sin", "month_cos", "is_peak_shopping_season",
]
WEEKLY_CATEGORICAL_COLS = ["category"]

# ==========================================
# 4. HYPERPARAMETER GRIDS CHO GRIDSEARCHCV
# ==========================================
PARAM_GRIDS = {
    "linear_regression": {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_iterations": [500, 1000]
    },
    "decision_tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "random_forest": {
        "n_estimators": [30, 50, 100],
        "max_depth": [5, 8, 12, None],
        "min_samples_split": [2, 5]
    },
    "gradient_boosting": {
        "n_estimators": [30, 50, 80],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4]
    }
}
