
"""
config.py - Cấu hình trung tâm cho toàn bộ dự án Dự đoán Doanh số (Sales Predictor)
"""
from pathlib import Path

# ==========================================
# 1. ĐƯỜNG DẪN HỆ THỐNG (PATHS)
# ==========================================
# BASE_DIR là thư mục gốc của project (sales_predictor/)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Đảm bảo các thư mục luôn tồn tại
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Tên file dataset mặc định
RAW_DATA_FILE = RAW_DATA_DIR / "olist_customers_dataset.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_sales_data.csv"

# ==========================================
# 2. TOÀN BỘ ĐẶC TRƯNG (FULL FEATURE TAXONOMY THEO ĐỀ BÀI)
# ==========================================
TARGET_COLUMN = "sales"

# 1. Product Information & Popularity (Category, Price, Description, Reviews, Ratings)
PRODUCT_BASE_COLS = ["category", "price", "description"]
PRODUCT_POPULARITY_COLS = ["rating", "num_reviews", "popularity_score", "price_relative_to_cat_avg"]
PRODUCT_TEXT_COLS = ["description_length", "description_word_count"]

# 2. Customer Demographics (Age, Gender, Location)
CUSTOMER_DEMOGRAPHIC_COLS = ["age", "gender", "location", "age_group"]

# 3. Purchase History & CLV (Previous Purchases, Purchase Frequency, RFM, CLV)
PURCHASE_HISTORY_COLS = [
    "previous_purchases",
    "purchase_frequency",
    "recency_days",
    "avg_order_value",
    "customer_lifetime_value"
]

# 4. Marketing Campaign Data (Campaigns Run, Ad Spend, Discount, ROI)
MARKETING_COLS = [
    "campaigns_run",
    "ad_spend",
    "campaign_channel",
    "discount_rate",
    "discounted_price",
    "ad_spend_to_price_ratio"
]

# 5. Seasonal & Holiday Information (Month, Quarter, Holidays, Lag/Rolling)
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

# Danh sách biến phân loại cần mã hóa (Categorical)
CATEGORICAL_COLS = [
    "category",
    "gender",
    "location",
    "age_group",
    "campaign_channel",
    "season"
]

# Danh sách biến số học cần chuẩn hóa (Numerical)
NUMERICAL_COLS = [
    "price",
    "description_length",
    "description_word_count",
    "rating",
    "num_reviews",
    "popularity_score",
    "price_relative_to_cat_avg",
    "age",
    "previous_purchases",
    "purchase_frequency",
    "recency_days",
    "avg_order_value",
    "customer_lifetime_value",
    "campaigns_run",
    "ad_spend",
    "discount_rate",
    "discounted_price",
    "ad_spend_to_price_ratio",
    "month",
    "quarter",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_peak_shopping_season"
]

# ==========================================
# 3. THAM SỐ HUẤN LUYỆN (MODEL PARAMETERS)
# ==========================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
USE_TEMPORAL_SPLIT = True  # True: chia theo mốc thời gian, False: random split

# ==========================================
# 4. HYPERPARAMETER GRIDS CHO GRIDSEARCHCV
# ==========================================
PARAM_GRIDS = {
    "decision_tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5]
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5]
    },
    "xgboost": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    }
}
