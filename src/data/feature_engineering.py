"""
feature_engineering.py - Bộ Kỹ Nghệ Đặc Trưng Đa Chiều & Chọn Lọc Đặc Trưng From-Scratch
Dự án: Lab 2 - Dự đoán Doanh số Bán hàng Sản phẩm (Product Sales Predictor)

Kiến trúc chuẩn mực tương đương Lab 1 (Spam Classifier):
1. Theo dõi lát cắt ranh giới từng nhóm đặc trưng (feature_slices_) phục vụ giải thích mô hình (Explainability).
2. Class hướng đối tượng SalesFeatureEngineer: Triệt tiêu 100% rò rỉ dữ liệu (Data Leakage) giữa Train và Val/Test.
3. Bảng kiểm toán tương quan mục tiêu (audit_feature_target_correlation): Đo lường Pearson r và Spearman rho.
4. Bộ phát hiện đa cộng tuyến nghiêm trọng (detect_multicollinear_pairs): Quét các cặp đặc trưng trùng lặp (|r| >= threshold).
5. Class SalesFeatureSelectorScratch:
   - Lọc phương sai Variance Threshold
   - Điểm quan trọng kết hợp Combined Importance (Correlation + Variance + Density)
   - Thuật toán Top-K Sweeping trên Validation set đối sánh đa mục tiêu (R2, RMSE, Dimension Saving, Overfit Gap)
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config


def _resolve_category_column(df: pd.DataFrame) -> Optional[str]:
    """Xác định cột danh mục ưu tiên trong DataFrame."""
    for col in ["product_category_name", "category", "product_category_name_english"]:
        if col in df.columns:
            return col
    return None


def validate_sales_prices(df: pd.DataFrame) -> None:
    """Từ chối giá không hợp lệ; giá thiếu hoặc NaN vẫn được phép điền khuyết."""
    if "price" not in df:
        return
    supplied = df["price"].notna()
    numeric = pd.to_numeric(df["price"], errors="coerce")
    if (supplied & (numeric.isna() | ~np.isfinite(numeric) | (numeric <= 0))).any():
        raise ValueError("price phải là số hữu hạn lớn hơn 0; giá thiếu được phép điền từ tập Train.")


class WeeklySalesFeatureEngineer:
    """Tạo đặc trưng lịch cho tuần dự báo; không học lại khi suy luận."""

    def fit(self, train_df: pd.DataFrame) -> "WeeklySalesFeatureEngineer":
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise ValueError("WeeklySalesFeatureEngineer cần được fit trước khi biến đổi dữ liệu.")
        result = df.copy()
        validate_sales_prices(result)
        for column in config.WEEKLY_NUMERICAL_COLS:
            if column in result and column not in ["price", "month_sin", "month_cos"]:
                values = pd.to_numeric(result[column], errors="coerce")
                invalid = result[column].notna() & (~np.isfinite(values) | (values < 0))
                if column == "past_review_rating":
                    invalid |= values > 5
                if invalid.any():
                    raise ValueError(f"{column} phải là số hữu hạn không âm; điểm đánh giá tối đa 5.")
                result[column] = values
        if config.TIMESTAMP_COLUMN not in result:
            raise ValueError("Cần cung cấp thứ Hai của tuần dự báo trong order_purchase_timestamp.")
        dates = pd.to_datetime(result[config.TIMESTAMP_COLUMN], errors="raise")
        if dates.isna().any() or (dates.dt.dayofweek != 0).any():
            raise ValueError("Ngày bắt đầu tuần dự báo phải là thứ Hai và không được để trống.")
        if "category" not in result and "product_category_name" in result:
            result["category"] = result["product_category_name"]
        result["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        result["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
        result["is_peak_shopping_season"] = dates.dt.month.isin([11, 12]).astype(int)
        return result


class SalesFeatureEngineer:
    """
    Bộ kỹ nghệ đặc trưng hướng đối tượng (OOP) cho bài toán dự báo doanh số bán hàng.
    Tuân thủ nguyên tắc ngăn chặn triệt để rò rỉ dữ liệu (Data Leakage):
      - fit(): Chỉ học các tham số thống kê (mean, std, distribution, medians) từ tập TRAIN.
      - transform(): Áp dụng thống kê đã học lên Train/Val/Test với cơ chế Global Fallback an toàn.
      - feature_slices_: Lưu trữ tọa độ ranh giới (start, end) của từng nhóm đặc trưng.
    """

    def __init__(self, use_log_target: bool = True):
        self.use_log_target = use_log_target
        self.is_fitted_ = False

        # Quản lý lát cắt ranh giới từng nhóm đặc trưng (Feature Slices)
        self.feature_slices_: Dict[str, Tuple[int, int]] = {}
        self.engineered_feature_names_: List[str] = []

        # Thống kê phân vị & định vị danh mục học từ Train
        self.cat_col_: Optional[str] = None
        self.category_price_stats_: Dict[str, Dict[str, float]] = {}
        self.category_sales_share_: Dict[str, float] = {}
        self.global_price_mean_: float = 120.0
        self.global_price_std_: float = 50.0
        self.global_sales_share_fallback_: float = 0.01

        # Giá trị trung vị phục vụ điền khuyết / biên giới logistics an toàn
        self.median_delivery_days_: float = 23.0
        self.median_product_volume_: float = 8000.0
        self.median_product_density_: float = 0.05
        self.median_weight_g_: float = 500.0

    def fit(self, train_df: pd.DataFrame) -> "SalesFeatureEngineer":
        """
        Học các tham số thống kê chỉ từ tập huấn luyện (Train Set).
        """
        train_df = train_df.copy()
        price = pd.to_numeric(train_df["price"], errors="coerce") if "price" in train_df.columns else pd.Series(100.0, index=train_df.index)
        
        # 1. Thống kê giá toàn cục (Global Price Stats)
        valid_prices = price.dropna()
        if not valid_prices.empty:
            self.global_price_mean_ = float(valid_prices.mean())
            self.global_price_std_ = float(valid_prices.std())
            if not np.isfinite(self.global_price_std_) or self.global_price_std_ <= 1e-4:
                self.global_price_std_ = 1.0

        # 2. Thống kê theo danh mục & thị phần doanh số
        cat_col = _resolve_category_column(train_df)
        self.cat_col_ = cat_col

        if cat_col is not None and cat_col in train_df.columns:
            grouped = train_df.groupby(cat_col)["price"]
            cat_means = grouped.mean().to_dict()
            cat_stds = grouped.std().fillna(self.global_price_std_).replace(0, self.global_price_std_).to_dict()

            self.category_price_stats_ = {
                cat: {"mean": float(cat_means[cat]), "std": float(cat_stds[cat])}
                for cat in cat_means
            }

            # Frequency prior, independent of the row's own regression target.
            self.category_sales_share_ = train_df[cat_col].value_counts(normalize=True).to_dict()
            
            n_cats = max(len(self.category_price_stats_), 1)
            self.global_sales_share_fallback_ = 1.0 / n_cats

        # 3. Trung vị Logistics và kích thước vật lý
        if "order_estimated_delivery_date" in train_df.columns and config.TIMESTAMP_COLUMN in train_df.columns:
            est_date = pd.to_datetime(train_df["order_estimated_delivery_date"], errors="coerce")
            pur_date = pd.to_datetime(train_df[config.TIMESTAMP_COLUMN], errors="coerce")
            deltas = (est_date - pur_date).dt.total_seconds() / 86400.0
            valid_deltas = deltas.dropna()
            if not valid_deltas.empty:
                self.median_delivery_days_ = float(valid_deltas.median())

        if "product_length_cm" in train_df.columns and "product_height_cm" in train_df.columns and "product_width_cm" in train_df.columns:
            l = pd.to_numeric(train_df["product_length_cm"], errors="coerce").fillna(20.0).clip(lower=1.0)
            h = pd.to_numeric(train_df["product_height_cm"], errors="coerce").fillna(10.0).clip(lower=1.0)
            w = pd.to_numeric(train_df["product_width_cm"], errors="coerce").fillna(15.0).clip(lower=1.0)
            vols = l * h * w
            self.median_product_volume_ = float(vols.median())

            if "product_weight_g" in train_df.columns:
                wg = pd.to_numeric(train_df["product_weight_g"], errors="coerce").fillna(500.0).clip(lower=10.0)
                self.median_weight_g_ = float(wg.median())
                densities = wg / (vols + 1e-8)
                self.median_product_density_ = float(densities.median())

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sinh ra toàn bộ 6 nhóm đặc trưng cao cấp lên DataFrame mục tiêu
        đồng thời cập nhật chính xác feature_slices_ theo dõi ranh giới từng nhóm.
        """
        if not self.is_fitted_:
            raise ValueError("SalesFeatureEngineer must be fitted on training data first.")

        res = df.copy()
        if "category" not in res and "product_category_name" in res:
            res["category"] = res["product_category_name"]
        if "location" not in res and "customer_state" in res:
            res["location"] = res["customer_state"]
        if config.TIMESTAMP_COLUMN in res:
            dates = pd.to_datetime(res[config.TIMESTAMP_COLUMN], errors="coerce")
            res["month"] = dates.dt.month
            res["quarter"] = dates.dt.quarter
            res["day_of_week"] = dates.dt.dayofweek
            res["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
            season_map = {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn", 6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring", 11: "Spring"}
            res["season"] = dates.dt.month.map(season_map)
        feature_names: List[str] = []
        feature_slices: Dict[str, Tuple[int, int]] = {}

        # ========================================================
        # 1. KINH TẾ ĐƠN VỊ & GIÁ (UNIT ECONOMICS & PRICING)
        # ========================================================
        start_idx = len(feature_names)
        price = pd.to_numeric(res["price"], errors="coerce").fillna(self.global_price_mean_)
        freight = pd.to_numeric(res["freight_value"], errors="coerce").fillna(0.0) if "freight_value" in res.columns else pd.Series(0.0, index=res.index)
        
        res["freight_ratio"] = np.round(freight / (price + freight + 1e-8), 4)

        disc_rate = pd.to_numeric(res["discount_rate"], errors="coerce").fillna(0.0) if "discount_rate" in res.columns else pd.Series(0.0, index=res.index)
        disc_price = price * (1.0 - disc_rate)
        res["discounted_price"] = disc_price
        
        res["discount_amount"] = np.round(price * disc_rate, 2)
        res["discount_depth"] = np.round((price - disc_price) / (price + 1e-8), 4).clip(lower=0.0, upper=1.0)

        ad_spend = pd.to_numeric(res["ad_spend"], errors="coerce").fillna(0.0) if "ad_spend" in res.columns else pd.Series(0.0, index=res.index)
        res["estimated_net_revenue"] = np.round(disc_price - (ad_spend / 100.0), 2)
        res["price_to_payment_ratio"] = np.round(price / (disc_price + 1e-8), 4)

        group_1_cols = ["freight_ratio", "discount_amount", "discount_depth", "estimated_net_revenue", "price_to_payment_ratio"]
        feature_names.extend(group_1_cols)
        feature_slices["pricing_dynamics"] = (start_idx, len(feature_names))

        # ========================================================
        # 2. VẬT LÝ 3D & MA SÁT LOGISTICS (PHYSICAL & LOGISTICS)
        # ========================================================
        start_idx = len(feature_names)
        plen = pd.to_numeric(res["product_length_cm"], errors="coerce").fillna(20.0).clip(lower=1.0) if "product_length_cm" in res.columns else pd.Series(20.0, index=res.index)
        phgt = pd.to_numeric(res["product_height_cm"], errors="coerce").fillna(10.0).clip(lower=1.0) if "product_height_cm" in res.columns else pd.Series(10.0, index=res.index)
        pwid = pd.to_numeric(res["product_width_cm"], errors="coerce").fillna(15.0).clip(lower=1.0) if "product_width_cm" in res.columns else pd.Series(15.0, index=res.index)
        pwgt = pd.to_numeric(res["product_weight_g"], errors="coerce").fillna(self.median_weight_g_).clip(lower=10.0) if "product_weight_g" in res.columns else pd.Series(self.median_weight_g_, index=res.index)

        vol = plen * phgt * pwid
        res["product_volume_cm3"] = np.round(vol, 1)
        res["product_density_g_cm3"] = np.round(pwgt / (vol + 1e-8), 4)
        res["freight_cost_per_kg"] = np.round(freight / (pwgt / 1000.0 + 1e-5), 2)

        if "customer_state" in res.columns and "seller_state" in res.columns:
            res["is_interstate"] = (res["customer_state"].astype(str) != res["seller_state"].astype(str)).astype(int)
        elif "location" in res.columns:
            res["is_interstate"] = (res["location"].astype(str) != "SP").astype(int)
        else:
            res["is_interstate"] = 0

        if "order_estimated_delivery_date" in res.columns and config.TIMESTAMP_COLUMN in res.columns:
            est_date = pd.to_datetime(res["order_estimated_delivery_date"], errors="coerce")
            pur_date = pd.to_datetime(res[config.TIMESTAMP_COLUMN], errors="coerce")
            delta_days = (est_date - pur_date).dt.total_seconds() / 86400.0
            res["estimated_delivery_days"] = np.round(delta_days.fillna(self.median_delivery_days_).clip(lower=1.0, upper=90.0), 1)
        else:
            res["estimated_delivery_days"] = round(self.median_delivery_days_, 1)

        group_2_cols = ["product_volume_cm3", "product_density_g_cm3", "freight_cost_per_kg", "is_interstate", "estimated_delivery_days"]
        feature_names.extend(group_2_cols)
        feature_slices["physical_logistics"] = (start_idx, len(feature_names))

        # ========================================================
        # 3. CHUỖI THỜI GIAN TRỄ & ĐỘNG LƯỢNG (LAG & MOMENTUM)
        # ========================================================
        start_idx = len(feature_names)
        res["sales_lag_1"] = pd.to_numeric(res["sales_lag_1"], errors="coerce").fillna(0.0) if "sales_lag_1" in res.columns else pd.Series(0.0, index=res.index)
        res["sales_lag_7"] = pd.to_numeric(res["sales_lag_7"], errors="coerce").fillna(0.0) if "sales_lag_7" in res.columns else pd.Series(0.0, index=res.index)
        res["sales_rolling_mean_7"] = pd.to_numeric(res["sales_rolling_mean_7"], errors="coerce").fillna(1.0) if "sales_rolling_mean_7" in res.columns else pd.Series(1.0, index=res.index)
        res["sales_rolling_std_7"] = pd.to_numeric(res["sales_rolling_std_7"], errors="coerce").fillna(0.0) if "sales_rolling_std_7" in res.columns else pd.Series(0.0, index=res.index)
        res["sales_momentum_ratio"] = np.round((res["sales_lag_1"] + 1e-5) / (res["sales_rolling_mean_7"] + 1e-5), 4)

        group_3_cols = ["sales_lag_1", "sales_lag_7", "sales_rolling_mean_7", "sales_rolling_std_7", "sales_momentum_ratio"]
        feature_names.extend(group_3_cols)
        feature_slices["lag_momentum"] = (start_idx, len(feature_names))

        # ========================================================
        # 4. CHU KỲ LỊCH & MÙA VỤ (CYCLICAL & SEASONALITY)
        # ========================================================
        start_idx = len(feature_names)
        month = pd.to_numeric(res["month"], errors="coerce").fillna(1.0) if "month" in res.columns else pd.Series(1.0, index=res.index)
        dow = pd.to_numeric(res["day_of_week"], errors="coerce").fillna(0.0) if "day_of_week" in res.columns else pd.Series(0.0, index=res.index)

        res["month_sin"] = np.round(np.sin(2.0 * np.pi * month / 12.0), 4)
        res["month_cos"] = np.round(np.cos(2.0 * np.pi * month / 12.0), 4)
        res["dow_sin"] = np.round(np.sin(2.0 * np.pi * dow / 7.0), 4)
        res["dow_cos"] = np.round(np.cos(2.0 * np.pi * dow / 7.0), 4)

        res["is_weekend"] = pd.to_numeric(res["is_weekend"], errors="coerce").fillna(0).astype(int) if "is_weekend" in res.columns else ((dow >= 5).astype(int))
        res["is_holiday"] = pd.to_numeric(res["is_holiday"], errors="coerce").fillna(0).astype(int) if "is_holiday" in res.columns else 0
        res["is_peak_shopping_season"] = pd.to_numeric(res["is_peak_shopping_season"], errors="coerce").fillna(0).astype(int) if "is_peak_shopping_season" in res.columns else ((month.isin([11, 12])).astype(int))

        group_4_cols = ["month_sin", "month_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "is_peak_shopping_season"]
        feature_names.extend(group_4_cols)
        feature_slices["calendar_cyclical"] = (start_idx, len(feature_names))

        # ========================================================
        # 5. KHÁCH HÀNG RFM & TIẾP THỊ (RFM & MARKETING EFFICIENCY)
        # ========================================================
        start_idx = len(feature_names)
        res["customer_recency_days"] = pd.to_numeric(res["recency_days"], errors="coerce").fillna(30.0) if "recency_days" in res.columns else pd.Series(30.0, index=res.index)
        res["customer_lifetime_value"] = pd.to_numeric(res["customer_lifetime_value"], errors="coerce").fillna(100.0) if "customer_lifetime_value" in res.columns else pd.Series(100.0, index=res.index)
        
        prev_p = pd.to_numeric(res["previous_purchases"], errors="coerce").fillna(0.0) if "previous_purchases" in res.columns else pd.Series(0.0, index=res.index)
        res["is_repeat_customer"] = (prev_p > 0).astype(int)

        res["ad_spend_to_price_ratio"] = np.round(ad_spend / (price + 1e-8), 4)
        res["ad_spend_efficiency"] = np.round((price / (ad_spend + 1.0)).clip(lower=0.0, upper=1000.0), 4)

        group_5_cols = ["customer_recency_days", "customer_lifetime_value", "is_repeat_customer", "ad_spend_to_price_ratio", "ad_spend_efficiency"]
        feature_names.extend(group_5_cols)
        feature_slices["rfm_marketing"] = (start_idx, len(feature_names))

        # ========================================================
        # 6. ĐỊNH VỊ DANH MỤC & BAYES (CATEGORY & BAYESIAN SOCIAL)
        # ========================================================
        start_idx = len(feature_names)
        cat_col = self.cat_col_ if (self.cat_col_ is not None and self.cat_col_ in res.columns) else _resolve_category_column(res)
        
        if cat_col is not None and cat_col in res.columns:
            cat_series = res[cat_col].astype(str)
            means = cat_series.map(lambda c: self.category_price_stats_.get(c, {}).get("mean", self.global_price_mean_)).fillna(self.global_price_mean_)
            stds = cat_series.map(lambda c: self.category_price_stats_.get(c, {}).get("std", self.global_price_std_)).fillna(self.global_price_std_)
            
            res["price_zscore_in_cat"] = np.round((price - means) / (stds + 1e-8), 3)
            res["price_relative_to_cat_avg"] = price / means.replace(0, 1.0)
            res["category_sales_share"] = np.round(cat_series.map(lambda c: self.category_sales_share_.get(c, self.global_sales_share_fallback_)).fillna(self.global_sales_share_fallback_), 4)
        else:
            res["price_zscore_in_cat"] = 0.0
            res["category_sales_share"] = round(self.global_sales_share_fallback_, 4)

        rating = pd.to_numeric(res["rating"], errors="coerce").fillna(4.0) if "rating" in res.columns else pd.Series(4.0, index=res.index)
        num_reviews = pd.to_numeric(res["num_reviews"], errors="coerce").fillna(1.0) if "num_reviews" in res.columns else pd.Series(1.0, index=res.index)
        res["bayesian_popularity"] = np.round(rating * np.log1p(np.maximum(num_reviews, 0.0)), 3)

        group_6_cols = ["price_zscore_in_cat", "category_sales_share", "bayesian_popularity"]
        feature_names.extend(group_6_cols)
        feature_slices["category_bayesian"] = (start_idx, len(feature_names))

        # ========================================================
        # 7. BIẾN ĐỔI KHÔNG GIAN MỤC TIÊU (LOG-SALES TRANSFORMATION)
        # ========================================================
        if self.use_log_target and config.TARGET_COLUMN in res.columns:
            sales = np.maximum(pd.to_numeric(res[config.TARGET_COLUMN], errors="coerce").fillna(0.0), 0.0)
            res[config.LOG_TARGET_COLUMN] = np.round(np.log1p(sales), 4)

        self.engineered_feature_names_ = feature_names
        self.feature_slices_ = feature_slices
        return res

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Học tham số trên tập dữ liệu và biến đổi ngay lập tức."""
        return self.fit(df).transform(df)

    def get_feature_slice(self, group_name: str) -> Tuple[int, int]:
        """Lấy tọa độ ranh giới (start, end) của một nhóm đặc trưng."""
        if group_name not in self.feature_slices_:
            raise KeyError(f"Nhóm đặc trưng {group_name} không tồn tại. Các nhóm hợp lệ: {list(self.feature_slices_.keys())}")
        return self.feature_slices_[group_name]

    def list_feature_groups(self) -> List[str]:
        """Danh sách các nhóm đặc trưng kỹ nghệ."""
        return list(self.feature_slices_.keys())


# =========================================================================
# KIỂM TOÁN THỐNG KÊ NÂNG CAO (STATISTICAL FEATURE AUDIT)
# =========================================================================

def audit_feature_target_correlation(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "log_sales"
) -> pd.DataFrame:
    """
    Kiểm toán mức độ tương quan tuyến tính (Pearson r) và phi tuyến hạng (Spearman rho)
    giữa các thuộc tính đặc trưng và biến mục tiêu doanh số.
    
    Phân loại mức độ ảnh hưởng:
      - Mạnh (Strong): |r| >= 0.30
      - Trung bình (Moderate): 0.10 <= |r| < 0.30
      - Yếu / Nhiễu (Weak / Noise): |r| < 0.10
    """
    if target_col not in df.columns:
        if config.TARGET_COLUMN in df.columns:
            target_col = config.TARGET_COLUMN
        else:
            raise KeyError(f"Biến mục tiêu {target_col} không tồn tại trong DataFrame.")

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std <= 1e-8:
        y_std = 1.0

    if feature_cols is None:
        # Lấy tất cả cột số học trừ biến định danh / mục tiêu
        num_candidates = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["sales", "log_sales", "order_id", "product_id", "customer_id", "customer_unique_id", "seller_id"]
        ]
        feature_cols = num_candidates

    records = []
    n = len(df)

    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        x_mean = float(np.mean(x))
        x_std = float(np.std(x))
        if x_std <= 1e-8:
            x_std = 1.0

        # Pearson r
        cov = float(np.mean((x - x_mean) * (y - y_mean)))
        pearson_r = round(cov / (x_std * y_std), 4)

        # Spearman rho (tương quan trên thứ hạng)
        rank_x = pd.Series(x).rank().values
        rank_y = pd.Series(y).rank().values
        rx_std = float(np.std(rank_x)) if np.std(rank_x) > 0 else 1.0
        ry_std = float(np.std(rank_y)) if np.std(rank_y) > 0 else 1.0
        spearman_rho = round(float(np.mean((rank_x - np.mean(rank_x)) * (rank_y - np.mean(rank_y)))) / (rx_std * ry_std), 4)

        abs_r = abs(pearson_r)
        if abs_r >= 0.30:
            impact = "Mạnh (Strong)"
        elif abs_r >= 0.10:
            impact = "Trung bình (Moderate)"
        else:
            impact = "Yếu / Nhiễu (Weak / Noise)"

        direction = "Thuận (Kích cầu +)" if pearson_r >= 0 else "Nghịch (Rào cản -)"

        records.append({
            "Thuộc tính đặc trưng": col,
            "Tương quan Pearson (r)": pearson_r,
            "Tương quan Spearman (rho)": spearman_rho,
            "Độ lớn (|r|)": abs_r,
            "Chiều hướng tác động": direction,
            "Mức độ ảnh hưởng": impact
        })

    audit_df = pd.DataFrame(records).sort_values(by="Độ lớn (|r|)", ascending=False).reset_index(drop=True)
    return audit_df


def detect_multicollinear_pairs(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    threshold: float = 0.85,
    target_col: str = "log_sales"
) -> pd.DataFrame:
    """
    Quét và phát hiện các cặp đặc trưng bị đa cộng tuyến nghiêm trọng (|r| >= threshold).
    Chỉ ra biến có tương quan mục tiêu cao hơn để giữ lại và biến dư thừa nên loại bỏ.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["sales", "log_sales", "order_id", "product_id", "customer_id", "customer_unique_id", "seller_id"]
        ]

    # Tính tương quan của từng biến với mục tiêu nếu có
    target_corrs = {}
    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values
        y_std = float(np.std(y)) if np.std(y) > 0 else 1.0
        for c in feature_cols:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
            x_std = float(np.std(x)) if np.std(x) > 0 else 1.0
            r = float(np.mean((x - np.mean(x)) * (y - np.mean(y)))) / (x_std * y_std)
            target_corrs[c] = abs(r)

    sub_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr_matrix = sub_df.corr().abs()

    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_val = float(corr_matrix.iloc[i, j])
            if r_val >= threshold and math.isfinite(r_val):
                f1, f2 = cols[i], cols[j]
                r1_y = target_corrs.get(f1, 0.0)
                r2_y = target_corrs.get(f2, 0.0)

                recommended_keep = f1 if r1_y >= r2_y else f2
                recommended_drop = f2 if r1_y >= r2_y else f1

                pairs.append({
                    "Đặc trưng 1 (Feature 1)": f1,
                    "Đặc trưng 2 (Feature 2)": f2,
                    "Hệ số tương quan cặp (|r|)": round(r_val, 4),
                    "|r| F1 với Target": round(r1_y, 4),
                    "|r| F2 với Target": round(r2_y, 4),
                    "Khuyến nghị giữ lại": recommended_keep,
                    "Khuyến nghị lược bỏ": recommended_drop,
                    "Lý do kinh tế": f"Đa cộng tuyến cao ({r_val:.2f}); giữ biến có lực dự báo mục tiêu mạnh hơn"
                })

    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(by="Hệ số tương quan cặp (|r|)", ascending=False).reset_index(drop=True)
    else:
        df_pairs = pd.DataFrame(columns=[
            "Đặc trưng 1 (Feature 1)", "Đặc trưng 2 (Feature 2)", "Hệ số tương quan cặp (|r|)",
            "|r| F1 với Target", "|r| F2 với Target", "Khuyến nghị giữ lại", "Khuyến nghị lược bỏ", "Lý do kinh tế"
        ])
    return df_pairs


# =========================================================================
# BỘ CHỌN LỌC ĐẶC TRƯNG HỒI QUY TỰ XÂY DỰNG (SALES FEATURE SELECTOR SCRATCH)
# =========================================================================

class SalesFeatureSelectorScratch:
    """
    Bộ chọn lọc đặc trưng cho bài toán Hồi quy doanh số tự viết từ đầu (From-Scratch).
    
    Nguyên lý:
    1. Variance Threshold: Lọc bỏ các biến có phương sai gần như bằng 0 (biến hằng số/nhiễu).
    2. Combined Importance: Điểm quan trọng tổng hợp:
       - 60% Tương quan Pearson chuẩn hóa với biến mục tiêu (Pearson Target Correlation)
       - 25% Độ phân tán phương sai chuẩn hóa (Normalized Variance)
       - 15% Mật độ hoạt động (Activity Density / Non-zero ratio)
    3. Top-K Sweeping: Thử nghiệm dải số lượng đặc trưng K trên tập Validation set bằng
       mô hình hồi quy cơ sở From-Scratch để tìm điểm cân bằng đa mục tiêu:
       Tối đa hóa R2, tối thiểu hóa RMSE, tối đa hóa % Tiết kiệm số chiều, và triệt tiêu Overfitting Gap.
    """

    def __init__(self, variance_threshold: float = 1e-4, top_k: Optional[int] = None, alpha: float = 1.0):
        self.variance_threshold = variance_threshold
        self.top_k = top_k
        self.alpha = alpha

        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_importance_scores_: Optional[np.ndarray] = None
        self.ranked_indices_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    @staticmethod
    def _ensure_csr(X: Any) -> sp.csr_matrix:
        """Đảm bảo ma trận đầu vào luôn ở định dạng scipy.sparse.csr_matrix chuẩn."""
        if sp.issparse(X):
            return X.tocsr()
        return sp.csr_matrix(np.asarray(X, dtype=np.float64))

    def fit_variance_threshold(self, X: Any, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Lọc phương sai cho ma trận đặc trưng thưa hoặc đặc."""
        th = self.variance_threshold if threshold is None else threshold
        X_csr = self._ensure_csr(X)
        means = np.asarray(X_csr.mean(axis=0)).ravel()
        mean_sq = np.asarray(X_csr.power(2).mean(axis=0)).ravel()
        variances = np.maximum(mean_sq - means ** 2, 0.0)

        mask = variances > float(th)
        if not mask.any() and variances.size > 0:
            mask[int(np.argmax(variances))] = True
        return mask, variances

    def compute_correlation_scores(self, X: Any, y: np.ndarray) -> np.ndarray:
        """Tính điểm tương quan Pearson giữa từng cột X và vector mục tiêu y."""
        X_csr = self._ensure_csr(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n = X_csr.shape[0]

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr))
        if y_std <= 1e-8:
            y_std = 1.0
        y_cent = y_arr - y_mean

        col_means = np.asarray(X_csr.mean(axis=0)).ravel()
        col_mean_sq = np.asarray(X_csr.power(2).mean(axis=0)).ravel()
        col_stds = np.sqrt(np.maximum(col_mean_sq - col_means ** 2, 0.0))
        col_stds[col_stds <= 1e-8] = 1.0

        # Covariance = X^T @ y_cent / n
        cov = X_csr.T.dot(y_cent) / max(n, 1)
        r = np.abs(cov / (col_stds * y_std))
        return np.where(np.isfinite(r), r, 0.0)

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        min_v = float(np.nanmin(arr)) if arr.size > 0 else 0.0
        max_v = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        denom = max(max_v - min_v, 1e-12)
        return (arr - min_v) / denom

    def compute_combined_importance(self, X_train: Any, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính điểm quan trọng kết hợp:
        60% Tương quan Pearson + 25% Phương sai + 15% Mật độ hoạt động.
        """
        X_csr = self._ensure_csr(X_train)
        y_arr = np.asarray(y_train, dtype=np.float64).ravel()

        # 1. Điểm tương quan mục tiêu
        corr_raw = self.compute_correlation_scores(X_csr, y_arr)
        corr_norm = self._normalize_scores(corr_raw)

        # 2. Điểm phương sai
        _, variances = self.fit_variance_threshold(X_csr, threshold=0.0)
        var_norm = self._normalize_scores(np.log1p(variances))

        # 3. Mật độ hoạt động (Tỷ lệ phần tử khác 0)
        nnz_per_col = np.diff(X_csr.tocsc().indptr)
        density_norm = self._normalize_scores(nnz_per_col / max(X_csr.shape[0], 1))

        combined = 0.60 * corr_norm + 0.25 * var_norm + 0.15 * density_norm
        ranked_indices = np.argsort(-combined)
        return combined, ranked_indices

    def sweep_top_k(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any,
        y_val: np.ndarray,
        candidate_k_list: Optional[List[int]] = None,
        r2_target: float = 0.85,
        eval_model_fn: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Quét dải Top-K đặc trưng trên Validation set để tìm điểm cân bằng tối ưu.
        Đánh giá bằng thuật toán Hồi quy Ridge From-Scratch: w = (X^T X + alpha*I)^(-1) X^T y.
        """
        X_tr_csr = self._ensure_csr(X_train)
        X_va_csr = self._ensure_csr(X_val)
        y_tr = np.asarray(y_train, dtype=np.float64).ravel()
        y_va = np.asarray(y_val, dtype=np.float64).ravel()
        full_p = X_tr_csr.shape[1]

        # Tính toán ranked indices nếu chưa có
        if self.ranked_indices_ is None:
            _, self.ranked_indices_ = self.compute_combined_importance(X_tr_csr, y_tr)

        ranked = self.ranked_indices_

        if candidate_k_list is None:
            candidate_k_list = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, full_p]
        candidate_k_list = sorted({min(int(k), full_p) for k in candidate_k_list if k > 0})

        sweep_records = []
        y_va_mean = float(np.mean(y_va))
        ss_tot_va = float(np.sum((y_va - y_va_mean) ** 2))
        y_tr_mean = float(np.mean(y_tr))
        ss_tot_tr = float(np.sum((y_tr - y_tr_mean) ** 2))
        sigma_y_va = float(np.std(y_va)) if np.std(y_va) > 0 else 1.0

        for top_k in candidate_k_list:
            selected_idx = ranked[:top_k]
            X_tr_sub = X_tr_csr[:, selected_idx]
            X_va_sub = X_va_csr[:, selected_idx]

            if eval_model_fn is not None:
                metrics = eval_model_fn(X_tr_sub, y_tr, X_va_sub, y_va)
                r2_va = float(metrics.get("r2_val", 0.0))
                r2_tr = float(metrics.get("r2_train", r2_va))
                rmse_va = float(metrics.get("rmse_val", 0.0))
                mae_va = float(metrics.get("mae_val", 0.0))
            else:
                # Fast Ridge Regression From-Scratch
                p_sub = top_k
                XtX = (X_tr_sub.T.dot(X_tr_sub)).toarray()
                Xty = X_tr_sub.T.dot(y_tr)
                A = XtX + self.alpha * np.eye(p_sub)
                w = np.linalg.solve(A, Xty)

                y_pred_va = X_va_sub.dot(w)
                y_pred_tr = X_tr_sub.dot(w)

                ss_res_va = float(np.sum((y_va - y_pred_va) ** 2))
                r2_va = 1.0 - (ss_res_va / (ss_tot_va + 1e-8))

                ss_res_tr = float(np.sum((y_tr - y_pred_tr) ** 2))
                r2_tr = 1.0 - (ss_res_tr / (ss_tot_tr + 1e-8))

                rmse_va = float(np.sqrt(np.mean((y_va - y_pred_va) ** 2)))
                mae_va = float(np.mean(np.abs(y_va - y_pred_va)))

            overfit_gap = max(0.0, r2_tr - r2_va)
            dim_saving_pct = float((full_p - top_k) / full_p * 100.0)
            target_met = bool(r2_va >= r2_target)

            # Điểm cân bằng đa mục tiêu (Balance Score)
            balance_score = (
                r2_va * 0.50
                + (dim_saving_pct / 100.0) * 0.30
                - overfit_gap * 0.40
                - (rmse_va / sigma_y_va) * 0.10
            )

            sweep_records.append({
                "Top-K": top_k,
                "Tiết kiệm số chiều (%)": round(dim_saving_pct, 1),
                "R2 Tập Kiểm thử (Val R2)": round(r2_va, 4),
                "R2 Tập Huấn luyện (Train R2)": round(r2_tr, 4),
                "Sai số RMSE (Val RMSE)": round(rmse_va, 4),
                "Sai số MAE (Val MAE)": round(mae_va, 4),
                "Khoảng cách Quá khớp (Overfit Gap)": round(overfit_gap, 4),
                "Đạt mục tiêu R2 (>= 0.85)": "ĐẠT" if target_met else "CHƯA",
                "Điểm Cân bằng (Balance Score)": round(balance_score, 4),
                "selected_indices": selected_idx
            })

        df_sweep = pd.DataFrame(sweep_records)

        # Chọn cấu hình tốt nhất: ưu tiên đạt mục tiêu R2, sau đó điểm cân bằng cao nhất
        met_df = df_sweep[df_sweep["Đạt mục tiêu R2 (>= 0.85)"] == "ĐẠT"]
        if not met_df.empty:
            best_row = met_df.sort_values(by=["Điểm Cân bằng (Balance Score)", "R2 Tập Kiểm thử (Val R2)"], ascending=False).iloc[0]
        else:
            best_row = df_sweep.sort_values(by=["R2 Tập Kiểm thử (Val R2)", "Điểm Cân bằng (Balance Score)"], ascending=False).iloc[0]

        return best_row.to_dict(), df_sweep

    def fit(self, X_train: Any, y_train: np.ndarray, top_k: Optional[int] = None) -> "SalesFeatureSelectorScratch":
        """Fit bộ chọn lọc đặc trưng trên tập huấn luyện."""
        X_csr = self._ensure_csr(X_train)
        y_arr = np.asarray(y_train, dtype=np.float64).ravel()

        self.feature_importance_scores_, self.ranked_indices_ = self.compute_combined_importance(X_csr, y_arr)
        
        k = top_k if top_k is not None else (self.top_k if self.top_k is not None else 60)
        k = min(k, X_csr.shape[1])
        self.selected_indices_ = self.ranked_indices_[:k]
        self.is_fitted_ = True
        return self

    def transform(self, X: Any) -> sp.csr_matrix:
        """Cắt lọc các đặc trưng theo danh sách chỉ số đã được chọn."""
        if not self.is_fitted_ or self.selected_indices_ is None:
            raise ValueError("SalesFeatureSelectorScratch chưa được fit trên Train.")
        X_csr = self._ensure_csr(X)
        return X_csr[:, self.selected_indices_]

    def fit_transform(self, X_train: Any, y_train: np.ndarray, top_k: Optional[int] = None) -> sp.csr_matrix:
        """Fit và biến đổi lọc ma trận ngay lập tức."""
        return self.fit(X_train, y_train, top_k=top_k).transform(X_train)


# =========================================================================
# CÁC HÀM TIỆN ÍCH TƯƠNG THÍCH NGƯỢC (STANDALONE HELPER FUNCTIONS)
# =========================================================================

def add_pricing_and_freight_features(df: pd.DataFrame) -> pd.DataFrame:
    """1. Kỹ nghệ tương tác Giá, Chiết khấu và Phí vận chuyển."""
    fe = SalesFeatureEngineer(use_log_target=False)
    fe.fit(df)
    return fe.transform(df)


def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    """2. Kỹ nghệ ma sát Logistics và Vận chuyển liên bang."""
    fe = SalesFeatureEngineer(use_log_target=False)
    fe.fit(df)
    return fe.transform(df)


def add_cyclical_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """3. Biến đổi lượng giác Sin/Cos cho các đặc trưng chu kỳ lịch."""
    fe = SalesFeatureEngineer(use_log_target=False)
    fe.fit(df)
    return fe.transform(df)


def add_reputation_and_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """4. Kỹ nghệ tín nhiệm xã hội Bayes và hành vi khách hàng."""
    fe = SalesFeatureEngineer(use_log_target=False)
    fe.fit(df)
    return fe.transform(df)


def add_target_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """5. Biến đổi Logarit cho biến mục tiêu doanh số log_sales = log(1 + sales)."""
    df = df.copy()
    if config.TARGET_COLUMN in df.columns:
        sales = np.maximum(pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce").fillna(0.0), 0.0)
        df[config.LOG_TARGET_COLUMN] = np.round(np.log1p(sales), 4)
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm điều phối thực thi trọn vẹn toàn bộ 6 nhóm kỹ nghệ đặc trưng đa chiều.
    Tương thích ngược 100% với các pipeline và notebook hiện tại.
    """
    print("[FEATURES] Khởi tạo SalesFeatureEngineer và trích xuất 6 nhóm đặc trưng...")
    fe = SalesFeatureEngineer(use_log_target=True)
    return fe.fit_transform(df)
