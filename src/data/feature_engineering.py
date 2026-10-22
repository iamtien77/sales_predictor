"""
feature_engineering.py - Tạo và làm giàu các đặc trưng (Feature Engineering)
=============================================================================
Các hàm bên dưới xây dựng đặc trưng Product–Week từ lịch sử quan sát, không dùng
doanh số hoặc đánh giá của chính tuần đang được dự báo.
"""

import numpy as np
import pandas as pd

from src import config


def engineer_product_week_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng Product–Week chỉ từ thông tin đã có trước tuần dự báo."""
    required_columns = {
        "product_id",
        "week_start",
        "category",
        "sales",
        "units_sold",
        "order_count",
        "review_count",
        "rating_sum",
        "new_customer_states",
        "avg_price_last_observed",
    }
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        raise ValueError(f"Weekly panel is missing required columns: {missing_columns}")

    featured = panel.copy()
    featured["week_start"] = pd.to_datetime(featured["week_start"], errors="raise")
    featured = featured.sort_values(
        ["product_id", "week_start"], kind="stable"
    ).reset_index(drop=True)
    grouped = featured.groupby("product_id", sort=False, observed=True)

    lagged_sales = grouped["sales"].shift(1)
    featured["sales_lag_1"] = lagged_sales.fillna(0)
    featured["sales_lag_2"] = grouped["sales"].shift(2).fillna(0)
    featured["sales_lag_4"] = grouped["sales"].shift(4).fillna(0)
    lagged_by_product = lagged_sales.fillna(0).groupby(
        featured["product_id"], sort=False
    )
    featured["sales_rolling_mean_4"] = (
        lagged_by_product.rolling(window=4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    featured["sales_rolling_std_4"] = (
        lagged_by_product.rolling(window=4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    featured["units_lag_1"] = grouped["units_sold"].shift(1).fillna(0)
    featured["product_age_weeks"] = grouped.cumcount()

    prior_review_count = (
        grouped["review_count"].cumsum()
        .groupby(featured["product_id"], sort=False)
        .shift(1)
        .fillna(0)
    )
    prior_rating_sum = (
        grouped["rating_sum"].cumsum()
        .groupby(featured["product_id"], sort=False)
        .shift(1)
        .fillna(0)
    )
    featured["reviews_count_prior"] = prior_review_count
    featured["rating_mean_prior"] = np.divide(
        prior_rating_sum,
        prior_review_count,
        out=np.full(len(featured), np.nan, dtype=np.float64),
        where=prior_review_count.to_numpy() > 0,
    )

    featured["cumulative_customer_spend"] = (
        grouped["sales"].cumsum()
        .groupby(featured["product_id"], sort=False)
        .shift(1)
        .fillna(0)
    )
    featured["purchase_frequency_prior"] = (
        grouped["order_count"].cumsum()
        .groupby(featured["product_id"], sort=False)
        .shift(1)
        .fillna(0)
    )
    featured["geographic_dispersion_prior"] = (
        grouped["new_customer_states"].cumsum()
        .groupby(featured["product_id"], sort=False)
        .shift(1)
        .fillna(0)
    )

    featured["month"] = featured["week_start"].dt.month.astype("int8")
    featured["quarter"] = featured["week_start"].dt.quarter.astype("int8")
    featured["week_of_year"] = (
        featured["week_start"].dt.isocalendar().week.astype("int8")
    )
    featured["is_peak_shopping_season"] = (
        featured["month"].isin([11, 12]).astype("int8")
    )

    missing_features = [
        column
        for column in config.WEEKLY_NUMERICAL_FEATURES
        if column not in featured.columns
    ]
    if missing_features:
        raise RuntimeError(f"Weekly feature engineering failed: {missing_features}")
    return featured


def get_weekly_feature_columns() -> list[str]:
    """Trả về các cột đầu vào số học và phân loại theo thứ tự mô hình."""
    return config.WEEKLY_NUMERICAL_FEATURES + config.WEEKLY_CATEGORICAL_FEATURES
