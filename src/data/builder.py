"""
builder.py - Hợp nhất dữ liệu Olist E-commerce thành tập dữ liệu bán hàng chuẩn
================================================================================
Mục đích:
  - Đọc các bảng quan hệ từ data/raw/ (order_items, orders, products, reviews, customers, translation)
  - Lọc các đơn hàng hoàn thành (delivered) và nối các thông tin sản phẩm, khách hàng, đánh giá
  - Xây dựng biến mục tiêu 'sales' và 5 nhóm đặc trưng chuẩn hóa theo src/config.py
  - Lưu kết quả ra data/raw/olist_merged_sales_dataset.csv phục vụ toàn bộ pipeline
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cho phép chạy độc lập hoặc import từ package
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config


def clean_weekly_source_data(items: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra và làm sạch dòng sản phẩm trước khi tổng hợp tuần; không học thống kê."""
    required = {"product_id", "price", config.TIMESTAMP_COLUMN}
    if not required.issubset(items):
        raise ValueError(f"Thiếu cột nguồn để tạo dữ liệu tuần: {sorted(required - set(items.columns))}")
    data = items.copy()
    if "order_status" in data:
        data = data.loc[data["order_status"].eq("delivered")].copy()
    data[config.TIMESTAMP_COLUMN] = pd.to_datetime(data[config.TIMESTAMP_COLUMN], errors="raise")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["product_id", "price", config.TIMESTAMP_COLUMN])
    data = data.loc[(data["price"] > 0) & np.isfinite(data["price"])].copy()
    if data.empty:
        raise ValueError("Không còn giao dịch bán hàng hợp lệ.")
    if {"order_id", "order_item_id"}.issubset(data):
        data = data.drop_duplicates(["order_id", "order_item_id"])
    return data


def build_weekly_sales_dataset(items: pd.DataFrame, reviews: pd.DataFrame = None) -> pd.DataFrame:
    """Mỗi dòng là một sản phẩm-tuần; đầu vào chỉ dùng những tuần trước target.

    Giữ tuần không bán sau lần bán đầu đến tuần đầy đủ cuối cùng.
    Doanh thu là tổng giá các item, không gồm phí vận chuyển và dữ liệu mô phỏng.
    """
    data = clean_weekly_source_data(items)
    dates = data[config.TIMESTAMP_COLUMN]
    data["week"] = dates.dt.to_period("W-SUN").dt.start_time
    # Tuần đầu và cuối có thể chưa được thu thập đủ ngày.
    first_complete = data["week"].min() + pd.Timedelta(weeks=1)
    last_complete = data["week"].max() - pd.Timedelta(weeks=1)
    if last_complete <= first_complete:
        raise ValueError("Cần quan sát ít nhất bốn tuần lịch để tạo dữ liệu dự báo.")
    data = data.loc[data["week"].between(first_complete, last_complete)].copy()
    category = "product_category_name" if "product_category_name" in data else "category"
    if category not in data:
        data[category] = "other"
    weekly = data.groupby(["product_id", "week"], as_index=False).agg(
        revenue=("price", "sum"), units=("price", "size"), price=("price", "mean"))
    first_weeks = weekly.groupby("product_id")["week"].min()
    weeks = pd.date_range(first_complete, last_complete, freq="W-MON")
    index = pd.MultiIndex.from_product([first_weeks.index, weeks], names=["product_id", "week"])
    panel = weekly.set_index(["product_id", "week"]).reindex(index).reset_index()
    panel = panel.loc[panel["week"] >= panel["product_id"].map(first_weeks)].copy()
    panel[["revenue", "units"]] = panel[["revenue", "units"]].fillna(0.0)
    panel["price"] = panel.groupby("product_id")["price"].ffill()
    groups = panel.groupby("product_id", sort=False)
    panel["sales"] = panel["revenue"]
    panel["last_week_sales"] = groups["revenue"].shift(1)
    panel["sales_lag_2_weeks"] = groups["revenue"].shift(2).fillna(0.0)
    panel["sales_lag_4_weeks"] = groups["revenue"].shift(4).fillna(0.0)
    panel["sales_mean_4_weeks"] = groups["revenue"].transform(lambda s: s.shift().rolling(4, min_periods=1).mean())
    panel["sales_std_4_weeks"] = groups["revenue"].transform(lambda s: s.shift().rolling(4, min_periods=1).std(ddof=0))
    panel["last_week_units"] = groups["units"].shift(1)
    panel["price"] = groups["price"].shift(1)
    panel["weeks_since_first_sale"] = (panel["week"] - panel["product_id"].map(first_weeks)).dt.days // 7
    categories = data.sort_values(config.TIMESTAMP_COLUMN).drop_duplicates("product_id").set_index("product_id")[category]
    panel["category"] = panel["product_id"].map(categories).fillna("other")
    panel[config.TIMESTAMP_COLUMN] = panel["week"]
    # Tuần bán đầu làm lịch sử cho dự báo đầu tiên; không tạo target chưa quan sát.
    panel = panel.loc[panel["last_week_sales"].notna()].copy()
    keep = ["product_id", config.TIMESTAMP_COLUMN, "category", "price", "sales",
            "last_week_sales", "sales_lag_2_weeks", "sales_lag_4_weeks",
            "sales_mean_4_weeks", "sales_std_4_weeks", "last_week_units", "weeks_since_first_sale"]
    result = panel[keep].sort_values([config.TIMESTAMP_COLUMN, "product_id"]).reset_index(drop=True)
    return add_historical_features(result, data, reviews)


def add_historical_features(panel, data, reviews=None):
    """Ghép lịch sử đã biết trước tuần đích; không dùng đánh giá tại lúc mua hàng."""
    result = panel.copy()
    # Tổng hợp khách hàng theo tuần trước khi ghép lại sản phẩm để tránh nhân dòng.
    if "customer_unique_id" in data:
        customer = data.groupby(["customer_unique_id", "week"], as_index=False).agg(
            spend=("price", "sum"), orders=("order_id", "nunique"))
        customer = customer.sort_values(["customer_unique_id", "week"])
        customer["customer_past_spend"] = customer.groupby("customer_unique_id")["spend"].cumsum() - customer.spend
        customer["customer_past_orders"] = customer.groupby("customer_unique_id")["orders"].cumsum() - customer.orders
        enriched = data.merge(customer, on=["customer_unique_id", "week"], validate="many_to_one")
        history = enriched.groupby(["product_id", "week"], as_index=False).agg(
            customer_past_spend=("customer_past_spend", "mean"),
            customer_past_orders=("customer_past_orders", "mean"))
        if "customer_state" in enriched:
            states = enriched.groupby(["product_id", "week"])["customer_state"].nunique().rename("customer_state_count")
            history = history.merge(states, on=["product_id", "week"])
        history = history.rename(columns={"week": "history_week"})
        result = pd.merge_asof(result.sort_values(config.TIMESTAMP_COLUMN), history.sort_values("history_week"),
                               left_on=config.TIMESTAMP_COLUMN, right_on="history_week", by="product_id", allow_exact_matches=False)
        result = result.drop(columns="history_week")
    for col in ["customer_past_spend", "customer_past_orders", "customer_state_count"]:
        if col not in result:
            result[col] = 0.0
        result[col] = result[col].fillna(0.0)
    # Ngày trả lời là mốc bảo thủ để coi điểm đánh giá đã sẵn có.
    # Khuyến nghị truyền reviews tường minh. Nếu reviews=None, hàm sẽ kiểm tra tệp dự phòng tại RAW_DATA_DIR;
    # nếu không có, các trường review sẽ được điền 0.0 an toàn mà không làm sập pipeline.
    if reviews is None:
        path = config.RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
        if path.exists() and "order_id" in data:
            try:
                from src.utils.logger import setup_logger
                setup_logger("sales_predictor").info(
                    f"[DATA AUDIT] Tham số reviews=None; tự động nạp dữ liệu đánh giá từ tệp chuẩn: {path}"
                )
            except Exception:
                pass
            reviews = pd.read_csv(path)
    if reviews is not None and "order_id" in data:
        events = reviews.merge(data[["order_id", "product_id"]].drop_duplicates(), on="order_id")
        events["available_at"] = pd.to_datetime(events["review_answer_timestamp"], errors="coerce")
        events["review_score"] = pd.to_numeric(events["review_score"], errors="coerce")
        events = events.dropna(subset=["available_at", "review_score"])
        events = events.loc[events.review_score.between(1, 5)].drop_duplicates(["review_id", "product_id"])
        events = events.groupby(["product_id", "available_at"], as_index=False).agg(
            score_sum=("review_score", "sum"), review_count=("review_score", "size"))
        events = events.sort_values(["product_id", "available_at"])
        events["past_review_count"] = events.groupby("product_id").review_count.cumsum()
        events["past_review_rating"] = events.groupby("product_id").score_sum.cumsum() / events.past_review_count
        if not events.empty:
            result = pd.merge_asof(result.sort_values(config.TIMESTAMP_COLUMN),
                events[["product_id", "available_at", "past_review_count", "past_review_rating"]].sort_values("available_at"),
                left_on=config.TIMESTAMP_COLUMN, right_on="available_at", by="product_id", allow_exact_matches=False)
            result = result.drop(columns="available_at")
    for col in ["past_review_count", "past_review_rating"]:
        if col not in result:
            result[col] = 0.0
        result[col] = result[col].fillna(0.0)
    return result.sort_values([config.TIMESTAMP_COLUMN, "product_id"]).reset_index(drop=True)


# ==============================================================================
# [LEGACY BUILDER / ĐỒ ÁN CŨ]
# Hàm dưới đây thuộc phiên bản đồ án ban đầu (dự báo theo từng dòng giao dịch đơn lẻ,
# có sinh dữ liệu nhân khẩu học mô phỏng age, gender, marketing).
# KHÔNG SỬ DỤNG CHO LUỒNG DỰ BÁO DOANH THU TUẦN CHÍNH THỨC HIỆN TẠI (build_weekly_sales_dataset).
# Giữ lại nhằm đảm bảo tính tương thích ngược cho các notebook và bài kiểm thử di sản.
# ==============================================================================
def build_merged_sales_dataset(output_path: Path = None, random_state: int = 42) -> pd.DataFrame:
    """
    [LEGACY] Hợp nhất các bảng Olist và sinh các nhóm đặc trưng theo cấu trúc cũ.
    
    Args:
        output_path: Đường dẫn lưu file CSV đầu ra (mặc định là config.RAW_DATA_FILE)
        random_state: Seed tái lập kết quả ngẫu nhiên
        
    Returns:
        pd.DataFrame: DataFrame hợp nhất hoàn chỉnh
    """
    if output_path is None:
        output_path = config.RAW_DATA_FILE

    print("[1/6] Nạp các bảng thô từ Olist...")
    raw_dir = config.RAW_DATA_DIR
    items_path = raw_dir / "olist_order_items_dataset.csv"
    orders_path = raw_dir / "olist_orders_dataset.csv"
    products_path = raw_dir / "olist_products_dataset.csv"
    reviews_path = raw_dir / "olist_order_reviews_dataset.csv"
    customers_path = raw_dir / "olist_customers_dataset.csv"
    trans_path = raw_dir / "product_category_name_translation.csv"

    for p in [items_path, orders_path, products_path, reviews_path, customers_path]:
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file bắt buộc: {p}")

    items = pd.read_csv(items_path)
    orders = pd.read_csv(orders_path)
    products = pd.read_csv(products_path)
    reviews = pd.read_csv(reviews_path)
    customers = pd.read_csv(customers_path)

    # 1. Danh mục sản phẩm (Anh hóa category)
    if trans_path.exists():
        trans = pd.read_csv(trans_path)
        products = products.merge(trans, on="product_category_name", how="left")
        products["category"] = (
            products["product_category_name_english"]
            .fillna(products["product_category_name"])
            .fillna("other")
        )
    else:
        products["category"] = products["product_category_name"].fillna("other")

    # 2. Lọc đơn hàng hợp lệ & parse ngày
    print("[2/6] Lọc đơn hàng thành công và chuẩn hóa mốc thời gian...")
    orders = orders[orders["order_status"] == "delivered"].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # 3. Nối các bảng quan hệ
    print("[3/6] Hợp nhất Order Items, Orders, Products, Reviews, Customers...")
    df = items.merge(orders, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")

    # Review tổng hợp theo đơn hàng (tránh nhân bản dòng)
    rev_agg = (
        reviews.groupby("order_id")
        .agg(rating=("review_score", "mean"), num_reviews=("review_score", "count"))
        .reset_index()
    )
    df = df.merge(rev_agg, on="order_id", how="left")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["num_reviews"] = df["num_reviews"].fillna(1).astype(int)
    df["popularity_score"] = np.round(df["rating"] * np.log1p(df["num_reviews"]), 2)

    # Sắp xếp tuần tự theo chuỗi thời gian để chống rò rỉ dữ liệu (Temporal Sorting)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    print("[4/6] Khởi tạo biến mục tiêu và 5 nhóm đặc trưng...")
    # Biến mục tiêu: sales (tổng giá trị giao dịch đơn hàng)
    df["sales"] = df["price"].round(2)  # Item revenue; ordinal is not quantity.

    # Nhóm 1: Product Information & Text
    cat_avg = df.groupby("category")["price"].transform("mean")
    df["price_relative_to_cat_avg"] = np.round(df["price"] / cat_avg.replace(0, 1), 3)
    df["description_length"] = (
        df["product_description_lenght"]
        .fillna(df["product_description_lenght"].median())
        .astype(int)
    )
    df["description_word_count"] = np.round(df["description_length"] / 6.0).astype(int)
    df["description"] = df["category"].astype(str)

    # Nhóm 2: Customer Demographics
    np.random.seed(random_state)
    n = len(df)
    df["location"] = df["customer_state"].fillna("SP")
    df["age"] = np.random.randint(18, 70, size=n)
    df["gender"] = np.random.choice(["M", "F"], size=n, p=[0.48, 0.52])
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["Under 25", "26-35", "36-50", "50+"],
    ).astype(str)

    # Nhóm 3: Purchase History & CLV (RFM)
    df["previous_purchases"] = df.groupby("customer_unique_id").cumcount()
    df["purchase_frequency"] = df.groupby("customer_unique_id")["order_id"].transform("count")
    max_date = df["order_purchase_timestamp"].max()
    df["recency_days"] = (max_date - df["order_purchase_timestamp"]).dt.days
    df["avg_order_value"] = np.round(
        df.groupby("customer_unique_id")["sales"].transform("mean"), 2
    )
    df["customer_lifetime_value"] = np.round(
        df["purchase_frequency"] * df["avg_order_value"], 2
    )

    # Nhóm 4: Marketing Campaigns
    df["campaigns_run"] = np.random.randint(1, 6, size=n)
    df["campaign_channel"] = np.random.choice(
        ["Social Media", "Search Ads", "Influencer", "Email", "Organic"], size=n
    )
    df["discount_rate"] = np.round(np.random.uniform(0.0, 0.25, size=n), 2)
    df["discounted_price"] = np.round(df["price"] * (1 - df["discount_rate"]), 2)
    df["ad_spend"] = np.round(df["price"] * np.random.uniform(0.05, 0.20, size=n), 2)
    df["ad_spend_to_price_ratio"] = np.round(
        df["ad_spend"] / np.maximum(df["price"], 1.0), 4
    )

    # Nhóm 5: Seasonal & Calendar Information
    df["month"] = df["order_purchase_timestamp"].dt.month
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Mùa vụ Brazil (Nam bán cầu): Hè: 12-2, Thu: 3-5, Đông: 6-8, Xuân: 9-11
    season_map = {
        12: "Summer", 1: "Summer", 2: "Summer",
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring",
    }
    df["season"] = df["month"].map(season_map)
    df["is_holiday"] = (
        (df["month"] == 11) & (df["order_purchase_timestamp"].dt.day >= 20)
        | (df["month"] == 12) & (df["order_purchase_timestamp"].dt.day >= 20)
    ).astype(int)
    df["is_peak_shopping_season"] = df["month"].isin([11, 12]).astype(int)

    # Chuỗi thời gian: Biến trễ Lag & Rolling
    df["sales_lag_1"] = df["sales"].shift(1).bfill()
    df["sales_lag_7"] = df["sales"].shift(7).bfill()
    df["sales_rolling_mean_7"] = (
        df["sales"].rolling(window=7, min_periods=1).mean().round(2)
    )
    df["sales_rolling_std_7"] = (
        df["sales"].rolling(window=7, min_periods=1).std().fillna(0).round(2)
    )

    # Kiểm tra đầy đủ các cột khai báo trong config
    all_needed_cols = (
        [config.TARGET_COLUMN]
        + config.PRODUCT_BASE_COLS
        + config.PRODUCT_POPULARITY_COLS
        + config.PRODUCT_TEXT_COLS
        + config.CUSTOMER_DEMOGRAPHIC_COLS
        + config.PURCHASE_HISTORY_COLS
        + config.MARKETING_COLS
        + config.SEASONAL_CALENDAR_COLS
        + config.TEMPORAL_LAG_COLS
    )

    missing_cols = [c for c in all_needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột định nghĩa trong config: {missing_cols}")

    # Đưa các cột chính lên đầu DataFrame để tiện theo dõi
    metadata_cols = [
        "order_id",
        "product_id",
        "customer_id",
        "customer_unique_id",
        "order_purchase_timestamp",
    ]
    reordered_cols = metadata_cols + [
        c for c in df.columns if c not in metadata_cols
    ]
    df = df[reordered_cols]

    print(f"[5/6] Ghi dữ liệu ra tệp: {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[6/6] Hoàn thành! Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột (~{file_size_mb:.2f} MB)")
    return df


def build_cold_start_dataset(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Xây dựng proxy doanh thu 7 ngày kể từ giao dịch đầu tiên đã quan sát.
    Giá đầu vào là giá giao dịch đầu, không phải median giá trong kỳ đích.
    Không chứng minh dự báo trước ngày niêm yết hoặc cho sản phẩm chưa từng bán.
    Trạng thái delivered và thuộc tính sản phẩm là snapshot cuối của Olist;
    đánh giá này giả định chúng đại diện cho dữ liệu tại thời điểm giao dịch đầu.
    
    Quy trình:
    1. Đọc olist_orders_dataset (lọc order_status == 'delivered') và olist_order_items_dataset.
    2. Với mỗi product_id, xác định thời điểm mua đầu tiên (first_purchase_time).
    3. Thu thập các giao dịch trong tuần mở bán: [first_purchase_time, first_purchase_time + 7 days).
    4. Tổng hợp doanh số 7 ngày (launch_sales_brl, tên giữ để tương thích).
    5. Ghép nối với olist_products_dataset để lấy các đặc trưng nội dung vật lý.
    """
    if raw_dir is None:
        raw_dir = config.RAW_DATA_DIR

    orders_path = raw_dir / "olist_orders_dataset.csv"
    items_path = raw_dir / "olist_order_items_dataset.csv"
    products_path = raw_dir / "olist_products_dataset.csv"

    if not (orders_path.exists() and items_path.exists() and products_path.exists()):
        raise FileNotFoundError(f"Không đủ tệp dữ liệu thô trong {raw_dir} để xây dựng tập Cold-Start.")

    orders = pd.read_csv(orders_path, usecols=["order_id", "order_status", config.TIMESTAMP_COLUMN])
    orders[config.TIMESTAMP_COLUMN] = pd.to_datetime(orders[config.TIMESTAMP_COLUMN])
    # Conservative observation cutoff from actual delivered purchases, not late
    # cancelled orders that could extend the apparent collection period.
    orders = orders.loc[orders["order_status"].eq("delivered")].copy()
    observed_until = orders[config.TIMESTAMP_COLUMN].max()

    items = pd.read_csv(items_path, usecols=["order_id", "product_id", "price"])
    items["price"] = pd.to_numeric(items["price"], errors="coerce")
    items = items.loc[(items["price"] > 0) & np.isfinite(items["price"])].copy()

    products = pd.read_csv(products_path)

    # Nối đơn hàng với item
    merged_items = items.merge(orders[["order_id", config.TIMESTAMP_COLUMN]], on="order_id", how="inner")

    # Xác định thời điểm xuất hiện lần đầu của từng sản phẩm
    first_times = (merged_items.sort_values([config.TIMESTAMP_COLUMN, "order_id"], kind="stable")
                   .drop_duplicates("product_id")
                   [["product_id", config.TIMESTAMP_COLUMN, "price"]]
                   .rename(columns={config.TIMESTAMP_COLUMN: "first_purchase_time", "price": "first_price"}))
    first_times["target_window_end"] = first_times["first_purchase_time"] + pd.Timedelta(days=7)
    first_times = first_times.loc[first_times["target_window_end"] <= observed_until].copy()
    merged_items = merged_items.merge(first_times, on="product_id", how="inner")

    # Giữ lại các giao dịch trong khoảng 7 ngày đầu kể từ lúc ra mắt
    launch_window_end = merged_items["first_purchase_time"] + pd.Timedelta(days=7)
    launch_items = merged_items.loc[merged_items[config.TIMESTAMP_COLUMN] < launch_window_end].copy()

    launch_sales = launch_items.groupby("product_id").agg(
        launch_sales_brl=("price", "sum"),
        launch_units=("price", "count")
    ).reset_index()

    cold_df = products.merge(launch_sales, on="product_id", how="inner", validate="one_to_one")
    cold_df = cold_df.merge(first_times, on="product_id", validate="one_to_one")
    cold_df["price"] = cold_df.pop("first_price")
    cold_df = cold_df.sort_values(["first_purchase_time", "product_id"]).reset_index(drop=True)
    return cold_df


if __name__ == "__main__":
    build_merged_sales_dataset()
