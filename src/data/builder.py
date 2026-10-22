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


def build_merged_sales_dataset(output_path: Path = None, random_state: int = 42) -> pd.DataFrame:
    """
    Hợp nhất các bảng Olist và sinh các nhóm đặc trưng bám sát src/config.py.
    
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
    df["sales"] = np.round(df["price"] * df["order_item_id"], 2)

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


def build_product_week_panel(
    raw_dir: Path = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """Tạo panel Sản phẩm–Tuần đầy đủ từ sáu bảng dữ liệu Olist gốc."""
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR
    output_path = Path(output_path) if output_path is not None else config.WEEKLY_PANEL_FILE

    required_files = {
        "items": "olist_order_items_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "products": "olist_products_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "translations": "product_category_name_translation.csv",
    }
    missing_files = [name for name in required_files.values() if not (raw_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing Olist source files in {raw_dir}: {missing_files}")

    items = pd.read_csv(raw_dir / required_files["items"])
    orders = pd.read_csv(raw_dir / required_files["orders"])
    products = pd.read_csv(raw_dir / required_files["products"])
    reviews = pd.read_csv(raw_dir / required_files["reviews"])
    customers = pd.read_csv(
        raw_dir / required_files["customers"],
        usecols=["customer_id", "customer_state"],
    )
    translations = pd.read_csv(raw_dir / required_files["translations"])

    products = products.merge(translations, on="product_category_name", how="left")
    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("other")
    )
    product_metadata = products[["product_id", "category"]].drop_duplicates("product_id")

    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id", "customer_id", "order_purchase_timestamp"],
    ].copy()
    delivered_orders["order_purchase_timestamp"] = pd.to_datetime(
        delivered_orders["order_purchase_timestamp"], errors="coerce"
    )
    delivered_orders = delivered_orders.dropna(subset=["order_purchase_timestamp"])

    transactions = (
        items.merge(delivered_orders, on="order_id", how="inner")
        .merge(product_metadata, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )
    transactions["week_start"] = (
        transactions["order_purchase_timestamp"].dt.normalize()
        - pd.to_timedelta(transactions["order_purchase_timestamp"].dt.dayofweek, unit="D")
    )
    if transactions.empty:
        raise ValueError("No delivered Olist transactions with valid purchase timestamps.")

    # Olist lưu mỗi đơn vị hàng trên một dòng; order_item_id là thứ tự mặt hàng,
    # không phải số lượng, nên số lượng bán được tính bằng cách đếm dòng.
    weekly_sales = (
        transactions.groupby(["product_id", "week_start"], observed=True)
        .agg(
            sales=("price", "sum"),
            units_sold=("price", "size"),
            avg_price=("price", "mean"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )

    review_events = (
        reviews.merge(items[["order_id", "product_id"]], on="order_id", how="inner")
        [["product_id", "review_score", "review_creation_date"]]
        .copy()
    )
    review_events["review_creation_date"] = pd.to_datetime(
        review_events["review_creation_date"], errors="coerce"
    )
    review_events = review_events.dropna(subset=["review_creation_date"])
    review_events["week_start"] = (
        review_events["review_creation_date"].dt.normalize()
        - pd.to_timedelta(review_events["review_creation_date"].dt.dayofweek, unit="D")
    )
    weekly_reviews = (
        review_events.groupby(["product_id", "week_start"], observed=True)
        .agg(
            review_count=("review_score", "count"),
            rating_sum=("review_score", "sum"),
        )
        .reset_index()
    )

    first_state_weeks = (
        transactions.dropna(subset=["customer_state"])
        .groupby(["product_id", "customer_state"], observed=True)["week_start"]
        .min()
        .reset_index()
    )
    weekly_new_states = (
        first_state_weeks.groupby(["product_id", "week_start"], observed=True)
        .size()
        .rename("new_customer_states")
        .reset_index()
    )

    first_weeks = weekly_sales.groupby("product_id")["week_start"].min()
    all_weeks = pd.date_range(
        weekly_sales["week_start"].min(),
        weekly_sales["week_start"].max(),
        freq="W-MON",
    )
    week_positions = {week: position for position, week in enumerate(all_weeks)}
    first_positions = first_weeks.map(week_positions).to_numpy(dtype=np.int32)

    # Mỗi sản phẩm bắt đầu từ tuần bán đầu tiên; các tuần không bán tiếp theo
    # vẫn được giữ lại đến tuần quan sát cuối cùng.
    panel_index = pd.MultiIndex.from_arrays(
        [
            np.repeat(first_weeks.index.to_numpy(), len(all_weeks) - first_positions),
            all_weeks.to_numpy()[
                np.concatenate([
                    np.arange(start, len(all_weeks), dtype=np.int32)
                    for start in first_positions
                ])
            ],
        ],
        names=["product_id", "week_start"],
    )
    panel = pd.DataFrame(index=panel_index).reset_index()
    panel = panel.merge(product_metadata, on="product_id", how="left")
    panel = panel.merge(weekly_sales, on=["product_id", "week_start"], how="left")
    panel = panel.merge(weekly_reviews, on=["product_id", "week_start"], how="left")
    panel = panel.merge(weekly_new_states, on=["product_id", "week_start"], how="left")

    zero_columns = [
        "sales", "units_sold", "order_count", "review_count", "rating_sum",
        "new_customer_states",
    ]
    panel[zero_columns] = panel[zero_columns].fillna(0)
    panel["sales"] = panel["sales"].astype("float32")
    panel["units_sold"] = panel["units_sold"].astype("int32")
    panel["order_count"] = panel["order_count"].astype("int32")
    panel["review_count"] = panel["review_count"].astype("int32")
    panel["rating_sum"] = panel["rating_sum"].astype("float32")
    panel["new_customer_states"] = panel["new_customer_states"].astype("int16")

    # Giá đầu vào lấy từ tuần đã hoàn tất gần nhất, không lấy giá của tuần mục tiêu.
    panel = panel.sort_values(["product_id", "week_start"], kind="stable")
    panel["avg_price_last_observed"] = (
        panel.groupby("product_id", observed=True)["avg_price"]
        .transform(lambda values: values.shift(1).ffill())
        .astype("float32")
    )
    panel = panel.drop(columns="avg_price")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    print(f"Saved weekly product panel: {panel.shape[0]:,} rows, {panel.shape[1]} columns to {output_path}")
    return panel


if __name__ == "__main__":
    build_merged_sales_dataset()
