"""
loader.py - Module tải và kiểm toán chất lượng dữ liệu Olist E-commerce
Chuẩn hóa hiển thị kết quả kiểm toán bằng pandas.DataFrame
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config


def load_raw_sales_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Tải tập dữ liệu bán hàng hợp nhất Olist E-Commerce.
    
    Args:
        filepath: Đường dẫn tới file CSV (mặc định lấy từ config.RAW_DATA_FILE)
        
    Returns:
        pd.DataFrame: Bảng dữ liệu thô đã được định dạng datetime
    """
    if filepath is None:
        filepath = config.RAW_DATA_FILE

    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"[LOADER ERROR] Không tìm thấy tệp dữ liệu bán hàng: {filepath}. "
            "Vui lòng đảm bảo tệp dữ liệu đã sẵn sàng trong data/raw/ hoặc kiểm tra lại đường dẫn."
        )
    df = pd.read_csv(filepath, low_memory=False)

    if config.TIMESTAMP_COLUMN in df.columns:
        df[config.TIMESTAMP_COLUMN] = pd.to_datetime(df[config.TIMESTAMP_COLUMN])
        df = df.sort_values(config.TIMESTAMP_COLUMN).reset_index(drop=True)

    # Legacy exports multiplied price by the item ordinal, not a quantity.
    # Keep source EDA correct; the weekly builder independently sums item prices.
    if {"order_id", "order_item_id", "price"}.issubset(df.columns):
        df[config.TARGET_COLUMN] = pd.to_numeric(df["price"], errors="coerce")

    return df


def audit_data_quality(df: pd.DataFrame, max_cols: int = 40) -> pd.DataFrame:
    """
    Kiểm toán chất lượng dữ liệu và trả về bảng pandas.DataFrame trực quan.
    
    Args:
        df: DataFrame cần kiểm toán
        max_cols: Số lượng cột hiển thị tối đa để tránh quá tải
        
    Returns:
        pd.DataFrame: Bảng kiểm toán gồm kiểu dữ liệu, số thiếu, tỷ lệ thiếu, số unique
    """
    total_rows = len(df)
    records = []

    target_cols = df.columns[:max_cols] if len(df.columns) > max_cols else df.columns

    for col in target_cols:
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        unique_count = int(df[col].nunique(dropna=True))
        sample_val = str(df[col].dropna().iloc[0]) if unique_count > 0 else "None"
        if len(sample_val) > 25:
            sample_val = sample_val[:22] + "..."

        records.append({
            "Thuộc tính (Feature)": col,
            "Kiểu dữ liệu (Dtype)": str(df[col].dtype),
            "Số lượng bản ghi": f"{total_rows:,}",
            "Số giá trị thiếu (NaN)": f"{missing_count:,}",
            "Tỷ lệ thiếu (%)": f"{missing_pct}%",
            "Số giá trị duy nhất (Unique)": f"{unique_count:,}",
            "Mẫu dữ liệu thực tế": sample_val
        })

    return pd.DataFrame(records)


def get_dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xuất bảng tổng quan vĩ mô về tập dữ liệu thương mại điện tử Olist.
    """
    n_rows, n_cols = df.shape
    min_date = df[config.TIMESTAMP_COLUMN].min() if config.TIMESTAMP_COLUMN in df.columns else "N/A"
    max_date = df[config.TIMESTAMP_COLUMN].max() if config.TIMESTAMP_COLUMN in df.columns else "N/A"

    sales_series = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else pd.Series(dtype=float)
    total_sales = sales_series.sum() if not sales_series.empty else 0.0
    mean_sales = sales_series.mean() if not sales_series.empty else 0.0
    median_sales = sales_series.median() if not sales_series.empty else 0.0

    overview = [
        {"Chỉ tiêu tổng quan (Metric)": "Tổng số lượng quan sát (Transactions)", "Giá trị": f"{n_rows:,} đơn hàng"},
        {"Chỉ tiêu tổng quan (Metric)": "Số chiều đặc trưng (Features)", "Giá trị": f"{n_cols:,} cột"},
        {"Chỉ tiêu tổng quan (Metric)": "Mốc thời gian bắt đầu (Start Date)", "Giá trị": str(min_date)},
        {"Chỉ tiêu tổng quan (Metric)": "Mốc thời gian kết thúc (End Date)", "Giá trị": str(max_date)},
        {"Chỉ tiêu tổng quan (Metric)": "Tổng doanh số toàn sàn (Total Revenue)", "Giá trị": f"{total_sales:,.2f} BRL"},
        {"Chỉ tiêu tổng quan (Metric)": "Doanh số trung bình / đơn (Mean Sales)", "Giá trị": f"{mean_sales:,.2f} BRL"},
        {"Chỉ tiêu tổng quan (Metric)": "Doanh số trung vị / đơn (Median Sales)", "Giá trị": f"{median_sales:,.2f} BRL"},
        {"Chỉ tiêu tổng quan (Metric)": "Số danh mục sản phẩm (Categories)", "Giá trị": f"{df['category'].nunique() if 'category' in df.columns else 'N/A'}"},
        {"Chỉ tiêu tổng quan (Metric)": "Số bang khách hàng (States)", "Giá trị": f"{df['customer_state'].nunique() if 'customer_state' in df.columns else 'N/A'}"},
    ]

    return pd.DataFrame(overview)


# Alias tương thích linh hoạt cho quy trình và notebook
load_data = load_raw_sales_data
explore_data = audit_data_quality
