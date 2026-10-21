"""
visualization.py - Module trực quan hóa dữ liệu và kết quả thực nghiệm chuẩn xuất bản (dpi=300)
Hỗ trợ đầy đủ: EDA đa chiều (Phân bố, Outliers, Mùa vụ, Bivariate, Nhân khẩu học, Tương quan)
Tuân thủ: Có nhánh plt.show() khi chạy interactive trên Notebook và lưu file khi có save_path
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib
if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config

# Thiết lập phong cách trực quan hóa thẩm mỹ cao
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

def _save_and_show(
    fig: plt.Figure,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """Hàm helper lưu biểu đồ ra 1 hoặc nhiều thư mục và hiển thị duy nhất 1 lần"""
    plt.tight_layout()
    if save_path:
        paths = [save_path] if isinstance(save_path, (str, Path)) else list(save_path)
        for p in paths:
            target = Path(p)
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=300, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)




def plot_target_sales_distribution(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    1. Trực quan hóa phân bố biến mục tiêu Doanh số (Sales):
    So sánh giữa thang đo gốc (Lệch phải nặng) và Không gian Log (Tiệm cận chuẩn Gaussian).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sales = df[config.TARGET_COLUMN].dropna().values
    log_sales = np.log1p(np.maximum(sales, 0.0))

    # Thang đo gốc
    p99 = float(np.percentile(sales, 99))
    axes[0].hist(sales[sales <= p99], bins=40, color="#1F77B4", edgecolor="black", alpha=0.75)
    axes[0].axvline(np.mean(sales), color="#D62728", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(sales):,.1f} BRL")
    axes[0].axvline(np.median(sales), color="#FF7F0E", linestyle=":", linewidth=1.5, label=f"Median = {np.median(sales):,.1f} BRL")
    axes[0].set_title(f"Phân bố Doanh số gốc (Sales <= p99 = {p99:,.1f} BRL)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Doanh số (BRL)", fontsize=10)
    axes[0].set_ylabel("Tần suất", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Thang đo Log1p
    axes[1].hist(log_sales, bins=40, color="#2CA02C", edgecolor="black", alpha=0.75)
    axes[1].axvline(np.mean(log_sales), color="#D62728", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(log_sales):.2f}")
    axes[1].axvline(np.median(log_sales), color="#FF7F0E", linestyle=":", linewidth=1.5, label=f"Median = {np.median(log_sales):.2f}")
    axes[1].set_title("Phân bố sau biến đổi Log1p (Tiệm cận chuẩn Gaussian)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("log(1 + Sales)", fontsize=10)
    axes[1].set_ylabel("Tần suất", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    _save_and_show(fig, save_path)


def plot_outliers_boxplots(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    2. Biểu đồ Boxplot 4 góc phát hiện ngoại lai (Outliers) cho các biến số học chủ chốt.
    """
    if cols is None:
        cols = ["price", "freight_value", "num_reviews", "ad_spend"]
    valid_cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for idx, col in enumerate(valid_cols[:4]):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        p99 = float(s.quantile(0.99))
        n_outliers = int((s > q3 + 1.5 * iqr).sum())
        pct_outliers = (n_outliers / len(s)) * 100

        axes[idx].boxplot(s, vert=False, patch_artist=True,
                         boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.7),
                         flierprops=dict(marker="o", color="red", alpha=0.3, markersize=3))
        axes[idx].set_title(f"Boxplot: {col} (Ngoại lai IQR: {n_outliers:,} mẫu ~ {pct_outliers:.1f}%)",
                            fontsize=10.5, fontweight="bold")
        axes[idx].set_xlabel("Giá trị", fontsize=9.5)
        axes[idx].grid(axis="x", linestyle=":", alpha=0.6)

    _save_and_show(fig, save_path)


def plot_seasonal_and_time_trends(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    3. Biểu đồ phân tích chuỗi thời gian & tính mùa vụ đa cấp độ:
    - Doanh số trung bình theo Tháng (Month)
    - Doanh số theo Ngày trong tuần (Day of Week)
    - Doanh số Cuối tuần vs Ngày thường (is_weekend)
    - Doanh số Mùa mua sắm cao điểm (is_peak_shopping_season)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Theo tháng
    if "month" in df.columns:
        m_sales = df.groupby("month")[config.TARGET_COLUMN].mean()
        axes[0, 0].bar(m_sales.index, m_sales.values, color="#1F77B4", edgecolor="black", alpha=0.75)
        axes[0, 0].set_title("Doanh số trung bình theo Tháng (1 - 12)", fontsize=11, fontweight="bold")
        axes[0, 0].set_xlabel("Tháng", fontsize=10)
        axes[0, 0].set_ylabel("Doanh số trung bình (BRL)", fontsize=10)
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Theo ngày trong tuần
    if "day_of_week" in df.columns:
        dow_names = ["T2 (Mon)", "T3 (Tue)", "T4 (Wed)", "T5 (Thu)", "T6 (Fri)", "T7 (Sat)", "CN (Sun)"]
        dow_sales = df.groupby("day_of_week")[config.TARGET_COLUMN].mean()
        axes[0, 1].plot(range(len(dow_sales)), dow_sales.values, marker="o", color="#2CA02C", linewidth=2.2)
        axes[0, 1].set_title("Doanh số trung bình theo Ngày trong tuần", fontsize=11, fontweight="bold")
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(dow_names, fontsize=9.5)
        axes[0, 1].set_ylabel("Doanh số trung bình (BRL)", fontsize=10)
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Cuối tuần vs Ngày thường
    if "is_weekend" in df.columns:
        wk_sales = df.groupby("is_weekend")[config.TARGET_COLUMN].mean()
        lbls = ["Ngày thường (Weekdays)", "Cuối tuần (Weekends)"]
        axes[1, 0].bar(lbls, wk_sales.values, color=["#4C72B0", "#E15759"], edgecolor="black", width=0.45)
        for idx, v in enumerate(wk_sales.values):
            axes[1, 0].annotate(f"{v:,.1f} BRL", xy=(idx, v), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontweight="bold")
        axes[1, 0].set_title("So sánh Doanh số: Ngày thường vs Cuối tuần", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Doanh số TB (BRL)", fontsize=10)
        axes[1, 0].set_ylim([0, max(wk_sales.values) * 1.2])
        axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Mùa cao điểm mua sắm
    if "is_peak_shopping_season" in df.columns:
        peak_sales = df.groupby("is_peak_shopping_season")[config.TARGET_COLUMN].mean()
        lbls_p = ["Mùa thông thường", "Mùa cao điểm (Black Friday/Tết)"]
        axes[1, 1].bar(lbls_p, peak_sales.values, color=["#76B7B2", "#F28E2B"], edgecolor="black", width=0.45)
        for idx, v in enumerate(peak_sales.values):
            axes[1, 1].annotate(f"{v:,.1f} BRL", xy=(idx, v), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontweight="bold")
        axes[1, 1].set_title("Tác động của Mùa mua sắm cao điểm", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Doanh số TB (BRL)", fontsize=10)
        axes[1, 1].set_ylim([0, max(peak_sales.values) * 1.2])
        axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    _save_and_show(fig, save_path)


def plot_bivariate_relationships(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    4. Trực quan hóa mối quan hệ 2 biến (Bivariate Analysis) với Doanh số (Sales).
    - Price vs Sales
    - Ad Spend vs Sales
    - Discount Rate vs Sales
    - Popularity Score vs Sales
    """
    sample_df = df.sample(n=min(3000, len(df)), random_state=42).copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Price vs Sales
    axes[0, 0].scatter(sample_df["price"], sample_df["sales"], alpha=0.35, color="#1F77B4", s=18)
    axes[0, 0].set_title("Mối tương quan giữa Giá bán (Price) và Doanh số (Sales)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Giá sản phẩm (BRL)", fontsize=10)
    axes[0, 0].set_ylabel("Doanh số (BRL)", fontsize=10)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Ad Spend vs Sales
    if "ad_spend" in sample_df.columns:
        axes[0, 1].scatter(sample_df["ad_spend"], sample_df["sales"], alpha=0.35, color="#FF7F0E", s=18)
        axes[0, 1].set_title("Tác động của Ngân sách quảng cáo (Ad Spend) lên Doanh số", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Ngân sách quảng cáo (BRL)", fontsize=10)
        axes[0, 1].set_ylabel("Doanh số (BRL)", fontsize=10)
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Discount Rate vs Sales
    if "discount_rate" in sample_df.columns:
        disc_sales = sample_df.groupby("discount_rate")["sales"].mean().reset_index()
        axes[1, 0].bar(disc_sales["discount_rate"].astype(str), disc_sales["sales"], color="#2CA02C", edgecolor="black", width=0.45)
        for idx, row in disc_sales.iterrows():
            axes[1, 0].annotate(f"{row['sales']:,.1f}", xy=(idx, row["sales"]), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontweight="bold")
        axes[1, 0].set_title("Doanh số trung bình theo Tỷ lệ chiết khấu (Discount Rate)", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Tỷ lệ giảm giá", fontsize=10)
        axes[1, 0].set_ylabel("Doanh số TB (BRL)", fontsize=10)
        axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Popularity Score vs Sales
    if "popularity_score" in sample_df.columns:
        axes[1, 1].scatter(sample_df["popularity_score"], sample_df["sales"], alpha=0.35, color="#D62728", s=18)
        axes[1, 1].set_title("Tác động của Điểm phổ biến (Popularity Score) lên Doanh số", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Popularity Score = Rating * log(1 + Reviews)", fontsize=10)
        axes[1, 1].set_ylabel("Doanh số (BRL)", fontsize=10)
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    _save_and_show(fig, save_path)


def plot_customer_demographics(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    5. Phân tích khách hàng: Doanh thu theo Nhóm tuổi (Age Group) & Giới tính (Gender).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Doanh số theo Age Group
    if "age_group" in df.columns:
        age_sales = df.groupby("age_group")[config.TARGET_COLUMN].mean().sort_index()
        bars = axes[0].bar(age_sales.index, age_sales.values, color="#59A14F", edgecolor="black", alpha=0.85, width=0.5)
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(f"{h:,.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
        axes[0].set_title("Doanh số trung bình theo Nhóm tuổi khách hàng (Age Group)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Doanh số TB (BRL)", fontsize=10)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Doanh số theo Giới tính
    if "gender" in df.columns:
        gender_sales = df.groupby("gender")[config.TARGET_COLUMN].sum()
        axes[1].pie(gender_sales.values, labels=gender_sales.index, autopct="%1.1f%%",
                    colors=["#4E79A7", "#F28E2B", "#E15759"], startangle=90,
                    wedgeprops=dict(edgecolor="black", linewidth=1))
        axes[1].set_title("Tỷ trọng Doanh số theo Giới tính khách hàng", fontsize=11, fontweight="bold")

    _save_and_show(fig, save_path)


def plot_top_categories_sales(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    6. Biểu đồ thanh ngang Top N danh mục sản phẩm đóng góp doanh số cao nhất.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    cat_sales = (
        df.groupby("category")[config.TARGET_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    y_pos = np.arange(len(cat_sales))
    bars = ax.barh(y_pos, cat_sales.values / 1e3, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_sales.index, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Tổng doanh số (Nghìn BRL)", fontsize=11)
    ax.set_title(f"Top {top_n} Danh mục Sản phẩm có Doanh thu Cao nhất", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:,.1f}k", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5)

    _save_and_show(fig, save_path)


def plot_geographic_sales(
    df: pd.DataFrame,
    top_n: int = 8,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    7. Biểu đồ so sánh doanh số theo các Bang trọng điểm (Customer States).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    state_sales = (
        df.groupby("customer_state")[config.TARGET_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    x = np.arange(len(state_sales))
    bars = ax.bar(x, state_sales.values / 1e3, color="#C44E52", edgecolor="black", alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(state_sales.index, fontsize=10, fontweight="bold")
    ax.set_ylabel("Tổng doanh số (Nghìn BRL)", fontsize=11)
    ax.set_title(f"Doanh thu theo Top {top_n} Bang Khách hàng (Thị trường Brazil)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,.0f}k", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    _save_and_show(fig, save_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    8. Vẽ ma trận nhiệt tương quan Pearson giữa các đặc trưng số học chính và Doanh số.
    """
    if cols is None:
        cols = [
            config.TARGET_COLUMN, "price", "freight_ratio", "discount_amount",
            "popularity_score", "bayesian_popularity", "estimated_delivery_days",
            "recency_days", "customer_lifetime_value", "month", "day_of_week"
        ]
    valid_cols = [c for c in cols if c in df.columns]
    corr_matrix = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
        square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8.5}
    )
    ax.set_title("Ma trận tương quan đặc trưng số học & Doanh số (Pearson Correlation)", fontsize=12, fontweight="bold", pad=12)

    _save_and_show(fig, save_path)


def plot_temporal_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    9. Trực quan hóa quá trình phân chia chuỗi thời gian Temporal Split (8:1:1).
    Minh chứng không có hiện tượng chồng lấn hoặc rò rỉ dữ liệu.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    t_col = config.TIMESTAMP_COLUMN
    for sub_df, lbl, col in [
        (train_df, f"Train Set (80% - {len(train_df):,} mẫu)", "#1F77B4"),
        (val_df, f"Validation Set (10% - {len(val_df):,} mẫu)", "#FF7F0E"),
        (test_df, f"Test Set (10% - {len(test_df):,} mẫu)", "#2CA02C"),
    ]:
        daily_counts = sub_df.set_index(t_col).resample("W")["sales"].count()
        ax.plot(daily_counts.index, daily_counts.values, label=lbl, color=col, linewidth=1.8)

    ax.set_title("Phân chia Chuỗi Thời gian Độc lập 8:1:1 (Chống rò rỉ dữ liệu tương lai)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mốc thời gian giao dịch (Năm - Tháng)", fontsize=11)
    ax.set_ylabel("Số lượng đơn hàng / Tuần", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    _save_and_show(fig, save_path)




def plot_model_comparison(
    summary_df: pd.DataFrame,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    10. Biểu đồ cột nhóm so sánh hiệu năng các mô hình (RMSE, MAE, R²).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = summary_df["Model"].values
    x = np.arange(len(models))
    width = 0.45

    # 1. RMSE (Càng thấp càng tốt)
    bars1 = axes[0].bar(x, summary_df["RMSE"].values, width, color="#1F77B4", edgecolor="black", alpha=0.85)
    axes[0].set_title("Root Mean Squared Error (RMSE) ↓", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right", fontsize=9.5)
    scale = summary_df["Target_Scale"].iloc[0] if "Target_Scale" in summary_df else "log1p"
    axes[0].set_ylabel(f"RMSE ({scale})", fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars1:
        h = b.get_height()
        axes[0].annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # 2. MAE (Càng thấp càng tốt)
    bars2 = axes[1].bar(x, summary_df["MAE"].values, width, color="#FF7F0E", edgecolor="black", alpha=0.85)
    axes[1].set_title("Mean Absolute Error (MAE) ↓", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right", fontsize=9.5)
    axes[1].set_ylabel("MAE", fontsize=10)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars2:
        h = b.get_height()
        axes[1].annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # 3. R² Score (Càng cao càng tốt)
    bars3 = axes[2].bar(x, summary_df["R2"].values, width, color="#2CA02C", edgecolor="black", alpha=0.85)
    axes[2].set_title("Hệ số Xác định (R² Score) ↑", fontsize=11, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right", fontsize=9.5)
    axes[2].set_ylabel("R² Score", fontsize=10)
    axes[2].set_ylim([min(0.0, float(summary_df["R2"].min()) - 0.05), 1.05])
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars3:
        h = b.get_height()
        axes[2].annotate(f"{h*100:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("So Sánh Hiệu Năng 4 Thuật Toán Hồi Quy From Scratch", fontsize=13, fontweight="bold", y=1.02)
    _save_and_show(fig, save_path)


def plot_actual_vs_predicted(
    results_or_preds: Any,
    y_test: np.ndarray,
    model_name: Optional[str] = None,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    11. Đồ thị phân tán Giá trị Thực tế vs Dự đoán (Actual vs Predicted)
    kèm đường chuẩn y = x (hoàn hảo).
    """
    y_t = np.asarray(y_test, dtype=float).ravel()

    # Trường hợp truyền cả results dict: vẽ lưới các mô hình
    if isinstance(results_or_preds, dict) and "y_pred" not in results_or_preds:
        n_models = len(results_or_preds)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        colors = ["#1F77B4", "#2CA02C", "#FF7F0E", "#D62728"]
        for idx, (m_name, res) in enumerate(results_or_preds.items()):
            y_p = np.asarray(res["y_pred"]).ravel()
            ax = axes[idx]
            if len(y_t) > 2000:
                sub_idx = np.random.RandomState(42).choice(len(y_t), 2000, replace=False)
                y_t_sub, y_p_sub = y_t[sub_idx], y_p[sub_idx]
            else:
                y_t_sub, y_p_sub = y_t, y_p

            ax.scatter(y_t_sub, y_p_sub, alpha=0.35, s=15, color=colors[idx % len(colors)], label="Dự đoán")
            min_val = min(float(np.min(y_t)), float(np.min(y_p)))
            max_val = max(float(np.max(y_t)), float(np.max(y_p)))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="y = x (Lý tưởng)")

            r2 = res.get("metrics", {}).get("R2", 0.0)
            rmse = res.get("metrics", {}).get("RMSE", 0.0)
            ax.set_title(f"{m_name}\nR²={r2:.4f} | RMSE={rmse:.4f}", fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Thực tế (Actual)", fontsize=9.5)
            ax.set_ylabel("Dự đoán (Predicted)", fontsize=9.5)
            ax.legend(loc="upper left", fontsize=8.5)
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("Đồ thị Actual vs Predicted trên Tập Kiểm tra (10% Test Set)", fontsize=13, fontweight="bold", y=1.03)
        _save_and_show(fig, save_path)
        return

    # Trường hợp 1 mô hình đơn lẻ
    y_p = results_or_preds["y_pred"] if isinstance(results_or_preds, dict) else np.asarray(results_or_preds).ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_t, y_p, alpha=0.35, s=18, color="#1F77B4", label="Mẫu Test")
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y = x (Chuẩn)")
    title = f"Actual vs Predicted: {model_name}" if model_name else "Actual vs Predicted"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Giá trị Thực tế (Actual)", fontsize=10)
    ax.set_ylabel("Giá trị Dự đoán (Predicted)", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    _save_and_show(fig, save_path)


def plot_residuals(
    results_or_preds: Any,
    y_test: np.ndarray,
    model_name: Optional[str] = None,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    12. Đồ thị Phân tích Phần dư (Residuals Plot):
    - Trục tung: Phần dư e = y - y_pred
    - Trục hoành: Giá trị dự đoán y_pred
    - Histogram phân phối sai số để kiểm tra tính đối xứng và đẳng phương sai.
    """
    y_t = np.asarray(y_test, dtype=float).ravel()

    if isinstance(results_or_preds, dict) and "y_pred" not in results_or_preds:
        best_name = min(results_or_preds.keys(), key=lambda k: results_or_preds[k].get("metrics", {}).get("RMSE", 999))
        y_p = np.asarray(results_or_preds[best_name]["y_pred"]).ravel()
        name_label = best_name
    elif isinstance(results_or_preds, dict):
        y_p = np.asarray(results_or_preds["y_pred"]).ravel()
        name_label = model_name or "Model"
    else:
        y_p = np.asarray(results_or_preds).ravel()
        name_label = model_name or "Model"

    residuals = y_t - y_p

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Residuals vs Predicted
    axes[0].scatter(y_p, residuals, alpha=0.35, s=15, color="#4C72B0")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Phần dư theo Giá trị Dự đoán ({name_label})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Giá trị Dự đoán (Predicted)", fontsize=10)
    axes[0].set_ylabel("Phần dư e = (y - ŷ)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Histogram phân phối phần dư
    axes[1].hist(residuals, bins=40, color="#55A868", edgecolor="black", alpha=0.75, density=True)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    axes[1].axvline(mean_err, color="blue", linestyle=":", linewidth=1.5, label=f"Mean = {mean_err:.4f}")
    axes[1].set_title(f"Phân phối Sai số (Std = {std_err:.4f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Độ lệch sai số", fontsize=10)
    axes[1].set_ylabel("Mật độ", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    _save_and_show(fig, save_path)


def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 15,
    save_path: Optional[Union[Path, str, List[Union[Path, str]]]] = None
) -> None:
    """
    13. Biểu đồ thanh ngang Mức độ quan trọng của Đặc trưng (Feature Importance).
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    n_feats = len(importances)
    if feature_names is None or len(feature_names) != n_feats:
        names = [f"Feature_{i}" for i in range(n_feats)]
    else:
        names = list(feature_names)

    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, int(top_n * 0.35))))
    bars = ax.barh(range(len(top_names)), top_scores, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9.5)
    ax.set_xlabel("Độ quan trọng tương đối (Tích lũy giảm phương sai)", fontsize=10.5)
    model_type = model.__class__.__name__
    ax.set_title(f"Top {top_n} Đặc trưng Quan trọng nhất ({model_type})", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w*100:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold")

    _save_and_show(fig, save_path)
