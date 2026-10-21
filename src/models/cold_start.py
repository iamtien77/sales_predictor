"""
cold_start.py - Mô hình Dự báo Doanh số Tuần Ra mắt cho Sản phẩm Mới (Cold-Start Product Predictor)
====================================================================================================
Triển khai giải pháp học máy cho sản phẩm mới hoàn toàn chưa có lịch sử bán hàng:
1. CategoryMedianColdStartBaseline: Đường cơ sở thống kê theo trung vị doanh số tuần đầu của danh mục.
2. ColdStartProductRegressor: Mô hình cây hồi quy From-Scratch học trên các đặc trưng nội dung
   (Content-based features: mức giá niêm yết, danh mục, kích thước, khối lượng, số lượng ảnh).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.models.tree import DecisionTreeRegressorFromScratch
from src.models.base import BaseRegressorFromScratch


class CategoryMedianColdStartBaseline(BaseRegressorFromScratch):
    """
    Baseline đối chuẩn: Dự báo doanh thu tuần ra mắt bằng Trung vị Doanh số Tuần đầu
    của từng danh mục sản phẩm trong tập huấn luyện (với fallback về trung vị toàn sàn).
    """

    def __init__(self, category_col: str = "product_category_name"):
        super().__init__()
        self.category_col = category_col
        self.category_medians_: Dict[str, float] = {}
        self.global_median_: float = 0.0
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> "CategoryMedianColdStartBaseline":
        """
        Fit trung vị doanh số tuần đầu trên tập huấn luyện.
        
        Parameters:
        -----------
        df: DataFrame chứa cột danh mục và cột doanh số 'sales' (hoặc 'launch_sales_brl')
        y: Mảng nhãn doanh số tuỳ chọn (nếu không có trong df)
        """
        target = y if y is not None else (df["launch_sales_brl"] if "launch_sales_brl" in df else df["sales"])
        df_fit = pd.DataFrame({
            self.category_col: df[self.category_col].astype(str),
            "target": np.asarray(target, dtype=float)
        })

        self.global_median_ = float(df_fit["target"].median())
        grouped = df_fit.groupby(self.category_col)["target"].median()
        self.category_medians_ = grouped.to_dict()
        self.is_fitted = True
        return self

    def predict(self, df: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Dự báo trung vị doanh số theo danh mục.
        """
        if not self.is_fitted:
            raise RuntimeError("Baseline chưa được fit! Vui lòng gọi fit() trước khi predict().")

        if isinstance(df, dict):
            df = pd.DataFrame([df])

        cats = df[self.category_col].astype(str)
        preds = cats.map(self.category_medians_).fillna(self.global_median_).to_numpy(dtype=float)
        return np.maximum(0.0, preds)


class ColdStartProductRegressor(BaseRegressorFromScratch):
    """
    Mô hình Hồi quy Sản phẩm Mới (Cold-Start Regressor From-Scratch):
    - Trích xuất và tiền xử lý 6 đặc trưng nội dung sản phẩm:
      + Giá niêm yết (price)
      + Trọng lượng (product_weight_g)
      + Số lượng ảnh đại diện (product_photos_qty)
      + Độ dài tên sản phẩm (product_name_lenght)
      + Độ dài mô tả sản phẩm (product_description_lenght)
      + Mã hóa danh mục sản phẩm (Target Encoding có Laplace Smoothing)
    - Học mối quan hệ phi tuyến trên không gian Log tự nhiên log(1 + y)
      bằng DecisionTreeRegressorFromScratch.
    - Nghịch đảo hàm mũ và kẹp dưới 0 khi dự báo (BRL).
    """

    FEATURE_NUMERIC = [
        "price",
        "product_weight_g",
        "product_photos_qty",
        "product_name_lenght",
        "product_description_lenght"
    ]

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        category_col: str = "product_category_name",
        smoothing: float = 10.0
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.category_col = category_col
        self.smoothing = smoothing

        self.tree_ = DecisionTreeRegressorFromScratch(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        self.medians_: Dict[str, float] = {}
        self.cat_target_map_: Dict[str, float] = {}
        self.global_target_mean_: float = 0.0
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """
        Chuẩn bị ma trận đặc trưng nội dung: Impute median, Target Encode category, Log-transform.
        """
        df_copy = df.copy()

        # 1. Điền khuyết các cột số
        for col in self.FEATURE_NUMERIC:
            if col not in df_copy:
                df_copy[col] = self.medians_.get(col, 1.0)
            if is_training:
                val = float(df_copy[col].median())
                self.medians_[col] = val if not np.isnan(val) else 1.0
            df_copy[col] = df_copy[col].fillna(self.medians_.get(col, 1.0))

        # 2. Target Encoding có Smoothing cho danh mục
        if self.category_col not in df_copy:
            df_copy[self.category_col] = "unknown"
        cats = df_copy[self.category_col].astype(str)

        if is_training:
            pass  # Đã tính trong fit()

        cat_encoded = cats.map(self.cat_target_map_).fillna(self.global_target_mean_).to_numpy(dtype=float)

        # 3. Kết hợp các cột và biến đổi log1p
        num_matrix = df_copy[self.FEATURE_NUMERIC].to_numpy(dtype=float)
        num_matrix = np.maximum(0.0, num_matrix)  # Chống giá trị âm không hợp lệ
        full_features = np.column_stack([num_matrix, cat_encoded])

        # Áp dụng log1p để ổn định phân phối lệch
        return np.log1p(full_features)

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> "ColdStartProductRegressor":
        """
        Huấn luyện mô hình cây trên dữ liệu sản phẩm tuần đầu.
        
        Parameters:
        -----------
        df: DataFrame chứa thông tin sản phẩm và doanh số tuần đầu.
        y: Mảng nhãn doanh số tuần đầu (BRL). Nếu None, tự đọc từ df['launch_sales_brl'] hoặc df['sales'].
        """
        target = y if y is not None else (df["launch_sales_brl"] if "launch_sales_brl" in df else df["sales"])
        target_arr = np.asarray(target, dtype=float)

        # Tính Target Encoding cho danh mục với Laplace smoothing
        self.global_target_mean_ = float(np.mean(target_arr))
        cat_series = df[self.category_col].astype(str)
        cat_stats = pd.DataFrame({"cat": cat_series, "y": target_arr}).groupby("cat").agg(
            count=("y", "count"),
            sum=("y", "sum")
        )
        # Formula: smoothed_mean = (sum + smoothing * global_mean) / (count + smoothing)
        smoothed = (cat_stats["sum"] + self.smoothing * self.global_target_mean_) / (cat_stats["count"] + self.smoothing)
        self.cat_target_map_ = smoothed.to_dict()

        X_train = self._prepare_features(df, is_training=True)
        y_log = np.log1p(np.maximum(0.0, target_arr))

        self.tree_.fit(X_train, y_log)
        self.is_fitted = True
        return self

    def predict(self, df: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Dự báo doanh số tuần đầu tiên (BRL) cho sản phẩm mới.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit() trước khi predict().")

        if isinstance(df, dict):
            df = pd.DataFrame([df])

        X_eval = self._prepare_features(df, is_training=False)
        pred_log = self.tree_.predict(X_eval)
        pred_sales = np.expm1(pred_log)
        return np.maximum(0.0, pred_sales)

    def evaluate(self, df_test: pd.DataFrame, y_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Đánh giá sai số trên tập kiểm tra độc lập.
        """
        y_true = np.asarray(y_test if y_test is not None else (
            df_test["launch_sales_brl"] if "launch_sales_brl" in df_test else df_test["sales"]
        ), dtype=float)

        y_pred = self.predict(df_test)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))

        return {
            "MAE_BRL": round(mae, 2),
            "RMSE_BRL": round(rmse, 2),
            "R2_Score": round(r2, 4)
        }
