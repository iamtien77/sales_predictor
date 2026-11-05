"""
preprocessing.py - Bộ tiền xử lý dữ liệu thuần túy From-Scratch toàn diện (NumPy/Pandas)
Không dùng Scikit-Learn:
  - CustomStandardScaler, CustomOneHotEncoder, CustomImputer, CustomOutlierCapper
  - Data Cleaning & Sanitization (Khử giá trị âm/không hợp lệ)
  - Phân chia Temporal Split 8:1:1 vs Random Split đối sánh Data Leakage
  - Đóng gói CustomDataPipeline chuẩn Hướng đối tượng (OOP)
"""
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.data.feature_engineering import SalesFeatureEngineer, SalesFeatureSelectorScratch


def clean_sales_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Kiểm tra làm sạch và chuẩn hóa dữ liệu giao dịch (Data Sanitization).
    Loại bỏ các bản ghi vi phạm tính hợp lý kinh tế:
    - Giá bán price <= 0
    - Doanh số sales <= 0
    - Phí vận chuyển freight_value < 0
    - Đơn hàng chưa hoàn tất (nếu có cột order_status)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_cleaned, df_clean_audit)
    """
    initial_count = len(df)
    work_df = df.copy()

    # Điều kiện lọc hợp lệ
    cond_price = work_df["price"] > 0 if "price" in work_df.columns else pd.Series(True, index=work_df.index)
    cond_sales = work_df[config.TARGET_COLUMN] > 0 if config.TARGET_COLUMN in work_df.columns else pd.Series(True, index=work_df.index)
    cond_freight = work_df["freight_value"] >= 0 if "freight_value" in work_df.columns else pd.Series(True, index=work_df.index)
    cond_status = work_df["order_status"] == "delivered" if "order_status" in work_df.columns else pd.Series(True, index=work_df.index)

    cleaned_df = work_df[cond_price & cond_sales & cond_freight & cond_status].copy()
    cleaned_count = len(cleaned_df)
    dropped_count = initial_count - cleaned_count

    df_clean_audit = pd.DataFrame([
        {"Chỉ tiêu kiểm toán làm sạch (Sanitization)": "Tổng số bản ghi ban đầu", "Số lượng": f"{initial_count:,} dòng", "Tỷ lệ": "100.0%"},
        {"Chỉ tiêu kiểm toán làm sạch (Sanitization)": "Số bản ghi bị loại (Giá <= 0, Sales <= 0, Lỗi)", "Số lượng": f"{dropped_count:,} dòng", "Tỷ lệ": f"{(dropped_count / initial_count * 100):.2f}%"},
        {"Chỉ tiêu kiểm toán làm sạch (Sanitization)": "Số bản ghi hợp lệ giữ lại (Cleaned Data)", "Số lượng": f"{cleaned_count:,} dòng", "Tỷ lệ": f"{(cleaned_count / initial_count * 100):.2f}%"}
    ])

    return cleaned_df.reset_index(drop=True), df_clean_audit


def audit_outliers_table(df: pd.DataFrame, num_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    2. Bảng kiểm toán ngoại lai (Outliers Audit Table) theo quy tắc 1.5 * IQR.
    """
    if num_cols is None:
        num_cols = ["price", "freight_value", "freight_ratio", "discount_amount", "num_reviews", "ad_spend", "sales"]
    valid_cols = [c for c in num_cols if c in df.columns]

    records = []
    total = len(df)

    for col in valid_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        outliers_count = int(((s < lb) | (s > ub)).sum())
        outliers_pct = round((outliers_count / total) * 100, 2)

        records.append({
            "Thuộc tính số học": col,
            "Phân vị Q1 (25%)": f"{q1:,.2f}",
            "Trung vị (50%)": f"{float(s.median()):,.2f}",
            "Phân vị Q3 (75%)": f"{q3:,.2f}",
            "Ngưỡng kẹp dưới (Lower)": f"{lb:,.2f}",
            "Ngưỡng kẹp trên (Upper)": f"{ub:,.2f}",
            "Số mẫu ngoại lai": f"{outliers_count:,}",
            "Tỷ lệ ngoại lai (%)": f"{outliers_pct}%"
        })

    return pd.DataFrame(records)


class CustomStandardScaler:
    """
    Chuẩn hóa đặc trưng số học: Z = (X - mean) / (std + eps)
    Chỉ fit trên Train (80%), dùng tham số đã lưu để transform Val (10%) và Test (10%).
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "CustomStandardScaler":
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("CustomStandardScaler chưa được fit!")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class CustomOneHotEncoder:
    """
    Mã hóa biến phân loại (Categorical) thành ma trận nhị phân {0, 1}.
    Hỗ trợ handle_unknown='ignore': Gán vector 0 nếu gặp danh mục lạ ở Test set.
    """
    def __init__(self, handle_unknown: str = "ignore"):
        self.handle_unknown = handle_unknown
        self.categories_ = {}
        self.encoded_feature_names_ = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "CustomOneHotEncoder":
        self.categories_ = {}
        self.encoded_feature_names_ = []

        for col in columns:
            if col in df.columns:
                uniques = sorted([str(x) for x in df[col].dropna().unique()])
                self.categories_[col] = uniques
                for val in uniques:
                    clean_val = str(val).replace(" ", "_").replace("/", "_")
                    self.encoded_feature_names_.append(f"{col}_{clean_val}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.categories_:
            raise ValueError("CustomOneHotEncoder chưa được fit!")

        cols_arrays = []
        for col, uniques in self.categories_.items():
            if col in df.columns:
                series_str = df[col].astype(str)
                for val in uniques:
                    binary_col = (series_str == val).astype(float).values
                    cols_arrays.append(binary_col)
            else:
                for _ in uniques:
                    cols_arrays.append(np.zeros(len(df), dtype=float))

        if not cols_arrays:
            return np.empty((len(df), 0), dtype=float)

        return np.column_stack(cols_arrays)

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        return self.fit(df, columns).transform(df)


class CustomImputer:
    """
    Xử lý giá trị khuyết thiếu (Missing Values):
    - Điền Median cho các thuộc tính số học
    - Điền Mode cho các thuộc tính phân loại
    """
    def __init__(self):
        self.num_medians_ = {}
        self.cat_modes_ = {}

    def fit(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> "CustomImputer":
        for col in num_cols:
            if col in df.columns:
                val = float(df[col].dropna().median()) if not df[col].dropna().empty else 0.0
                self.num_medians_[col] = val

        for col in cat_cols:
            if col in df.columns:
                modes = df[col].dropna().mode()
                val = str(modes.iloc[0]) if not modes.empty else "other"
                self.cat_modes_[col] = val

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, median_val in self.num_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
            else:
                # Tự động điền giá trị trung vị mặc định nếu thiếu cột lúc inference
                df[col] = median_val

        for col, mode_val in self.cat_modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)
            else:
                # Tự động điền giá trị mode nếu thiếu cột lúc inference
                df[col] = mode_val

        return df

    def fit_transform(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, num_cols, cat_cols).transform(df)


class CustomOutlierCapper:
    """
    Kẹp biên giá trị ngoại lai theo phân vị IQR: [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    Giữ nguyên kích thước và tính liên tục thời gian của chuỗi dữ liệu.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "CustomOutlierCapper":
        for col in columns:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    q1 = float(s.quantile(0.25))
                    q3 = float(s.quantile(0.75))
                    iqr = q3 - q1
                    self.lower_bounds_[col] = q1 - self.factor * iqr
                    self.upper_bounds_[col] = q3 + self.factor * iqr

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.lower_bounds_:
            if col in df.columns:
                lb = self.lower_bounds_[col]
                ub = self.upper_bounds_[col]
                df[col] = df[col].clip(lower=lb, upper=ub)

        return df

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


def temporal_split_data(
    df: pd.DataFrame,
    timestamp_col: str = config.TIMESTAMP_COLUMN,
    train_ratio: float = config.TRAIN_SIZE,
    val_ratio: float = config.VAL_SIZE,
    test_ratio: float = config.TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Phân chia chuỗi thời gian độc lập 8:1:1 (Temporal Split).
    Bảo đảm: max(Train_timestamp) <= min(Val_timestamp) <= max(Val_timestamp) <= min(Test_timestamp).
    Triệt tiêu 100% hiện tượng Data Leakage (Rò rỉ dữ liệu tương lai).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0) or min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive and sum to 1.")
    if timestamp_col not in df:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")
    if df[timestamp_col].isna().any():
        raise ValueError("Temporal split requires non-missing timestamps.")
    df = df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Simultaneous items/orders stay together; ratios are approximate.
    if 0 < train_end < n:
        train_end = int(df[timestamp_col].searchsorted(df[timestamp_col].iloc[train_end], side="left"))
    if 0 < val_end < n:
        val_end = int(df[timestamp_col].searchsorted(df[timestamp_col].iloc[val_end], side="left"))
    if not 0 < train_end < val_end < n:
        raise ValueError("Not enough distinct timestamps for three non-empty partitions.")
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def weekly_split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split distinct forecast weeks 80/10/10, keeping every product together."""
    weeks = np.sort(df[config.TIMESTAMP_COLUMN].unique())
    train_end = int(len(weeks) * config.TRAIN_SIZE)
    val_end = int(len(weeks) * (config.TRAIN_SIZE + config.VAL_SIZE))
    if not 0 < train_end < val_end < len(weeks):
        raise ValueError("Not enough target weeks for train/validation/test.")
    return (
        df.loc[df[config.TIMESTAMP_COLUMN] < weeks[train_end]].copy(),
        df.loc[df[config.TIMESTAMP_COLUMN].between(weeks[train_end], weeks[val_end], inclusive="left")].copy(),
        df.loc[df[config.TIMESTAMP_COLUMN] >= weeks[val_end]].copy(),
    )


def random_split_data(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SIZE,
    val_ratio: float = config.VAL_SIZE,
    test_ratio: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Phân chia ngẫu nhiên (Random Split 8:1:1) để phục vụ đối sánh với Temporal Split.
    """
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return df.iloc[train_idx].copy().reset_index(drop=True),            df.iloc[val_idx].copy().reset_index(drop=True),            df.iloc[test_idx].copy().reset_index(drop=True)


def compare_splits_temporal_vs_random(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng đối sánh sự khác biệt bản chất giữa Temporal Split và Random Split.
    Chứng minh nguy cơ rò rỉ dữ liệu chuỗi thời gian.
    """
    tr_temp, val_temp, te_temp = temporal_split_data(df)
    tr_rand, val_rand, te_rand = random_split_data(df)

    t_col = config.TIMESTAMP_COLUMN
    records = [
        {
            "Phương pháp phân chia": "1. Temporal Split 8:1:1 (Chuẩn)",
            "Khoảng thời gian Train": f"{str(tr_temp[t_col].min())[:10]} đến {str(tr_temp[t_col].max())[:10]}",
            "Khoảng thời gian Test": f"{str(te_temp[t_col].min())[:10]} đến {str(te_temp[t_col].max())[:10]}",
            "Chồng lấn mốc thời gian": "HOÀN TOÀN KHÔNG (0 ngày giao thoa)",
            "Nguy cơ rò rỉ dữ liệu (Data Leakage)": "Được triệt tiêu 100% (Mô phỏng tương lai chuẩn)"
        },
        {
            "Phương pháp phân chia": "2. Random Split 8:1:1 (Lỗi phổ biến)",
            "Khoảng thời gian Train": f"{str(tr_rand[t_col].min())[:10]} đến {str(tr_rand[t_col].max())[:10]}",
            "Khoảng thời gian Test": f"{str(te_rand[t_col].min())[:10]} đến {str(te_rand[t_col].max())[:10]}",
            "Chồng lấn mốc thời gian": "XẢY RA 100% (Train/Test trộn lẫn mọi ngày)",
            "Nguy cơ rò rỉ dữ liệu (Data Leakage)": "Rất cao (Lấy tương lai đoán quá khứ, lạc quan ảo)"
        }
    ]

    return pd.DataFrame(records)


class CustomDataPipeline:
    """
    Đóng gói toàn bộ Pipeline tiền xử lý From-Scratch theo chuẩn Hướng Đối Tượng (OOP).
    
    Nguyên tắc cốt lõi trong Học Máy:
    - Luôn chia dữ liệu (Train/Validation/Test Split) TRƯỚC khi tiền xử lý.
    - CustomDataPipeline chỉ được `fit` duy nhất trên tập Train (train_df).
    - Các tập Validation và Test chỉ được `transform` dựa trên tham số đã học từ Train
      (giá trị trung vị imputer, ngưỡng capper, category vocabulary, mean/std scaler),
      triệt tiêu hoàn toàn nguy cơ rò rỉ dữ liệu (Data Leakage).
    """
    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        outlier_cols: Optional[List[str]] = None,
        use_log_target: bool = True,
        feature_engineer: Optional[SalesFeatureEngineer] = None,
        feature_selector: Optional[SalesFeatureSelectorScratch] = None
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.outlier_cols = outlier_cols if outlier_cols is not None else ["price", "freight_ratio", "discount_amount"]
        self.use_log_target = use_log_target
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector

        self.imputer = CustomImputer()
        self.outlier_capper = CustomOutlierCapper()
        self.encoder = CustomOneHotEncoder()
        self.scaler = CustomStandardScaler()
        self.feature_names_ = []

    def fit(self, train_df: pd.DataFrame) -> "CustomDataPipeline":
        work_df = train_df.copy()
        if self.feature_engineer is not None:
            self.feature_engineer.fit(work_df)
            work_df = self.feature_engineer.transform(work_df)

        if self.num_cols is None:
            self.num_cols = [c for c in config.NUMERICAL_COLS if c in work_df.columns]
        if self.cat_cols is None:
            self.cat_cols = [c for c in config.CATEGORICAL_COLS if c in work_df.columns]

        # 1. Fit Imputer
        self.imputer.fit(work_df, self.num_cols, self.cat_cols)
        train_imp = self.imputer.transform(work_df)

        # 2. Fit Outlier Capper
        valid_outlier_cols = [c for c in self.outlier_cols if c in train_imp.columns]
        self.outlier_capper.fit(train_imp, valid_outlier_cols)
        train_cap = self.outlier_capper.transform(train_imp)

        # 3. Fit One-Hot Encoder
        self.encoder.fit(train_cap, self.cat_cols)

        # 4. Fit Standard Scaler
        self.scaler.fit(train_cap[self.num_cols].values)

        self.feature_names_ = list(self.num_cols) + list(self.encoder.encoded_feature_names_)

        # 5. Fit Feature Selector nếu có
        if self.feature_selector is not None:
            # Tạo ma trận tạm để fit selector
            X_tmp_num = self.scaler.transform(train_cap[self.num_cols].values)
            X_tmp_cat = self.encoder.transform(train_cap)
            X_tmp = np.hstack([X_tmp_num, X_tmp_cat])
            
            if config.TARGET_COLUMN in train_cap.columns:
                y_tmp = train_cap[config.TARGET_COLUMN].values.astype(float)
                if self.use_log_target:
                    y_tmp = np.log1p(np.maximum(0.0, y_tmp))
                self.feature_selector.fit(X_tmp, y_tmp)
                if self.feature_selector.selected_indices_ is not None:
                    self.feature_names_ = [self.feature_names_[i] for i in self.feature_selector.selected_indices_]

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        work_df = df.copy()
        if self.feature_engineer is not None:
            work_df = self.feature_engineer.transform(work_df)

        # 1. Transform Imputer
        df_imp = self.imputer.transform(work_df)

        # 2. Transform Outlier Capper
        df_cap = self.outlier_capper.transform(df_imp)

        # 3. Transform One-Hot Encoder
        X_cat = self.encoder.transform(df_cap)

        # 4. Transform Standard Scaler
        X_num = self.scaler.transform(df_cap[self.num_cols].values)

        # 5. Ghép nối ma trận
        X = np.hstack([X_num, X_cat])

        # 6. Biến mục tiêu nếu có
        y = None
        if config.TARGET_COLUMN in df_cap.columns:
            y = df_cap[config.TARGET_COLUMN].values.astype(float)
            if self.use_log_target:
                y = np.log1p(np.maximum(0.0, y))

        # 7. Áp dụng Feature Selector nếu có
        if self.feature_selector is not None and getattr(self.feature_selector, "is_fitted_", False):
            X = self.feature_selector.transform(X)

        return X, y

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    def load(filepath: Path) -> "CustomDataPipeline":
        with open(filepath, "rb") as f:
            return pickle.load(f)


def build_full_preprocessing_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    use_log_target: bool = True,
    feature_engineer: Optional[SalesFeatureEngineer] = None
) -> Dict[str, Any]:
    """
    Hàm cầu nối thực thi trọn vẹn tiền xử lý cho Train/Val/Test qua CustomDataPipeline.
    """
    pipeline = CustomDataPipeline(num_cols=num_cols, cat_cols=cat_cols, use_log_target=use_log_target, feature_engineer=feature_engineer)
    pipeline.fit(train_df)

    X_train, y_train = pipeline.transform(train_df)
    X_val, y_val = pipeline.transform(val_df)
    X_test, y_test = pipeline.transform(test_df)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": pipeline.feature_names_,
        "pipeline": pipeline
    }


def export_processed_data(
    pipeline_data: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Xuất và lưu trữ ma trận đặc trưng đệm vào thư mục data/processed/.
    """
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    f_xtrain = output_dir / "X_train.npz"
    f_ytrain = output_dir / "y_train.npy"
    f_xval = output_dir / "X_val.npz"
    f_yval = output_dir / "y_val.npy"
    f_xtest = output_dir / "X_test.npz"
    f_ytest = output_dir / "y_test.npy"
    f_names = output_dir / "feature_names.json"

    sp.save_npz(f_xtrain, sp.csr_matrix(pipeline_data["X_train"]))
    np.save(f_ytrain, pipeline_data["y_train"])

    sp.save_npz(f_xval, sp.csr_matrix(pipeline_data["X_val"]))
    np.save(f_yval, pipeline_data["y_val"])

    sp.save_npz(f_xtest, sp.csr_matrix(pipeline_data["X_test"]))
    np.save(f_ytest, pipeline_data["y_test"])

    with open(f_names, "w", encoding="utf-8") as f:
        json.dump(pipeline_data["feature_names"], f, ensure_ascii=False, indent=2)

    return {
        "X_train": f_xtrain,
        "y_train": f_ytrain,
        "X_val": f_xval,
        "y_val": f_yval,
        "X_test": f_xtest,
        "y_test": f_ytest,
        "feature_names": f_names
    }
