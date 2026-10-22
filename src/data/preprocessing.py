"""
preprocessing.py - Tiền xử lý dữ liệu chuỗi thời gian theo tuần.
Tiền xử lý luôn fit trên tập train lịch sử và chỉ transform các giai đoạn sau.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def split_product_week_data(
    panel: pd.DataFrame,
    target_col: str = config.WEEKLY_TARGET_COLUMN,
    week_col: str = config.WEEKLY_TIMESTAMP_COLUMN,
    train_size: float = 0.8,
    validation_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Chia theo thứ tự thời gian và nguyên tuần để ngăn rò rỉ dữ liệu."""
    if target_col not in panel.columns or week_col not in panel.columns:
        raise ValueError(f"Panel must contain '{target_col}' and '{week_col}'.")
    if not 0 < train_size < 1 or not 0 < validation_size < 1:
        raise ValueError("train_size and validation_size must each be between 0 and 1.")
    if train_size + validation_size >= 1:
        raise ValueError("train_size + validation_size must be less than 1.")

    ordered = panel.copy()
    ordered[week_col] = pd.to_datetime(ordered[week_col], errors="raise")
    if ordered[week_col].isna().any():
        raise ValueError(f"Column '{week_col}' cannot contain missing timestamps.")
    ordered = ordered.sort_values(week_col, kind="stable")
    weeks = np.sort(ordered[week_col].dropna().unique())
    if len(weeks) < 3:
        raise ValueError("At least three distinct weeks are required for a temporal split.")

    rows_per_week = (
        ordered.groupby(week_col, sort=True).size().reindex(weeks).to_numpy()
    )
    cumulative_rows = np.cumsum(rows_per_week)
    total_rows = cumulative_rows[-1]
    train_end = int(
        np.argmin(np.abs(cumulative_rows[:-2] - total_rows * train_size)) + 1
    )
    validation_candidates = cumulative_rows[train_end:-1]
    validation_end = train_end + int(
        np.argmin(
            np.abs(
                validation_candidates
                - total_rows * (train_size + validation_size)
            )
        )
    ) + 1
    train_cutoff = weeks[train_end]
    validation_cutoff = weeks[validation_end]

    train = ordered.loc[ordered[week_col] < train_cutoff].copy()
    validation = ordered.loc[
        (ordered[week_col] >= train_cutoff)
        & (ordered[week_col] < validation_cutoff)
    ].copy()
    test = ordered.loc[ordered[week_col] >= validation_cutoff].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition.")
    return {"train": train, "validation": validation, "test": test}


def build_weekly_preprocessor(
    numerical_columns: list[str] = None,
    categorical_columns: list[str] = None,
    category_values: list[str] = None,
) -> ColumnTransformer:
    """Tạo pipeline điền khuyết, chuẩn hóa, one-hot; chỉ fit trên train."""
    numerical_columns = (
        list(config.WEEKLY_NUMERICAL_FEATURES)
        if numerical_columns is None
        else list(numerical_columns)
    )
    categorical_columns = (
        list(config.WEEKLY_CATEGORICAL_FEATURES)
        if categorical_columns is None
        else list(categorical_columns)
    )
    if not numerical_columns and not categorical_columns:
        raise ValueError("At least one numerical or categorical feature is required.")
    if category_values is not None:
        if len(categorical_columns) != 1:
            raise ValueError("category_values can only be used with one categorical column.")
        category_values = sorted(set(category_values))
        if not category_values:
            raise ValueError("category_values must contain at least one known category.")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[category_values] if category_values is not None else "auto",
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
