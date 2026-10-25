"""
predictor.py - Dự đoán sales cho sản phẩm mới
================================================
- predict_sales(features, model, scaler): Dự đoán sales → giá trị số
- demo_prediction(model, scaler, X_test, y_test): Demo trên mẫu test
- predict_scenarios(model, scaler, scenarios): Dự đoán nhiều kịch bản
    (new product launch, marketing campaign, seasonal promotion)
"""
"""Dịch vụ suy luận , tái sử dụng pipeline và mô hình."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.persistence import load_artifacts
from src import config
from src.data.launch import HISTORY_COLUMNS


class InputValidationError(ValueError):
    """Lỗi dữ liệu do tầng kiểm tra của My tạo, có thể hiển thị có kiểm soát."""


def finite_number(value, name, minimum=0, maximum=None, strictly_positive=False):
    """Không chấp nhận bool, NaN, vô cực hoặc số ngoài miền hợp lệ."""
    if isinstance(value, (bool, np.bool_)):
        raise InputValidationError(f'{name} phải là số, không phải boolean.')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InputValidationError(f'{name} phải là số.') from None
    if not np.isfinite(number) or number < minimum or (strictly_positive and number <= 0):
        raise InputValidationError(f'{name} phải hữu hạn và {"lớn hơn 0" if strictly_positive else f">= {minimum}"}.')
    if maximum is not None and number > maximum:
        raise InputValidationError(f'{name} phải <= {maximum}.')
    return number


def validate_inputs(features, pipeline):
    """Kiểm tra cả lô trước khi dự báo; trả bản sao, giữ thứ tự hàng.

    Các biến lịch do WeeklySalesFeatureEngineer tính. Không tự điền lịch sử
    cho sản phẩm đang bán; NaN của biến đã cung cấp bị từ chối tại tầng dịch vụ.
    """
    if isinstance(features, dict):
        frame = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        frame = features.copy(deep=True)
    else:
        raise InputValidationError('Đầu vào phải là dict hoặc DataFrame.')
    if frame.empty or frame.columns.duplicated().any():
        raise InputValidationError('Dữ liệu rỗng hoặc trùng tên cột.')
    if 'category' not in frame and 'product_category_name' in frame:
        frame['category'] = frame['product_category_name']
    if 'price' in frame:
        for value in frame['price']:
            finite_number(value, 'price', strictly_positive=True)
    required = {'price', 'category', config.TIMESTAMP_COLUMN}
    derived = {'month_sin', 'month_cos', 'is_peak_shopping_season'}
    required.update(set(getattr(pipeline, 'num_cols', None) or []) - derived)
    required.update(getattr(pipeline, 'cat_cols', None) or [])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError('Thiếu cột: ' + ', '.join(missing))
    for col in required:
        if frame[col].isna().any():
            raise InputValidationError(f'{col} không được thiếu hoặc NaN.')
    dates = pd.to_datetime(frame[config.TIMESTAMP_COLUMN], errors='coerce')
    if dates.isna().any() or (dates.dt.dayofweek != 0).any():
        raise InputValidationError('Tuần dự báo phải bắt đầu vào thứ Hai hợp lệ.')
    frame[config.TIMESTAMP_COLUMN] = dates
    numeric = set(config.WEEKLY_NUMERICAL_COLS) | set(getattr(pipeline, 'num_cols', None) or [])
    numeric |= {'ad_spend', 'discount_rate', 'product_weight_g', 'product_photos_qty',
                'product_name_lenght', 'product_description_lenght'}
    for col in numeric & set(frame.columns):
        low = -1 if col in ('month_sin', 'month_cos') else 0
        high = 5 if col == 'past_review_rating' else (1 if col in derived | {'discount_rate'} else None)
        frame[col] = [finite_number(v, col, minimum=low, maximum=high,
                                  strictly_positive=col == 'price') for v in frame[col]]
    for col in {'category'} | set(getattr(pipeline, 'cat_cols', None) or []):
        if not frame[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise InputValidationError(f'{col} phải là chuỗi không rỗng.')
    return frame


def to_brl(values, use_log_target=True):
    """Đảo log đúng một lần; từ chối tràn số thay vì tạo doanh thu giả."""
    result = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError('Mô hình trả kết quả không hữu hạn.')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.expm1(result) if use_log_target else result.copy()
    if not np.isfinite(result).all():
        raise ValueError('Dự báo vượt miền số khi chuyển về BRL.')
    return np.maximum(0, result)


class SalesPredictor:
    """Chỉ transform/predict; tuyệt đối không fit lại trong ứng dụng."""
    def __init__(self, model, pipeline, cold_start_model=None, metadata=None):
        if not callable(getattr(model, 'predict', None)) or not callable(getattr(pipeline, 'transform', None)):
            raise ValueError('Artifact phải có model.predict và pipeline.transform.')
        self.model = model
        self.pipeline = pipeline
        self.cold_start_model = cold_start_model
        self.metadata = metadata or {}

    @classmethod
    def from_artifact(cls, filepath):
        # Chỉ nạp file nội bộ từ nhóm: pickle không dành cho nguồn không tin cậy.
        payload = load_artifacts(Path(filepath))
        if not isinstance(payload, dict) or not {'model', 'pipeline'} <= payload.keys():
            raise ValueError('Artifact không đúng định dạng save_artifacts của Task 2.')
        return cls(payload['model'], payload['pipeline'], payload.get('cold_start_model'), payload.get('metadata'))

    def predict(self, features):
        frame = validate_inputs(features, self.pipeline)
        # Bỏ nhãn nếu CSV chứa sẵn; suy luận không sử dụng doanh thu tuần đích.
        inputs = frame.drop(columns=['sales', 'log_sales'], errors='ignore')
        X, _ = self.pipeline.transform(inputs)
        predictions = to_brl(self.model.predict(X), getattr(self.pipeline, 'use_log_target', True))
        if len(predictions) != len(frame):
            raise ValueError('Số dự báo không khớp số hàng đầu vào.')
        result = frame.copy()
        result['predicted_revenue_brl'] = predictions
        return result


def predict_sales(features, model, pipeline=None, scaler=None):
    """API tương thích điểm gọi có sẵn; cần bộ pipeline đã fit."""
    # Test Task 1 gọi dạng cũ predict_sales(model, pipeline, features).
    if callable(getattr(features, 'predict', None)) and callable(getattr(model, 'transform', None)):
        features, model, pipeline = pipeline, features, model
    pipeline = pipeline if pipeline is not None else scaler
    if pipeline is None:
        raise ValueError('Cần cung cấp pipeline đã fit của Task 1.')
    values = SalesPredictor(model, pipeline).predict(features)['predicted_revenue_brl'].to_numpy()
    return float(values[0]) if isinstance(features, dict) else values


def demo_prediction(model, pipeline, X_test, y_test, n_samples=5):
    if n_samples <= 0:
        raise ValueError('n_samples phải lớn hơn 0.')
    size = min(n_samples, len(y_test))
    use_log = getattr(pipeline, 'use_log_target', True)
    actual = to_brl(y_test[:size], use_log)
    predicted = to_brl(model.predict(X_test[:size]), use_log)
    percentage = np.full(size, np.nan)
    np.divide(np.abs(predicted - actual) * 100, actual, out=percentage, where=actual != 0)
    return pd.DataFrame({'actual_revenue_brl': actual, 'predicted_revenue_brl': predicted,
                         'Error_Percentage_%': percentage})


# Giữ tên hàm đã được main.py tham chiếu mà không sửa main.py.
def predict_scenario_new_product(product_spec, model, pipeline, cold_start_model=None):
    from src.evaluation.listing_launch import new_product_scenario
    return new_product_scenario(product_spec, SalesPredictor(model, pipeline, cold_start_model))


def predict_scenario_marketing_campaign(base_product, ad_spend_levels, model=None, pipeline=None):
    return marketing_scenario(ad_spend_levels)


def calculate_discount_scenario(original_price, discount_rates, assumed_units, reference_revenue=None):
    return discount_scenario(original_price, discount_rates, assumed_units, reference_revenue)


def marketing_scenario(ad_spend_levels):
    """Olist không có dữ liệu quảng cáo: không ước lượng uplift/ROI."""
    levels = list(ad_spend_levels)
    if not levels:
        raise ValueError('Cần ít nhất một mức ngân sách.')
    return pd.DataFrame({
        'ad_spend_brl': [finite_number(v, 'ad_spend') for v in levels],
        'incremental_revenue_brl': np.nan,
        'roi': np.nan,
        'status': 'Chưa đủ dữ liệu marketing để ước lượng tác động.'})


def discount_scenario(original_price, discount_rates, assumed_units, reference_revenue=None):
    """Doanh thu = giá gốc × (1 − giảm giá) × sản lượng người dùng giả định."""
    price = finite_number(original_price, 'original_price', strictly_positive=True)
    units = finite_number(assumed_units, 'assumed_units')
    if not units.is_integer():
        raise ValueError('Sản lượng giả định phải là số nguyên không âm.')
    rates = list(discount_rates)
    if not rates:
        raise ValueError('Cần ít nhất một tỷ lệ giảm giá.')
    reference = price * units if reference_revenue is None else finite_number(reference_revenue, 'reference_revenue')
    rows = []
    for rate in rates:
        rate = finite_number(rate, 'discount_rate', maximum=1)
        discounted = price * (1 - rate)
        revenue = discounted * units
        required = np.ceil(reference / discounted) if discounted > 0 else (0 if reference == 0 else np.nan)
        if not np.isfinite(price * units) or not np.isfinite(revenue) or (discounted > 0 and not np.isfinite(required)):
            raise ValueError('Giá trị mô phỏng vượt miền số.')
        rows.append({'discount_rate': rate, 'discounted_price_brl': discounted,
                     'assumed_units': int(units), 'scenario_revenue_brl': revenue,
                     'reference_revenue_brl': reference, 'units_to_match_reference': required,
                     'baseline_revenue_brl': price * units,
                     'revenue_change_brl': revenue - price * units,
                     'method': 'Phép tính giả định, không phải dự báo tác động ML.'})
    return pd.DataFrame(rows)
