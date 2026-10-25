"""Chuẩn bị đầu vào ra mắt sản phẩm """
from collections.abc import Mapping
from copy import deepcopy
from src import config

HISTORY_COLUMNS = [c for c in config.WEEKLY_NUMERICAL_COLS
                   if c not in ('price', 'month_sin', 'month_cos', 'is_peak_shopping_season')]


def prepare_launch_input(product_spec):
    """Trả bản sao một sản phẩm mới với lịch sử bằng 0 theo chế độ bản cũ.

    Đây là giả định cold-start tường minh, không phải lịch sử quan sát thực tế.
    Việc kiểm tra kiểu số, giá và tuần được dịch vụ predictor thực hiện sau đó.
    """
    if not isinstance(product_spec, Mapping):
        raise ValueError('Thông tin sản phẩm mới phải là dictionary.')
    spec = deepcopy(dict(product_spec))
    if 'category' not in spec and 'product_category_name' in spec:
        spec['category'] = spec['product_category_name']
    for col in HISTORY_COLUMNS:
        spec[col] = 0.0
    spec.pop('sales', None)
    spec.pop('log_sales', None)
    return spec
