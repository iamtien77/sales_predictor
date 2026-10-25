"""Kịch bản ra mắt sản phẩm."""
import numpy as np
from src.data.launch import prepare_launch_input
from src.deployment.predictor import validate_inputs


def new_product_scenario(product_spec, predictor):
    """Không lịch sử: dùng cold-start nếu có, nếu không nêu rõ giả định lag=0."""
    spec = prepare_launch_input(product_spec)
    frame = validate_inputs(spec, predictor.pipeline)
    if predictor.cold_start_model is not None:
        frame['product_category_name'] = frame['category']
        # ColdStartProductRegressor của Task 2 trả BRL sẵn, không expm1 lần nữa.
        value = np.asarray(predictor.cold_start_model.predict(frame), dtype=float).reshape(-1)
        if value.size != 1 or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Mô hình cold-start trả doanh thu không hợp lệ.')
        method = 'cold_start_model_brl'
        revenue = float(value[0])
        note = 'Dùng mô hình cold-start đã huấn luyện; cần kiểm chứng trên sản phẩm mới.'
    else:
        revenue = float(predictor.predict(frame)['predicted_revenue_brl'].iloc[0])
        method = 'weekly_model_zero_history'
        note = 'Ước lượng tham khảo bằng mô hình tuần với lịch sử bằng 0; chưa kiểm định cold-start/tuần ra mắt.'
    return {'predicted_revenue_brl': revenue, 'method': method, 'note': note}


