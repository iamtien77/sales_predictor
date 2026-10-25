"""Kiểm thử hợp đồng dịch vụ dự báo."""
import numpy as np
import pandas as pd
import pytest
from src import config

from tests.test_deployment import service


def test_my_prediction_schema_and_output_contract(service):
    predictor, frame = service
    from src.deployment.predictor import InputValidationError
    for column in ['price', 'category', config.TIMESTAMP_COLUMN]:
        with pytest.raises(InputValidationError, match='Thiếu cột'):
            predictor.predict(frame.drop(columns=column))
    result = predictor.predict(frame)
    assert len(result) == len(frame)
    assert result.index.equals(frame.index)
    assert np.isfinite(result.predicted_revenue_brl).all()
    assert (result.predicted_revenue_brl >= 0).all()


def test_my_batch_rejects_entire_invalid_input(service):
    predictor, frame = service
    frame.loc[frame.index[-1], 'price'] = np.inf
    with pytest.raises(ValueError):
        predictor.predict(frame)


def test_my_duplicate_columns_and_invalid_category(service):
    predictor, frame = service
    with pytest.raises(ValueError):
        predictor.predict(pd.concat([frame, frame[['price']]], axis=1))
    for value in [None, ' ', 1, True]:
        invalid = frame.copy()
        invalid['category'] = value
        with pytest.raises(ValueError):
            predictor.predict(invalid)


def test_my_rejects_model_output_wrong_length(service):
    predictor, frame = service
    class BadModel:
        def predict(self, X):
            return np.array([1.0])
    predictor.model = BadModel()
    with pytest.raises(ValueError, match='khớp'):
        predictor.predict(frame)
