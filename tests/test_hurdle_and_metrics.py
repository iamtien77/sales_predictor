import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_metrics,
    inverse_log_target,
    wape,
)


def test_mse_mae_rmse_r2():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 280.0])

    result = calculate_metrics(y_true, y_pred)

    assert result["MSE"] == pytest.approx(200.0)
    assert result["MAE"] == pytest.approx(13.3333333333)
    assert result["RMSE"] == pytest.approx(np.sqrt(200.0))
    assert result["R2"] == pytest.approx(
        1 - 200 / 20000
    )


def test_wape():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([80.0, 240.0])

    assert wape(y_true, y_pred) == pytest.approx(60 / 300)


def test_wape_zero_total_actual_returns_nan():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([10.0, 20.0])

    assert np.isnan(wape(y_true, y_pred))


def test_inverse_log_target():
    original = np.array([0.0, 100.0, 1000.0])
    transformed = np.log1p(original)

    restored = inverse_log_target(transformed)

    np.testing.assert_allclose(restored, original)


def test_metrics_inverse_log_to_brl():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred_brl = np.array([90.0, 210.0, 280.0])
    y_pred_log = np.log1p(y_pred_brl)

    result = calculate_metrics(
        y_true,
        y_pred_log,
        predictions_are_log=True,
    )

    assert result["scale"] == "BRL"
    assert result["MAE"] == pytest.approx(
        (10 + 10 + 20) / 3
    )


def test_no_double_inverse_when_prediction_is_already_brl():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 210.0])

    result = calculate_metrics(
        y_true,
        y_pred,
        predictions_are_log=False,
    )

    assert result["MAE"] == pytest.approx(10.0)
