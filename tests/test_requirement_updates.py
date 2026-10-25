import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics


def test_target_scale_and_evaluation():
    y_true = np.array([1000.0, 2000.0, 3000.0])
    y_pred_log = np.log1p(
        np.array([900.0, 2100.0, 2900.0])
    )

    result = calculate_metrics(
        y_true,
        y_pred_log,
        predictions_are_log=True,
    )

    assert result["scale"] == "BRL"
    assert result["MAE"] == pytest.approx(100.0)
    assert result["RMSE"] == pytest.approx(100.0)


def test_brl_scale_is_explicit():
    result = calculate_metrics(
        [100.0, 200.0],
        [90.0, 210.0],
    )

    assert result["scale"] == "BRL"
    assert result["MSE"] == pytest.approx(100.0)
    assert result["RMSE"] == pytest.approx(10.0)
