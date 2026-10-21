"""Kiểm thử hồi quy tại interface tiền xử lý tuần và suy luận công khai."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.feature_engineering import WeeklySalesFeatureEngineer
from src.deployment.predictor import predict_sales
from src.models.regressors import MeanRegressorFromScratch
from src.data.preprocessing import CustomDataPipeline


@pytest.mark.parametrize("price", [-10.0, 0.0, np.inf, -np.inf, "invalid"])
def test_weekly_transform_rejects_invalid_provided_price(price):
    engineer = WeeklySalesFeatureEngineer().fit(pd.DataFrame())
    data = pd.DataFrame({config.TIMESTAMP_COLUMN: ["2020-01-06"], "price": [price]})
    with pytest.raises(ValueError, match="price"):
        engineer.transform(data)


def test_public_prediction_rejects_negative_price_with_plain_pipeline():
    train = pd.DataFrame({"price": [10., 20., 30.], "category": ["a"] * 3, "sales": [1., 2., 3.]})
    pipe = CustomDataPipeline(num_cols=["price"], cat_cols=["category"])
    X, y = pipe.fit_transform(train)
    model = MeanRegressorFromScratch().fit(X, y)
    with pytest.raises(ValueError, match="price"):
        predict_sales(model, pipe, {"price": -10., "category": "a"})


def test_weekly_cli_preparation_runs_real_steps_two_and_three(capsys):
    import main
    weeks = pd.date_range("2020-01-06", periods=12, freq="W-MON")
    items = pd.DataFrame({"product_id": ["p"] * 11, "order_id": [str(i) for i in range(11)],
                          "order_item_id": [1] * 11, "price": [10.] * 11,
                          config.TIMESTAMP_COLUMN: weeks.delete(4), "category": ["a"] * 11})
    cleaned = main.step_2_clean_and_preprocess(items)
    panel = main.step_3_feature_engineering(cleaned)
    assert {"last_week_sales", "sales_mean_4_weeks", "sales"}.issubset(panel)
    assert panel.loc[panel[config.TIMESTAMP_COLUMN].eq(weeks[4]), "sales"].iloc[0] == 0
    output = capsys.readouterr().out
    assert "BƯỚC 2" in output and "BƯỚC 3" in output


def test_cli_plot_saves_without_noninteractive_show_warning(tmp_path):
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    from src.evaluation.visualization import plot_target_sales_distribution
    matplotlib.use("Agg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_target_sales_distribution(pd.DataFrame({"sales": [0., 10., 20.]}), tmp_path / "target.png")
    assert (tmp_path / "target.png").exists()
    assert not any("non-interactive" in str(w.message) for w in caught)
    plt.close("all")
