"""Integration checks for weekly targets, chronology, and deployment."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.builder import build_weekly_sales_dataset
from src.data.feature_engineering import WeeklySalesFeatureEngineer
from src.data.preprocessing import CustomDataPipeline, weekly_split_data
from src.deployment.predictor import predict_sales
from src.models.regressors import LinearRegressionFromScratch, PreviousWeekRegressorFromScratch
from src.models.persistence import save_artifacts, load_artifacts


def transactions():
    weeks = pd.date_range("2020-01-06", periods=14, freq="W-MON")
    rows = []
    for i, week in enumerate(weeks):
        for product in ["a", "b"]:
            if product == "a" and i == 4:
                continue
            rows.append({"product_id": product, "order_id": f"{product}{i}",
                         "order_item_id": 7, "price": 10.0 + i,
                         config.TIMESTAMP_COLUMN: week + pd.Timedelta(days=1),
                         "category": "books", "sales": 999999.0,
                         "ad_spend": 999.0, "age": 42})
    return pd.DataFrame(rows)


def test_weekly_target_and_zero_weeks():
    panel = build_weekly_sales_dataset(transactions())
    row = panel.loc[(panel.product_id == "a") & (panel[config.TIMESTAMP_COLUMN] == "2020-02-03")].iloc[0]
    assert row.sales == 0.0
    assert row.last_week_sales == 13.0
    assert "ad_spend" not in panel and "age" not in panel
    assert not panel.duplicated(["product_id", config.TIMESTAMP_COLUMN]).any()
    assert panel[config.TIMESTAMP_COLUMN].dt.dayofweek.eq(0).all()


def test_current_and_future_transactions_do_not_change_prior_inputs():
    source = transactions()
    before = build_weekly_sales_dataset(source)
    cutoff = pd.Timestamp("2020-02-17")
    changed = source.copy()
    changed.loc[changed[config.TIMESTAMP_COLUMN] >= cutoff, "price"] *= 100
    after = build_weekly_sales_dataset(changed)
    columns = [c for c in before.columns if c != "sales"]
    pd.testing.assert_frame_equal(before.loc[before[config.TIMESTAMP_COLUMN] <= cutoff, columns],
                                  after.loc[after[config.TIMESTAMP_COLUMN] <= cutoff, columns])


def test_real_pipeline_batch_and_artifact_invariance(tmp_path):
    panel = build_weekly_sales_dataset(transactions())
    train = panel.loc[panel[config.TIMESTAMP_COLUMN] < "2020-03-02"]
    test = panel.loc[panel[config.TIMESTAMP_COLUMN] >= "2020-03-02"]
    pipe = CustomDataPipeline(num_cols=config.WEEKLY_NUMERICAL_COLS,
                              cat_cols=config.WEEKLY_CATEGORICAL_COLS,
                              outlier_cols=[], feature_engineer=WeeklySalesFeatureEngineer())
    X, y = pipe.fit_transform(train)
    model = LinearRegressionFromScratch().fit(X, y)
    batch = predict_sales(model, pipe, test)
    alone = predict_sales(model, pipe, test.iloc[[0]])
    np.testing.assert_allclose(batch[:1], alone)
    path = save_artifacts(pipe, model, tmp_path / "weekly.pkl")
    loaded = load_artifacts(path)
    np.testing.assert_allclose(batch, predict_sales(loaded["model"], loaded["pipeline"], test))
    lag = pipe.num_cols.index("last_week_sales")
    baseline = PreviousWeekRegressorFromScratch(lag, pipe.scaler.mean_[lag], pipe.scaler.std_[lag])
    np.testing.assert_allclose(np.expm1(baseline.predict(pipe.transform(test)[0])), test.last_week_sales, atol=1e-8)
    np.testing.assert_allclose(np.expm1(y), train.sales)
    from src.deployment.predictor import demo_prediction
    zero_demo = demo_prediction(model, pipe, X[:1], np.zeros(1), n_samples=1)
    assert np.isnan(zero_demo["Error_Percentage_%"].iloc[0])


def test_forecast_week_required():
    engineer = WeeklySalesFeatureEngineer().fit(pd.DataFrame())
    with pytest.raises(ValueError):
        engineer.transform(pd.DataFrame({"price": [10]}))
    with pytest.raises(ValueError):
        engineer.transform(pd.DataFrame({config.TIMESTAMP_COLUMN: ["2020-01-07"]}))


def test_weekly_split_is_disjoint_and_keeps_zero_targets():
    panel = build_weekly_sales_dataset(transactions())
    train, val, test = weekly_split_data(panel)
    assert len(train) + len(val) + len(test) == len(panel)
    assert train[config.TIMESTAMP_COLUMN].max() < val[config.TIMESTAMP_COLUMN].min()
    assert val[config.TIMESTAMP_COLUMN].max() < test[config.TIMESTAMP_COLUMN].min()
    assert (train.sales == 0).any()


class ConstantSalesModel:
    def __init__(self, sales):
        self.sales = sales

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.log1p(self.sales))


def test_tuning_uses_brl_objective():
    from src.models.tuning import grid_search_from_scratch
    X, y = np.zeros((2, 1)), np.log1p([0.0, 100.0])
    result = grid_search_from_scratch(ConstantSalesModel, {"sales": [0.0, 50.0]}, X, y, X, y, scoring="rmse_brl")
    assert result["best_params"] == {"sales": 50.0}
    assert result["best_score"] == pytest.approx(50.0)


def test_weekly_cache_roundtrip_and_invalidation(tmp_path, monkeypatch):
    import main
    names = ["CACHE_X_TRAIN", "CACHE_X_VAL", "CACHE_X_TEST", "CACHE_Y_TRAIN",
             "CACHE_Y_VAL", "CACHE_Y_TEST", "CACHE_PIPELINE", "CACHE_FEATURE_NAMES"]
    for name in names:
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).name)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(main.visualization, "plot_temporal_split_distribution", lambda **kwargs: None)
    panel = build_weekly_sales_dataset(transactions())
    fresh = main.step_5_split_and_pipeline(panel, force_reprocess=True)
    cached = main.step_5_split_and_pipeline(panel)
    np.testing.assert_allclose(fresh[0].toarray(), cached[0].toarray())
    np.testing.assert_array_equal(fresh[3], cached[3])
    changed = panel.copy()
    changed.loc[0, "sales"] += 10
    updated = main.step_5_split_and_pipeline(changed)
    assert fresh[3][0] != updated[3][0]
