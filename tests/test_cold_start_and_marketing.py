"""Kiểm tra độc lập ba kịch bản."""
import numpy as np
import pandas as pd
import pytest
from src.deployment.predictor import marketing_scenario, discount_scenario
from src.evaluation.listing_launch import new_product_scenario
from src.deployment.predictor import SalesPredictor
from src import config
from src.data.launch import HISTORY_COLUMNS


class FixedPipeline:
    num_cols = config.WEEKLY_NUMERICAL_COLS
    cat_cols = ['category']
    use_log_target = True
    def transform(self, frame):
        self.seen = frame.copy()
        return np.zeros((len(frame), 2)), None


class FixedModel:
    def predict(self, X): return np.full(len(X), np.log1p(25))


class ColdModel:
    def predict(self, frame): return np.full(len(frame), 35.0)


def test_new_product_does_not_modify_input():
    spec = {'category': 'toys', 'price': 100, config.TIMESTAMP_COLUMN: '2026-10-26', 'last_week_sales': 999}
    original = dict(spec)
    pipeline = FixedPipeline()
    result = new_product_scenario(spec, SalesPredictor(FixedModel(), pipeline))
    assert result['predicted_revenue_brl'] == pytest.approx(25)
    assert result['method'] == 'weekly_model_zero_history'
    assert (pipeline.seen[HISTORY_COLUMNS] == 0).all().all()
    assert spec == original
    result = new_product_scenario(spec, SalesPredictor(FixedModel(), pipeline, ColdModel()))
    assert result['predicted_revenue_brl'] == 35  # Không đảo log hai lần.


def test_marketing_no_invented_uplift():
    result = marketing_scenario([0, 100, 500])
    assert result.ad_spend_brl.tolist() == [0, 100, 500]
    assert result.incremental_revenue_brl.isna().all()
    assert result.roi.isna().all()


@pytest.mark.parametrize('levels', [[], [-1], [np.inf], ['bad']])
def test_bad_marketing(levels):
    with pytest.raises(ValueError): marketing_scenario(levels)


def test_discount_hand_calculation_and_full_discount():
    result = discount_scenario(100, [0, .2, 1], 10)
    assert result.scenario_revenue_brl.tolist() == [1000, 800, 0]
    assert result.units_to_match_reference.iloc[1] == 13
    assert pd.isna(result.units_to_match_reference.iloc[2])
    assert discount_scenario(100, [1], 0).units_to_match_reference.iloc[0] == 0


@pytest.mark.parametrize('price,rates,units', [(0,[0],1), (100,[-.1],1), (100,[1.1],1),
    (100,[],1), (100,[0],1.5), (100,[0],np.nan), (100,[0],True)])
def test_bad_discount(price, rates, units):
    with pytest.raises(ValueError): discount_scenario(price, rates, units)


def test_prepare_launch_input_schema_and_copy():
    from src.data.launch import prepare_launch_input
    from copy import deepcopy
    spec = {'product_category_name': 'toys', 'price': 100,
            config.TIMESTAMP_COLUMN: '2026-10-26', 'sales': 999,
            'log_sales': 99, 'last_week_sales': 10}
    before = deepcopy(spec)
    result = prepare_launch_input(spec)
    assert spec == before
    assert result['category'] == 'toys'
    assert all(result[c] == 0 for c in HISTORY_COLUMNS)
    assert 'sales' not in result and 'log_sales' not in result
    for invalid in [None, [], 'toys']:
        with pytest.raises(ValueError):
            prepare_launch_input(invalid)


def test_real_task2_cold_start_category_fallback_is_not_refitted():
    import pickle
    from src.models.cold_start import CategoryMedianColdStartBaseline
    model = CategoryMedianColdStartBaseline().fit(pd.DataFrame({
        'product_category_name': ['toys', 'toys', 'books'],
        'launch_sales_brl': [20., 40., 90.]}))
    snapshot = pickle.dumps(model)
    service = SalesPredictor(FixedModel(), FixedPipeline(), model)
    spec = {'category': 'toys', 'price': 100, config.TIMESTAMP_COLUMN: '2026-10-26'}
    result = new_product_scenario(spec, service)
    assert result['predicted_revenue_brl'] == 30.
    assert result['method'] == 'cold_start_model_brl'
    result = new_product_scenario({**spec, 'category': 'new_category'}, service)
    assert result['predicted_revenue_brl'] == 40.
    assert pickle.dumps(model) == snapshot


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_invalid_cold_start_prediction_is_rejected(value):
    class BadColdModel:
        def predict(self, frame): return np.array([value])
    service = SalesPredictor(FixedModel(), FixedPipeline(), BadColdModel())
    with pytest.raises(ValueError):
        new_product_scenario({'category': 'toys', 'price': 100,
            config.TIMESTAMP_COLUMN: '2026-10-26'}, service)
