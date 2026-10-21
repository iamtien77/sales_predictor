import pandas as pd
import numpy as np
import pytest

from src.data.builder import build_cold_start_dataset


def test_first_sale_price_does_not_use_later_transactions(tmp_path):
    pd.DataFrame({
        'order_id': ['a', 'b', 'c', 'end'],
        'order_status': ['delivered'] * 4,
        'order_purchase_timestamp': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-02-01'],
    }).to_csv(tmp_path / 'olist_orders_dataset.csv', index=False)
    items = pd.DataFrame({'order_id': ['a', 'b', 'c', 'end'],
                          'product_id': ['p', 'p', 'p', 'late'],
                          'price': [10., 100., 100., 20.]})
    items.to_csv(tmp_path / 'olist_order_items_dataset.csv', index=False)
    pd.DataFrame({'product_id': ['p', 'late'], 'product_category_name': ['x', 'x']}).to_csv(
        tmp_path / 'olist_products_dataset.csv', index=False)
    before = build_cold_start_dataset(tmp_path).set_index('product_id')
    items.loc[items.order_id.isin(['b', 'c']), 'price'] = 1000.
    items.to_csv(tmp_path / 'olist_order_items_dataset.csv', index=False)
    after = build_cold_start_dataset(tmp_path).set_index('product_id')
    assert before.loc['p', 'price'] == after.loc['p', 'price'] == 10.
    assert before.loc['p', 'launch_sales_brl'] != after.loc['p', 'launch_sales_brl']
    assert 'late' not in after.index  # Its seven-day target is not fully observed.
    assert after.loc['p', 'first_purchase_time'] == pd.Timestamp('2020-01-01')


def test_temporal_holdout_purges_labels_crossing_test_origin():
    from src.evaluation.cold_start import temporal_split
    origins = pd.date_range('2020-01-01', periods=40)
    df = pd.DataFrame({'product_id': range(40), 'first_purchase_time': origins,
                       'target_window_end': origins + pd.Timedelta(days=7)})
    train, test, cutoff = temporal_split(df)
    assert train.target_window_end.max() <= test.first_purchase_time.min()
    assert set(train.product_id).isdisjoint(test.product_id)
    assert len(train) + len(test) < len(df)
    assert cutoff == pd.Timestamp('2020-02-02')


def test_temporal_experiment_exports_price_baseline_and_manifest(tmp_path, monkeypatch):
    import hashlib
    import json
    from src.evaluation import cold_start
    origins = pd.date_range('2020-01-01', periods=50)
    frame = pd.DataFrame({'product_id': range(50), 'first_purchase_time': origins,
                          'target_window_end': origins + pd.Timedelta(days=7),
                          'price': [10.] * 50, 'launch_units': [1] * 50,
                          'launch_sales_brl': [10.] * 50, 'product_category_name': ['x'] * 50})
    monkeypatch.setattr(cold_start, 'build_cold_start_dataset', lambda raw_dir: frame)
    _, summary, manifest = cold_start.train_and_evaluate(output_dir=tmp_path)
    assert summary.set_index('Model').loc['SingleUnitPriceBaseline', 'MAE_BRL'] == 0
    assert summary.Test_Count.nunique() == 1
    assert manifest['train_last_label_end'] <= manifest['test_first_origin']
    assert json.loads((tmp_path / 'cold_start_manifest.json').read_text())['price_source'].startswith('first_observed')
    assert manifest['manifest_sha256'] == hashlib.sha256((tmp_path / 'cold_start_manifest.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('kwargs', [
    {'mode': 'typo'}, {'elasticity': -1}, {'elasticity': np.nan},
    {'reference_spend': 0}, {'gross_margin': 1.1}, {'gross_margin': np.nan},
])
def test_invalid_marketing_assumptions_are_rejected_before_prediction(kwargs):
    from src.deployment.predictor import predict_scenario_marketing_campaign
    with pytest.raises(ValueError):
        predict_scenario_marketing_campaign({'price': 10}, [100], None, None, **kwargs)


def test_cold_fallback_is_explicitly_not_trained():
    from src.deployment.predictor import predict_cold_start_product
    result = predict_cold_start_product({'price': 10})
    assert result['model_engine'] == 'RuleBasedSingleUnitFallback'
    assert 'chưa có mô hình' in result['interpretation']
