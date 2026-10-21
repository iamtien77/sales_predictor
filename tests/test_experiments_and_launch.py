import pandas as pd
import pytest

from src.data.launch import build_listing_launch_dataset
from src.evaluation.experiments import rolling_cohorts


def test_listing_dataset_includes_unsold_products_and_uses_catalog_price():
    catalog = pd.DataFrame({'product_id': ['sold', 'unsold', 'incomplete'],
                            'listing_date': ['2020-01-01', '2020-01-01', '2020-01-20'],
                            'features_available_at': ['2019-12-31'] * 3,
                            'listed_price': [10., 20., 30.], 'product_category_name': ['x'] * 3})
    transactions = pd.DataFrame({'product_id': ['sold', 'sold'],
                                 'purchased_at': ['2020-01-02', '2020-01-08'], 'revenue': [999., 100.]})
    df = build_listing_launch_dataset(catalog, transactions, '2020-01-21').set_index('product_id')
    assert df.loc['sold', 'price'] == 10.
    assert df.loc['sold', 'launch_sales_brl'] == 999.
    assert df.loc['unsold', 'launch_sales_brl'] == 0.
    assert 'incomplete' not in df.index


def test_catalog_features_after_origin_are_rejected():
    catalog = pd.DataFrame({'product_id': ['p'], 'listing_date': ['2020-01-01'],
                            'features_available_at': ['2020-01-02'], 'listed_price': [10.],
                            'product_category_name': ['x']})
    events = pd.DataFrame(columns=['product_id', 'purchased_at', 'revenue'])
    with pytest.raises(ValueError, match='available'):
        build_listing_launch_dataset(catalog, events, '2020-02-01')


def test_rolling_test_windows_are_disjoint_and_train_labels_complete():
    dates = pd.date_range('2020-01-01', periods=100)
    df = pd.DataFrame({'origin': dates, 'end': dates + pd.Timedelta(days=7)})
    seen = set()
    folds = list(rolling_cohorts(df, 'origin', 'end'))
    assert len(folds) == 3
    for _, train, test, start, _ in folds:
        assert train.end.max() <= start
        assert train.origin.max() < test.origin.min()
        assert seen.isdisjoint(test.index)
        seen.update(test.index)


def test_listing_training_exports_separate_model_with_zero_sale_cohorts(tmp_path):
    # Synthetic fixture validates plumbing only, never reported as real-world accuracy.
    from src.evaluation.listing_launch import evaluate_listing_launch
    origins = pd.date_range('2020-01-01', periods=100)
    catalog = pd.DataFrame({'product_id': [str(i) for i in range(100)],
                            'listing_date': origins, 'features_available_at': origins,
                            'listed_price': [10.] * 100, 'product_category_name': ['x'] * 100})
    events = pd.DataFrame({'product_id': [str(i) for i in range(0, 100, 2)],
                           'purchased_at': origins[::2] + pd.Timedelta(days=1), 'revenue': [10.] * 50})
    result = evaluate_listing_launch(catalog, events, '2020-06-01', tmp_path)
    assert len(result) == 12
    assert result.TestZeroFraction.between(.4, .6).all()
    assert (tmp_path / 'listing_model.pkl').exists()
