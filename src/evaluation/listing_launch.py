"""Train genuine listing-date cold-start ONLY when an external catalog exists.

python -m src.evaluation.listing_launch --catalog catalog.csv --transactions sales.csv
    --observed-until YYYY-MM-DD --output results/listing_launch
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from src.data.launch import build_listing_launch_dataset
from src.evaluation.experiments import rolling_cohorts, scores
from src.models.cold_start import CategoryMedianColdStartBaseline, ColdStartProductRegressor


def evaluate_listing_launch(catalog, transactions, observed_until, output):
    df = build_listing_launch_dataset(catalog, transactions, observed_until)
    rows = []
    model = None
    for fold, train, test, start, stop in rolling_cohorts(df, 'forecast_origin', 'target_window_end'):
        model = ColdStartProductRegressor().fit(train)
        base = CategoryMedianColdStartBaseline().fit(train)
        for name, pred in [('ContentTree', model.predict(test)), ('CategoryMedian', base.predict(test)),
                           ('Zero', test.price.to_numpy() * 0), ('OneUnit', test.price.to_numpy())]:
            rows.append({'Fold': fold, 'Model': name, 'Start': str(start), 'EndExclusive': str(stop),
                         'TrainCount': len(train), 'TestCount': len(test),
                         'TestZeroFraction': float(test.launch_sales_brl.eq(0).mean()),
                         **scores(test.launch_sales_brl, pred)})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'evaluation.csv', index=False)
    metadata = {'target': 'listing_date_seven_day_revenue', 'observed_until': str(observed_until),
                'dataset_sha256': hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest(),
                'selection': 'fixed tree; saved last-fold training model, no Test refit or selection',
                'catalog_contract': 'complete catalog including unsold products; attributes available by listing',
                'last_train_label_end': str(train.target_window_end.max()),
                'last_test_origin': str(start)}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (out / 'listing_model.pkl').open('wb') as f:
        pickle.dump({'model': model, 'metadata': metadata}, f)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--transactions', type=Path, required=True)
    parser.add_argument('--observed-until', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_listing_launch(pd.read_csv(args.catalog), pd.read_csv(args.transactions),
                                  args.observed_until, args.output).to_string(index=False))


if __name__ == '__main__':
    main()
