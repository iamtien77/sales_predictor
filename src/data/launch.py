"""Prospective launch cohort from an explicitly supplied, complete catalog.

Unlike Olist first-sale proxies, catalog rows include products with no purchases.
No catalog is inferred from transaction records.
"""
import numpy as np
import pandas as pd


def build_listing_launch_dataset(catalog, transactions, observed_until):
    """Catalog: one row/product, listing_date, listed_price, features_available_at.

    All supplied product attributes must be a snapshot known by listing_date.
    Transactions: product_id, purchased_at, revenue (realized line revenue).
    Caller certifies full catalog and transaction coverage through observed_until.
    Predict total revenue in [listing_date, listing_date + 7 days), including zeros.
    """
    required = {'product_id', 'listing_date', 'listed_price', 'features_available_at',
                'product_category_name'}
    if not required.issubset(catalog):
        raise ValueError(f'Missing catalog columns: {sorted(required - set(catalog))}')
    if not {'product_id', 'purchased_at', 'revenue'}.issubset(transactions):
        raise ValueError('Transactions require product_id, purchased_at, revenue')
    data = catalog.copy()
    if data.product_id.isna().any() or data.product_id.duplicated().any():
        raise ValueError('Catalog must have one non-null row per product')
    for col in ['listing_date', 'features_available_at']:
        data[col] = pd.to_datetime(data[col], errors='raise')
        if data[col].isna().any():
            raise ValueError(f'Missing {col}')
    if (data.features_available_at > data.listing_date).any():
        raise ValueError('Catalog features must be available by listing time')
    data['price'] = pd.to_numeric(data.listed_price, errors='raise')
    if (~np.isfinite(data.price) | (data.price <= 0)).any():
        raise ValueError('Listed price must be finite and positive')
    end = pd.Timestamp(observed_until)
    if pd.isna(end):
        raise ValueError('Explicit observation cutoff required')
    data['target_window_end'] = data.listing_date + pd.Timedelta(days=7)
    data = data.loc[data.target_window_end <= end].copy()
    events = transactions.copy()
    events['purchased_at'] = pd.to_datetime(events.purchased_at, errors='raise')
    events['revenue'] = pd.to_numeric(events.revenue, errors='raise')
    if events[['product_id', 'purchased_at']].isna().any().any() or (~np.isfinite(events.revenue) | (events.revenue < 0)).any():
        raise ValueError('Invalid transaction')
    events = events.merge(data[['product_id', 'listing_date', 'target_window_end']], on='product_id', validate='many_to_one')
    if (events.purchased_at < events.listing_date).any():
        raise ValueError('Purchase predates listing; check catalog coverage')
    events = events.loc[(events.purchased_at >= events.listing_date) & (events.purchased_at < events.target_window_end)]
    revenue = events.groupby('product_id').revenue.sum()
    data['launch_sales_brl'] = data.product_id.map(revenue).fillna(0.)
    data['forecast_origin'] = data.listing_date
    return data.sort_values(['forecast_origin', 'product_id']).reset_index(drop=True)
