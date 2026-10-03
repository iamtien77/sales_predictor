"""
feature_engineering.py - Tạo và làm giàu các đặc trưng (Feature Engineering)
=============================================================================
1. Product Features:
   - add_product_popularity(df): rating, num_reviews, popularity_score
   - add_description_features(df): description_length, description_word_count
   - add_price_competitiveness(df): price_relative_to_category_avg

2. Customer Demographics & Lifetime Value:
   - add_customer_segments(df): age_group
   - add_customer_lifetime_value(df): CLV, avg_order_value, recency_days
   - add_purchase_frequency(df): purchase_frequency, previous_purchases

3. Marketing Campaign Data:
   - add_marketing_features(df): ad_spend, campaigns_run, campaign_channel
   - add_pricing_promotions(df): discount_rate, discounted_price, ad_spend_to_price_ratio

4. Seasonal & Holiday Information:
   - add_seasonal_trends(df): month, quarter, day_of_week, is_weekend, season
   - add_holiday_features(df): is_holiday, is_peak_shopping_season
   - add_temporal_features(df): sales_lag_1, sales_lag_7, sales_rolling_mean_7
"""
