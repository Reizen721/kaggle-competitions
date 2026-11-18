"""
Store Sales - Time Series Forecasting
Ultra-Memory-Efficient Feature Engineering Pipeline (float32 / int16 downcasting)
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import gc
import numpy as np
import pandas as pd

def build_features(data_dir, start_date='2017-01-01'):
    print("=== Loading Raw Datasets (Filtered for 2017) ===")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=['date'])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=['date'])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=['date'])
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=['date'])
    
    # Filter 2017 data (include late 2016 for lag warm-up)
    train = train[train['date'] >= '2016-11-01'].copy()
    
    # Target log-transform in float32
    train['log_sales'] = np.log1p(train['sales'].astype(np.float32))
    test['log_sales'] = np.nan
    test['sales'] = np.nan
    
    # Combine train and test
    df = pd.concat([train, test], ignore_index=True)
    del train, test
    gc.collect()
    
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
    
    # Downcast IDs and base columns
    df['id'] = df['id'].astype(np.int32)
    df['store_nbr'] = df['store_nbr'].astype(np.int16)
    df['onpromotion'] = df['onpromotion'].fillna(0).astype(np.int16)
    
    print("=== Processing Oil Prices ===")
    full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    oil_full = pd.DataFrame({'date': full_dates})
    oil_full = oil_full.merge(oil, on='date', how='left')
    oil_full['dcoilwtico'] = oil_full['dcoilwtico'].interpolate(method='linear').ffill().bfill().astype(np.float32)
    oil_full['oil_rolling_7'] = oil_full['dcoilwtico'].rolling(7, min_periods=1).mean().astype(np.float32)
    
    df = df.merge(oil_full, on='date', how='left')
    del oil_full, oil
    gc.collect()
    
    print("=== Processing Holidays ===")
    holidays_clean = holidays[~holidays['transferred']].copy()
    nat_holidays = holidays_clean[holidays_clean['locale'] == 'National'][['date', 'type']].drop_duplicates(subset=['date'])
    nat_holidays = nat_holidays.rename(columns={'type': 'holiday_national'})
    df = df.merge(nat_holidays, on='date', how='left')
    
    df = df.merge(stores, on='store_nbr', how='left')
    
    reg_holidays = holidays_clean[holidays_clean['locale'] == 'Regional'][['date', 'locale_name', 'type']].drop_duplicates(subset=['date', 'locale_name'])
    reg_holidays = reg_holidays.rename(columns={'type': 'holiday_regional', 'locale_name': 'state'})
    df = df.merge(reg_holidays, on=['date', 'state'], how='left')
    
    loc_holidays = holidays_clean[holidays_clean['locale'] == 'Local'][['date', 'locale_name', 'type']].drop_duplicates(subset=['date', 'locale_name'])
    loc_holidays = loc_holidays.rename(columns={'type': 'holiday_local', 'locale_name': 'city'})
    df = df.merge(loc_holidays, on=['date', 'city'], how='left')
    
    df['is_holiday'] = (df['holiday_national'].notnull() | df['holiday_regional'].notnull() | df['holiday_local'].notnull()).astype(np.int8)
    df = df.drop(['holiday_national', 'holiday_regional', 'holiday_local'], axis=1)
    del holidays, holidays_clean, nat_holidays, reg_holidays, loc_holidays, stores
    gc.collect()
    
    print("=== Engineering Calendar Features ===")
    df['dayofweek'] = df['date'].dt.dayofweek.astype(np.int8)
    df['day'] = df['date'].dt.day.astype(np.int8)
    df['month'] = df['date'].dt.month.astype(np.int8)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
    
    days_in_m = df['date'].dt.days_in_month
    df['is_payday'] = ((df['day'] == 15) | (df['day'] == days_in_m)).astype(np.int8)
    del days_in_m
    
    print("=== Computing Lags (float32) ===")
    grp = df.groupby(['store_nbr', 'family'])['log_sales']
    
    for lag in [16, 17, 18, 19, 20, 21, 28]:
        df[f'lag_{lag}'] = grp.shift(lag).astype(np.float32)
        
    df['roll_mean_7'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype(np.float32)
    df['roll_mean_14'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14, min_periods=1).mean()).astype(np.float32)
    df['roll_mean_28'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(28, min_periods=1).mean()).astype(np.float32)
    df['roll_std_7'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0).astype(np.float32)
    
    df['promo_roll_7'] = df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype(np.float32)
    df['promo_lag_16'] = df.groupby(['store_nbr', 'family'])['onpromotion'].shift(16).fillna(0).astype(np.float32)
    
    # Categoricals
    cat_cols = ['family', 'city', 'state', 'type']
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    train_data = df[(df['date'] >= start_date) & (df['date'] <= '2017-08-15')].copy()
    test_data = df[df['date'] >= '2017-08-16'].copy()
    
    del df
    gc.collect()
    
    print(f"Prepared Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    return train_data, test_data
