"""
Store Sales - Time Series Forecasting
Ultra-Lightweight HistGradientBoosting Ensemble Forecaster
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import gc
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import build_features

WS_DIR = r"C:\Users\Devil__Zero\.gemini\antigravity\scratch\kaggle-store-sales"
DATA_DIR = os.path.join(WS_DIR, "data")
SUB_DIR = os.path.join(WS_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
    print("=== Building Engineered Features ===")
    train_data, test_data = build_features(DATA_DIR, start_date='2017-01-01')
    
    features = [
        'store_nbr', 'family', 'type', 'cluster', 'city', 'state',
        'onpromotion', 'promo_roll_7', 'promo_lag_16',
        'dcoilwtico', 'oil_rolling_7', 'is_holiday',
        'dayofweek', 'day', 'month', 'is_weekend', 'is_payday',
        'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20', 'lag_21', 'lag_28',
        'roll_mean_7', 'roll_mean_14', 'roll_mean_28', 'roll_std_7'
    ]
    cat_cols = ['store_nbr', 'family', 'type', 'cluster', 'city', 'state']
    
    # Encode categorical features as integer codes
    for c in cat_cols:
        train_data[c] = train_data[c].astype('category').cat.codes.astype(np.int16)
        test_data[c] = test_data[c].astype('category').cat.codes.astype(np.int16)
        
    cat_indices = [features.index(c) for c in cat_cols]
    print(f"Features ({len(features)}): {features}")
    print(f"Categorical indices: {cat_indices}")
    
    # 1. 16-Day Time-Series Holdout Validation (2017-07-31 to 2017-08-15)
    val_mask = train_data['date'] >= '2017-07-31'
    X_train_sub = train_data[~val_mask][features].copy()
    y_train_sub = train_data[~val_mask]['log_sales'].values
    
    X_val = train_data[val_mask][features].copy()
    y_val = train_data[val_mask]['log_sales'].values
    val_true_sales = train_data[val_mask]['sales'].values
    
    print(f"\n=== Training 16-Day Holdout Validation Set ===")
    print(f"Train samples: {len(X_train_sub)}, Validation samples: {len(X_val)}")
    
    # Model 1: HistGBM Standard (max_leaf_nodes=45, lr=0.06)
    print("Training HistGBM Model 1...")
    hgb1 = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=45,
        min_samples_leaf=25,
        l2_regularization=0.5,
        categorical_features=cat_indices,
        random_state=42
    )
    hgb1.fit(X_train_sub, y_train_sub)
    val_preds_1 = np.expm1(np.maximum(0, hgb1.predict(X_val)))
    rmsle_1 = root_mean_squared_error(np.log1p(val_true_sales), np.log1p(val_preds_1))
    print(f"HistGBM Model 1 Holdout RMSLE: {rmsle_1:.4f}")
    
    # Model 2: HistGBM Deeper (max_leaf_nodes=65, min_samples_leaf=40, lr=0.04, max_iter=500)
    print("Training HistGBM Model 2...")
    hgb2 = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=65,
        min_samples_leaf=40,
        l2_regularization=1.5,
        categorical_features=cat_indices,
        random_state=2024
    )
    hgb2.fit(X_train_sub, y_train_sub)
    val_preds_2 = np.expm1(np.maximum(0, hgb2.predict(X_val)))
    rmsle_2 = root_mean_squared_error(np.log1p(val_true_sales), np.log1p(val_preds_2))
    print(f"HistGBM Model 2 Holdout RMSLE: {rmsle_2:.4f}")
    
    # Blend Holdout
    val_blend = 0.5 * val_preds_1 + 0.5 * val_preds_2
    rmsle_blend = root_mean_squared_error(np.log1p(val_true_sales), np.log1p(val_blend))
    print(f"Ensemble (HGB1 + HGB2) Holdout RMSLE: {rmsle_blend:.4f}")
    
    del hgb1, hgb2, X_train_sub, X_val
    gc.collect()
    
    # 2. Retrain on Full 2017 Dataset & Predict Test
    print("\n=== Retraining on Full 2017 Dataset for Final Forecast ===")
    X_full = train_data[features].copy()
    y_full = train_data['log_sales'].values
    X_test = test_data[features].copy()
    
    print("Retraining HGB1 on full data...")
    hgb1_full = HistGradientBoostingRegressor(
        max_iter=450,
        learning_rate=0.06,
        max_leaf_nodes=45,
        min_samples_leaf=25,
        l2_regularization=0.5,
        categorical_features=cat_indices,
        random_state=42
    )
    hgb1_full.fit(X_full, y_full)
    test_preds_1 = np.expm1(np.maximum(0, hgb1_full.predict(X_test)))
    
    print("Retraining HGB2 on full data...")
    hgb2_full = HistGradientBoostingRegressor(
        max_iter=550,
        learning_rate=0.04,
        max_leaf_nodes=65,
        min_samples_leaf=40,
        l2_regularization=1.5,
        categorical_features=cat_indices,
        random_state=2024
    )
    hgb2_full.fit(X_full, y_full)
    test_preds_2 = np.expm1(np.maximum(0, hgb2_full.predict(X_test)))
    
    final_test_sales = 0.5 * test_preds_1 + 0.5 * test_preds_2
    
    # 3. Domain Zero-Sales Constraint & Perfect ID-Alignment
    raw_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=['store_nbr', 'family', 'sales'])
    store_fam_totals = raw_train.groupby(['store_nbr', 'family'])['sales'].sum()
    zero_series = set(store_fam_totals[store_fam_totals == 0].index)
    del raw_train
    gc.collect()
    
    # Align predictions with true IDs from test_data
    test_preds_df = pd.DataFrame({
        'id': test_data['id'].values.astype(int),
        'sales': np.maximum(0, final_test_sales)
    })
    
    # Merge back onto raw test.csv to guarantee exact sample_submission ID ordering
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    merged_sub = test_raw.merge(test_preds_df, on='id', how='left')
    
    zero_mask = merged_sub.set_index(['store_nbr', 'family']).index.isin(zero_series)
    merged_sub.loc[zero_mask, 'sales'] = 0.0
    print(f"Applied Zero-Sales Domain Constraint to {zero_mask.sum()} items.")
    
    # Save Final Submission
    sub = merged_sub[['id', 'sales']].copy()
    sub_path = os.path.join(SUB_DIR, "submission_store_sales.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved forecast submission to: {sub_path}")
    print(f"Forecast sales summary:\n{sub['sales'].describe()}")

if __name__ == "__main__":
    main()
