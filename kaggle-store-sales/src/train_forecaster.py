"""
Store Sales - Time Series Forecasting
Ultra-Memory-Efficient GBDT Forecaster & Holdout Validation
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
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import build_features

WS_DIR = r"C:\Users\Devil__Zero\.gemini\antigravity\scratch\kaggle-store-sales"
DATA_DIR = os.path.join(WS_DIR, "data")
SUB_DIR = os.path.join(WS_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
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
    
    # 1. 16-Day Time-Series Holdout Validation (2017-07-31 to 2017-08-15)
    val_mask = train_data['date'] >= '2017-07-31'
    X_train_sub = train_data[~val_mask][features].copy()
    y_train_sub = train_data[~val_mask]['log_sales'].values
    
    X_val = train_data[val_mask][features].copy()
    y_val = train_data[val_mask]['log_sales'].values
    val_true_sales = train_data[val_mask]['sales'].values
    
    print(f"\n=== Training 16-Day Holdout Validation Set ===")
    print(f"Train samples: {len(X_train_sub)}, Validation samples: {len(X_val)}")
    
    # Model 1: CatBoost on Holdout
    cb_val = CatBoostRegressor(
        iterations=900,
        learning_rate=0.07,
        depth=6,
        l2_leaf_reg=4,
        loss_function='RMSE',
        cat_features=cat_cols,
        random_seed=42,
        verbose=150
    )
    cb_val.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val), early_stopping_rounds=60)
    val_preds_cb = np.expm1(np.maximum(0, cb_val.predict(X_val)))
    rmsle_cb = root_mean_squared_error(np.log1p(val_true_sales), np.log1p(val_preds_cb))
    print(f"CatBoost 16-Day Holdout RMSLE: {rmsle_cb:.4f}")
    
    # Model 2: HistGBM on Holdout
    X_train_sub_enc = X_train_sub.copy()
    X_val_enc = X_val.copy()
    cat_indices = []
    for c in cat_cols:
        cat_indices.append(X_train_sub_enc.columns.get_loc(c))
        X_train_sub_enc[c] = X_train_sub_enc[c].astype('category').cat.codes
        X_val_enc[c] = X_val_enc[c].astype('category').cat.codes
        
    hgb_val = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=40,
        min_samples_leaf=30,
        categorical_features=cat_indices,
        random_state=42
    )
    hgb_val.fit(X_train_sub_enc, y_train_sub)
    val_preds_hgb = np.expm1(np.maximum(0, hgb_val.predict(X_val_enc)))
    rmsle_hgb = root_mean_squared_error(np.log1p(val_true_sales), np.log1p(val_preds_hgb))
    print(f"HistGBM 16-Day Holdout RMSLE:  {rmsle_hgb:.4f}")
    
    # Ensemble Validation
    val_blend = 0.65 * val_preds_cb + 0.35 * val_preds_hgb
    rmsle_blend = root_mean_squared_error(np.log1p(val_true_sales), np.log1p(val_blend))
    print(f"Ensemble (65% CB + 35% HGB) Holdout RMSLE: {rmsle_blend:.4f}")
    
    del cb_val, hgb_val, X_train_sub, X_val, X_train_sub_enc, X_val_enc
    gc.collect()
    
    # 2. Retrain on Full 2017 Dataset & Predict Test
    print("\n=== Retraining on Full 2017 Dataset for Final Forecast ===")
    X_full = train_data[features].copy()
    y_full = train_data['log_sales'].values
    X_test = test_data[features].copy()
    
    # CatBoost Full
    cb_full = CatBoostRegressor(
        iterations=950,
        learning_rate=0.07,
        depth=6,
        l2_leaf_reg=4,
        loss_function='RMSE',
        cat_features=cat_cols,
        random_seed=42,
        verbose=200
    )
    cb_full.fit(X_full, y_full)
    test_cb_sales = np.expm1(np.maximum(0, cb_full.predict(X_test)))
    
    # HistGBM Full
    X_full_enc = X_full.copy()
    X_test_enc = X_test.copy()
    for c in cat_cols:
        X_full_enc[c] = X_full_enc[c].astype('category').cat.codes
        X_test_enc[c] = X_test_enc[c].astype('category').cat.codes
        
    hgb_full = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=40,
        min_samples_leaf=30,
        categorical_features=cat_indices,
        random_state=42
    )
    hgb_full.fit(X_full_enc, y_full)
    test_hgb_sales = np.expm1(np.maximum(0, hgb_full.predict(X_test_enc)))
    
    # Blend Forecast
    final_test_sales = 0.65 * test_cb_sales + 0.35 * test_hgb_sales
    
    # 3. Domain Zero-Sales Filter
    raw_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=['store_nbr', 'family', 'sales'])
    store_fam_totals = raw_train.groupby(['store_nbr', 'family'])['sales'].sum()
    zero_series = set(store_fam_totals[store_fam_totals == 0].index)
    del raw_train
    gc.collect()
    
    test_reset = test_data.reset_index(drop=True)
    zero_mask = test_reset.set_index(['store_nbr', 'family']).index.isin(zero_series)
    final_test_sales[zero_mask] = 0.0
    print(f"Applied Zero-Sales Domain Constraint to {zero_mask.sum()} forecast items.")
    
    # Save Submission
    sub = pd.DataFrame({
        'id': test_reset['id'].astype(int),
        'sales': final_test_sales
    })
    sub_path = os.path.join(SUB_DIR, "submission_store_sales.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved forecast submission to: {sub_path}")
    print(f"Forecast sales summary:\n{sub['sales'].describe()}")

if __name__ == "__main__":
    main()
