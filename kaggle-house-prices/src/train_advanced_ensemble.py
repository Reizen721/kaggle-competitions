"""
House Prices: Advanced Regression Techniques
10-Fold Cross-Validation Multi-Model Regression Ensemble v2
Includes: Non-Linear Interactions, Neighborhood Tiers, and Constrained Optimal Blending
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import prepare_data_v2

WS_DIR = r"C:\Users\Devil__Zero\.gemini\antigravity\scratch\kaggle-house-prices"
DATA_DIR = os.path.join(WS_DIR, "data")
SUB_DIR = os.path.join(WS_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
    print("=== Loading and Preparing Data (v2) ===")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    X_train, X_test, y_train, test_id = prepare_data_v2(train_df, test_df)
    
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    print(f"Final aligned feature count: {X_train.shape[1]}")
    
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_ridge = np.zeros(len(y_train))
    test_ridge = np.zeros(len(X_test))
    
    oof_lasso = np.zeros(len(y_train))
    test_lasso = np.zeros(len(X_test))
    
    oof_enet = np.zeros(len(y_train))
    test_enet = np.zeros(len(X_test))
    
    oof_gbr = np.zeros(len(y_train))
    test_gbr = np.zeros(len(X_test))
    
    oof_xgb = np.zeros(len(y_train))
    test_xgb = np.zeros(len(X_test))
    
    oof_cat = np.zeros(len(y_train))
    test_cat = np.zeros(len(X_test))
    
    alphas_ridge = [0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30]
    alphas_lasso = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.001]
    
    print("\n=== Running 10-Fold Cross-Validation ===")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        
        # 1. Ridge
        ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=5))
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict(X_va)
        test_ridge += ridge.predict(X_test) / n_splits
        
        # 2. Lasso
        lasso = make_pipeline(RobustScaler(), LassoCV(alphas=alphas_lasso, max_iter=30000, cv=5, random_state=42))
        lasso.fit(X_tr, y_tr)
        oof_lasso[val_idx] = lasso.predict(X_va)
        test_lasso += lasso.predict(X_test) / n_splits
        
        # 3. ElasticNet
        enet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=alphas_lasso, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], max_iter=30000, cv=5, random_state=42))
        enet.fit(X_tr, y_tr)
        oof_enet[val_idx] = enet.predict(X_va)
        test_enet += enet.predict(X_test) / n_splits
        
        # 4. Gradient Boosting Regressor (Huber Loss)
        gbr = GradientBoostingRegressor(n_estimators=1600, learning_rate=0.018, max_depth=4, max_features='sqrt', loss='huber', subsample=0.8, random_state=42+fold)
        gbr.fit(X_tr, y_tr)
        oof_gbr[val_idx] = gbr.predict(X_va)
        test_gbr += gbr.predict(X_test) / n_splits
        
        # 5. XGBoost
        xm = xgb.XGBRegressor(n_estimators=1600, learning_rate=0.018, max_depth=4, subsample=0.75, colsample_bytree=0.7, gamma=0.01, random_state=42+fold, n_jobs=-1)
        xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xm.predict(X_va)
        test_xgb += xm.predict(X_test) / n_splits
        
        # 6. CatBoost
        cb = CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=5, l2_leaf_reg=4, loss_function='RMSE', random_seed=42+fold, verbose=0)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150)
        oof_cat[val_idx] = cb.predict(X_va)
        test_cat += cb.predict(X_test) / n_splits
        
        print(f"Fold {fold+1}: Ridge={root_mean_squared_error(y_va, oof_ridge[val_idx]):.4f} | "
              f"Lasso={root_mean_squared_error(y_va, oof_lasso[val_idx]):.4f} | "
              f"Cat={root_mean_squared_error(y_va, oof_cat[val_idx]):.4f} | "
              f"XGB={root_mean_squared_error(y_va, oof_xgb[val_idx]):.4f}")
        
    print("\n=== Single Model 10-Fold OOF RMSLE ===")
    models_oof = [oof_ridge, oof_lasso, oof_enet, oof_cat, oof_xgb, oof_gbr]
    models_test = [test_ridge, test_lasso, test_enet, test_cat, test_xgb, test_gbr]
    names = ["Ridge", "Lasso", "ElasticNet", "CatBoost", "XGBoost", "GradientBoost"]
    
    for name, oof in zip(names, models_oof):
        print(f"{name:15s} OOF RMSLE: {root_mean_squared_error(y_train, oof):.4f}")
        
    # Solve for optimal non-negative ensemble weights summing to 1
    print("\n=== Optimizing Stacked Weights ===")
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = sum(w[i] * models_oof[i] for i in range(len(w)))
        return root_mean_squared_error(y_train, pred)
        
    init_weights = [1.0 / len(models_oof)] * len(models_oof)
    bounds = [(0.0, 1.0)] * len(models_oof)
    res = minimize(loss_func, init_weights, bounds=bounds, method='SLSQP')
    
    opt_w = res.x / np.sum(res.x)
    print("Optimal Weights:")
    for name, w in zip(names, opt_w):
        print(f"  {name:15s}: {w:.4f}")
        
    opt_oof_pred = sum(opt_w[i] * models_oof[i] for i in range(len(opt_w)))
    opt_rmsle = root_mean_squared_error(y_train, opt_oof_pred)
    print("\n==============================================")
    print(f"Optimal Stacked Multi-Model OOF RMSLE: {opt_rmsle:.4f}")
    print("==============================================")
    
    # Test Predictions
    test_opt_blend = sum(opt_w[i] * models_test[i] for i in range(len(opt_w)))
    
    # Exponentiate back to original dollars
    final_sale_price = np.expm1(test_opt_blend)
    
    # Safety boundary clipping
    final_sale_price = np.clip(final_sale_price, 34000.0, 780000.0)
    
    sub = pd.DataFrame({
        'Id': test_id,
        'SalePrice': final_sale_price
    })
    sub_path = os.path.join(SUB_DIR, "submission_refined_stacked_v3.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved refined submission to: {sub_path}")
    print(f"Submission SalePrice summary:\n{sub['SalePrice'].describe()}")

if __name__ == "__main__":
    main()
