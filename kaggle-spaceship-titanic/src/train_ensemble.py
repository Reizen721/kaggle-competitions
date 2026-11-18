"""
Spaceship Titanic - 5-Fold Stratified Multi-Model GBDT Ensemble
Models: CatBoost + XGBoost + HistGradientBoosting + RandomForest
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import process_features

WS_DIR = r"C:\Users\Devil__Zero\.gemini\antigravity\scratch\kaggle-spaceship-titanic"
DATA_DIR = os.path.join(WS_DIR, "data")
SUB_DIR = os.path.join(WS_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
    print("=== Loading Spaceship Titanic Data ===")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    X_train, X_test, y = process_features(train_df, test_df)
    print(f"Engineered Feature count: {X_train.shape[1]}")
    
    cat_cols = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']
    
    # Label encode categorical columns for XGB and Sklearn
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
        
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_cat = np.zeros(len(y))
    test_cat = np.zeros(len(X_test))
    
    oof_xgb = np.zeros(len(y))
    test_xgb = np.zeros(len(X_test))
    
    oof_hgb = np.zeros(len(y))
    test_hgb = np.zeros(len(X_test))
    
    oof_rf = np.zeros(len(y))
    test_rf = np.zeros(len(X_test))
    
    print("\n=== Running 5-Fold Stratified Cross Validation ===")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr_cat, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va_cat, y_va = X_train.iloc[val_idx], y[val_idx]
        
        X_tr_enc, X_va_enc = X_train_encoded.iloc[tr_idx], X_train_encoded.iloc[val_idx]
        
        # 1. CatBoost
        train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va_cat, y_va, cat_features=cat_cols)
        test_pool = Pool(X_test, cat_features=cat_cols)
        
        cb_model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            eval_metric='Accuracy',
            random_seed=42 + fold,
            verbose=0
        )
        cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        p_cat_val = cb_model.predict_proba(val_pool)[:, 1]
        oof_cat[val_idx] = p_cat_val
        test_cat += cb_model.predict_proba(test_pool)[:, 1] / n_splits
        
        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42 + fold,
            n_jobs=-1
        )
        xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)
        p_xgb_val = xgb_model.predict_proba(X_va_enc)[:, 1]
        oof_xgb[val_idx] = p_xgb_val
        test_xgb += xgb_model.predict_proba(X_test_encoded)[:, 1] / n_splits
        
        # 3. HistGradientBoostingClassifier
        cat_indices = [X_train_encoded.columns.get_loc(c) for c in cat_cols]
        hgb_model = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            categorical_features=cat_indices,
            random_state=42 + fold
        )
        hgb_model.fit(X_tr_enc, y_tr)
        p_hgb_val = hgb_model.predict_proba(X_va_enc)[:, 1]
        oof_hgb[val_idx] = p_hgb_val
        test_hgb += hgb_model.predict_proba(X_test_encoded)[:, 1] / n_splits
        
        # 4. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42 + fold,
            n_jobs=-1
        )
        rf_model.fit(X_tr_enc, y_tr)
        p_rf_val = rf_model.predict_proba(X_va_enc)[:, 1]
        oof_rf[val_idx] = p_rf_val
        test_rf += rf_model.predict_proba(X_test_encoded)[:, 1] / n_splits
        
        print(f"Fold {fold+1}: CatBoost Acc = {accuracy_score(y_va, (p_cat_val>=0.5).astype(int)):.4f} | "
              f"XGB Acc = {accuracy_score(y_va, (p_xgb_val>=0.5).astype(int)):.4f} | "
              f"HGB Acc = {accuracy_score(y_va, (p_hgb_val>=0.5).astype(int)):.4f} | "
              f"RF Acc = {accuracy_score(y_va, (p_rf_val>=0.5).astype(int)):.4f}")
        
    print("\n=== Out-Of-Fold Accuracy Comparison ===")
    print(f"CatBoost OOF Accuracy: {accuracy_score(y, (oof_cat>=0.5).astype(int)):.4f}")
    print(f"XGBoost OOF Accuracy:  {accuracy_score(y, (oof_xgb>=0.5).astype(int)):.4f}")
    print(f"HistGBM OOF Accuracy:  {accuracy_score(y, (oof_hgb>=0.5).astype(int)):.4f}")
    print(f"RandomF OOF Accuracy:  {accuracy_score(y, (oof_rf>=0.5).astype(int)):.4f}")
    
    # Ensemble Optimization
    print("\n=== Searching Optimal Ensemble Weights ===")
    best_weights = None
    best_acc = 0.0
    best_th = 0.5
    
    weights_candidates = [
        (0.40, 0.30, 0.20, 0.10),
        (0.45, 0.25, 0.20, 0.10),
        (0.35, 0.35, 0.20, 0.10),
        (0.50, 0.25, 0.15, 0.10),
        (0.40, 0.20, 0.30, 0.10),
        (0.50, 0.50, 0.00, 0.00),
    ]
    
    for w in weights_candidates:
        blend_oof = w[0]*oof_cat + w[1]*oof_xgb + w[2]*oof_hgb + w[3]*oof_rf
        for th in np.arange(0.45, 0.55, 0.005):
            acc = accuracy_score(y, (blend_oof >= th).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_weights = w
                best_th = th
                
    print(f"Optimal Weights (CatBoost, XGB, HistGB, RF): {best_weights}")
    print(f"Optimal Threshold: {best_th:.3f}")
    print(f"Ensemble OOF Accuracy: {best_acc:.4f}")
    
    # Test Predictions
    test_blend_prob = (best_weights[0]*test_cat + 
                       best_weights[1]*test_xgb + 
                       best_weights[2]*test_hgb + 
                       best_weights[3]*test_rf)
    
    test_preds = (test_blend_prob >= best_th).astype(bool)
    
    sub = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': test_preds
    })
    sub_path = os.path.join(SUB_DIR, "submission_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nGenerated submission saved to: {sub_path}")
    print(f"Submission distribution:\n{sub['Transported'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
