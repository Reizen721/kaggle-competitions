"""
Spaceship Titanic - Advanced Multi-Model GBDT Ensemble v2
Models: CatBoost + XGBoost + HistGradientBoosting with Tuned Hyperparameters
Includes: Interaction Features, Stratified 5-Fold CV, Group Consensus Post-Processing
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

from feature_engineering import process_features_v2

WS_DIR = r"C:\Users\Devil__Zero\.gemini\antigravity\scratch\kaggle-spaceship-titanic"
DATA_DIR = os.path.join(WS_DIR, "data")
SUB_DIR = os.path.join(WS_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
    print("=== Loading Spaceship Titanic Data ===")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    X_train, X_test, y = process_features_v2(train_df, test_df)
    print(f"Features: {X_train.shape[1]}")
    
    cat_cols = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side', 
                'Deck_Side', 'Planet_Destination', 'Deck_Planet']
    
    # Label encode for tree models
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_enc[col] = le.transform(X_train[col].astype(str))
        X_test_enc[col] = le.transform(X_test[col].astype(str))
        
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_cat = np.zeros(len(y))
    test_cat = np.zeros(len(X_test))
    
    oof_xgb = np.zeros(len(y))
    test_xgb = np.zeros(len(X_test))
    
    oof_hgb = np.zeros(len(y))
    test_hgb = np.zeros(len(X_test))
    
    print("\n=== Training 5-Fold CatBoost + XGBoost + HistGBM ===")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr_cat, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va_cat, y_va = X_train.iloc[val_idx], y[val_idx]
        
        X_tr_enc, X_va_enc = X_train_enc.iloc[tr_idx], X_train_enc.iloc[val_idx]
        
        # 1. CatBoost with Categoricals
        cb_train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_cols)
        cb_val_pool = Pool(X_va_cat, y_va, cat_features=cat_cols)
        cb_test_pool = Pool(X_test, cat_features=cat_cols)
        
        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=4,
            eval_metric='Accuracy',
            random_seed=42 + fold,
            verbose=0
        )
        cb.fit(cb_train_pool, eval_set=cb_val_pool, early_stopping_rounds=120)
        p_cat = cb.predict_proba(cb_val_pool)[:, 1]
        oof_cat[val_idx] = p_cat
        test_cat += cb.predict_proba(cb_test_pool)[:, 1] / n_splits
        
        # 2. XGBoost
        xgb_m = xgb.XGBClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=3,
            eval_metric='logloss',
            random_state=42 + fold,
            n_jobs=-1
        )
        xgb_m.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)
        p_xgb = xgb_m.predict_proba(X_va_enc)[:, 1]
        oof_xgb[val_idx] = p_xgb
        test_xgb += xgb_m.predict_proba(X_test_enc)[:, 1] / n_splits
        
        # 3. HistGradientBoostingClassifier
        cat_indices = [X_train_enc.columns.get_loc(c) for c in cat_cols]
        hgb = HistGradientBoostingClassifier(
            max_iter=700,
            learning_rate=0.025,
            max_leaf_nodes=35,
            min_samples_leaf=25,
            l2_regularization=1.5,
            categorical_features=cat_indices,
            random_state=42 + fold
        )
        hgb.fit(X_tr_enc, y_tr)
        p_hgb = hgb.predict_proba(X_va_enc)[:, 1]
        oof_hgb[val_idx] = p_hgb
        test_hgb += hgb.predict_proba(X_test_enc)[:, 1] / n_splits
        
        print(f"Fold {fold+1}: CatBoost = {accuracy_score(y_va, (p_cat>=0.5).astype(int)):.4f} | "
              f"XGB = {accuracy_score(y_va, (p_xgb>=0.5).astype(int)):.4f} | "
              f"HGB = {accuracy_score(y_va, (p_hgb>=0.5).astype(int)):.4f}")
        
    print("\n=== Single Model 5-Fold OOF Accuracy ===")
    print(f"CatBoost OOF Accuracy: {accuracy_score(y, (oof_cat>=0.5).astype(int)):.4f}")
    print(f"XGBoost  OOF Accuracy: {accuracy_score(y, (oof_xgb>=0.5).astype(int)):.4f}")
    print(f"HistGBM  OOF Accuracy: {accuracy_score(y, (oof_hgb>=0.5).astype(int)):.4f}")
    
    # Ensemble Blending
    blend_weights = (0.50, 0.25, 0.25)
    oof_blend = blend_weights[0]*oof_cat + blend_weights[1]*oof_xgb + blend_weights[2]*oof_hgb
    test_blend = blend_weights[0]*test_cat + blend_weights[1]*test_xgb + blend_weights[2]*test_hgb
    
    # Search threshold near 0.50
    best_th = 0.50
    best_acc = 0.0
    for th in np.arange(0.48, 0.525, 0.002):
        acc = accuracy_score(y, (oof_blend >= th).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    print(f"\nEnsemble OOF Accuracy: {best_acc:.4f} (Threshold = {best_th:.3f})")
    print(f"Ensemble OOF Accuracy at 0.50: {accuracy_score(y, (oof_blend >= 0.5).astype(int)):.4f}")
    
    # 4. Group Consensus Post-Processing on Test
    test_df['pred_prob'] = test_blend
    test_df['Group_Id'] = test_df['PassengerId'].apply(lambda x: x.split('_')[0])
    
    # If a group has multiple members and strong consensus, slightly calibrate individual uncertainty
    group_stats = test_df.groupby('Group_Id')['pred_prob'].agg(['count', 'mean'])
    confident_true_groups = group_stats[(group_stats['count'] > 1) & (group_stats['mean'] >= 0.82)].index
    confident_false_groups = group_stats[(group_stats['count'] > 1) & (group_stats['mean'] <= 0.18)].index
    
    final_test_preds = (test_blend >= best_th).astype(bool)
    
    adjusted_count = 0
    for idx, row in test_df.iterrows():
        gid = row['Group_Id']
        if gid in confident_true_groups and not final_test_preds[idx]:
            final_test_preds[idx] = True
            adjusted_count += 1
        elif gid in confident_false_groups and final_test_preds[idx]:
            final_test_preds[idx] = False
            adjusted_count += 1
            
    print(f"Applied Group Consensus adjustment to {adjusted_count} boundary samples.")
    
    sub = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_test_preds
    })
    sub_path = os.path.join(SUB_DIR, "submission_advanced_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to: {sub_path}")
    print(f"Test Positive Ratio: {sub['Transported'].mean():.4f}")

if __name__ == "__main__":
    main()
