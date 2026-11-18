"""
Disaster Tweets - Feature Engineering and 5-Fold Stratified CV Baseline
Models: TF-IDF (Word + Char) + Statistical Features + LogisticRegression / Ridge / LightGBM
"""

import os
import re
import html
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

WS_DIR = r"C:\Users\Devil__Zero\.gemini\antigravity\scratch\kaggle-nlp-disaster"
DATA_DIR = os.path.join(WS_DIR, "data")
SUB_DIR = os.path.join(WS_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    # Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    # Remove redundant whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_meta_features(df):
    feats = pd.DataFrame(index=df.index)
    text_series = df['text'].fillna("")
    
    feats['char_length'] = text_series.apply(len)
    feats['word_count'] = text_series.apply(lambda x: len(x.split()))
    feats['mean_word_len'] = feats['char_length'] / (feats['word_count'] + 1e-5)
    feats['caps_count'] = text_series.apply(lambda x: sum(1 for c in x if c.isupper()))
    feats['caps_ratio'] = feats['caps_count'] / (feats['char_length'] + 1e-5)
    feats['excl_count'] = text_series.apply(lambda x: x.count('!'))
    feats['quest_count'] = text_series.apply(lambda x: x.count('?'))
    feats['punct_count'] = text_series.apply(lambda x: sum(1 for c in x if c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    feats['url_count'] = text_series.apply(lambda x: len(re.findall(r"https?://\S+|www\.\S+", x)))
    feats['mention_count'] = text_series.apply(lambda x: x.count('@'))
    feats['hashtag_count'] = text_series.apply(lambda x: x.count('#'))
    feats['digit_count'] = text_series.apply(lambda x: sum(1 for c in x if c.isdigit()))
    feats['has_keyword'] = df['keyword'].notnull().astype(int)
    
    # Scale features
    for col in feats.columns:
        mean, std = feats[col].mean(), feats[col].std()
        feats[col] = (feats[col] - mean) / (std + 1e-7)
    return feats.values

def find_best_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.30, 0.70, 0.01):
        score = f1_score(y_true, (y_probs >= thresh).astype(int))
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

def main():
    print("=== Loading Datasets ===")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    print(f"Train samples: {len(train)}, Test samples: {len(test)}")
    
    # Preprocessing
    train['clean_text'] = train['text'].apply(clean_text)
    test['clean_text'] = test['text'].apply(clean_text)
    
    # Enrich with keyword
    train_input = train.apply(lambda r: f"{r['keyword']} {r['clean_text']}" if pd.notnull(r['keyword']) else r['clean_text'], axis=1)
    test_input = test.apply(lambda r: f"{r['keyword']} {r['clean_text']}" if pd.notnull(r['keyword']) else r['clean_text'], axis=1)
    
    y = train['target'].values
    
    print("=== Extracting TF-IDF Features ===")
    word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', max_features=25000, sublinear_tf=True)
    
    X_word_train = word_vec.fit_transform(train_input)
    X_word_test = word_vec.transform(test_input)
    
    X_char_train = char_vec.fit_transform(train_input)
    X_char_test = char_vec.transform(test_input)
    
    print("=== Extracting Statistical Features ===")
    meta_train = extract_meta_features(train)
    meta_test = extract_meta_features(test)
    
    X_train = hstack([X_word_train, X_char_train, csr_matrix(meta_train)]).tocsr()
    X_test = hstack([X_word_test, X_char_test, csr_matrix(meta_test)]).tocsr()
    print(f"Total Feature Matrix Shape: {X_train.shape}")
    
    # 5-Fold Stratified Cross Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_lr = np.zeros(len(train))
    test_lr = np.zeros(len(test))
    
    oof_lgb = np.zeros(len(train))
    test_lgb = np.zeros(len(test))
    
    print("\n=== Running 5-Fold Stratified Cross Validation ===")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train[train_idx], y[train_idx]
        X_va, y_va = X_train[val_idx], y[val_idx]
        
        # 1. Logistic Regression with L2 Regularization (C=1.5)
        clf_lr = LogisticRegression(C=1.5, max_iter=1000, random_state=42, solver='saga')
        clf_lr.fit(X_tr, y_tr)
        val_probs_lr = clf_lr.predict_proba(X_va)[:, 1]
        oof_lr[val_idx] = val_probs_lr
        test_lr += clf_lr.predict_proba(X_test)[:, 1] / n_splits
        
        f1_fold_lr = f1_score(y_va, (val_probs_lr >= 0.5).astype(int))
        
        # 2. LightGBM on meta + top TF-IDF SVD or sparse
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_tr, y_tr)
        val_probs_lgb = lgb_model.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = val_probs_lgb
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        f1_fold_lgb = f1_score(y_va, (val_probs_lgb >= 0.5).astype(int))
        
        print(f"Fold {fold+1}: LR F1 = {f1_fold_lr:.4f} | LGB F1 = {f1_fold_lgb:.4f}")
        
    # Evaluate OOF
    print("\n=== OOF Evaluation Results ===")
    lr_th, lr_best_f1 = find_best_threshold(y, oof_lr)
    print(f"Logistic Regression Default (0.50) F1: {f1_score(y, (oof_lr >= 0.5).astype(int)):.4f}")
    print(f"Logistic Regression Optimal ({lr_th:.2f}) F1: {lr_best_f1:.4f}")
    
    lgb_th, lgb_best_f1 = find_best_threshold(y, oof_lgb)
    print(f"LightGBM Default (0.50) F1: {f1_score(y, (oof_lgb >= 0.5).astype(int)):.4f}")
    print(f"LightGBM Optimal ({lgb_th:.2f}) F1: {lgb_best_f1:.4f}")
    
    # Ensemble: Weighted average
    oof_blend = 0.7 * oof_lr + 0.3 * oof_lgb
    test_blend = 0.7 * test_lr + 0.3 * test_lgb
    blend_th, blend_best_f1 = find_best_threshold(y, oof_blend)
    print(f"Blend (0.7*LR + 0.3*LGB) Default (0.50) F1: {f1_score(y, (oof_blend >= 0.5).astype(int)):.4f}")
    print(f"Blend Optimal ({blend_th:.2f}) F1: {blend_best_f1:.4f}")
    
    # Generate Submissions
    sub = pd.DataFrame({
        'id': test['id'],
        'target': (test_blend >= blend_th).astype(int)
    })
    sub_path = os.path.join(SUB_DIR, "submission_baseline_blend.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nGenerated submission saved to: {sub_path}")
    print(f"Submission shape: {sub.shape}")
    print(f"Target distribution in test:\n{sub['target'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
