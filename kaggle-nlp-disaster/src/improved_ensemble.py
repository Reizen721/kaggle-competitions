"""
Improved Local Multi-Model Ensemble with Label Denoising and Exact Match Post-Processing
Models: Logistic Regression + Calibrated ComplementNB + SGDClassifier + LightGBM
"""

import os
import re
import html
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse import hstack, csr_matrix
from scipy.special import expit
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
    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    # Replace user mentions
    text = re.sub(r"@\w+", " [USER] ", text)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def denoise_train_labels(df):
    """Resolve conflicting labels for identical texts by majority vote."""
    df = df.copy()
    text_stats = df.groupby('text')['target'].agg(['count', 'mean'])
    conflicts = text_stats[(text_stats['count'] > 1) & (text_stats['mean'] > 0) & (text_stats['mean'] < 1)]
    resolved_count = 0
    for text, row in conflicts.iterrows():
        majority = 1 if row['mean'] >= 0.5 else 0
        df.loc[df['text'] == text, 'target'] = majority
        resolved_count += 1
    print(f"Denoised {resolved_count} conflicting text groups using majority voting.")
    return df

def extract_meta_features(df):
    feats = pd.DataFrame(index=df.index)
    text_series = df['text'].fillna("")
    
    feats['char_len'] = text_series.apply(len)
    feats['word_count'] = text_series.apply(lambda x: len(x.split()))
    feats['mean_word_len'] = feats['char_len'] / (feats['word_count'] + 1e-5)
    feats['caps_count'] = text_series.apply(lambda x: sum(1 for c in x if c.isupper()))
    feats['caps_ratio'] = feats['caps_count'] / (feats['char_len'] + 1e-5)
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
    for thresh in np.arange(0.30, 0.70, 0.005):
        score = f1_score(y_true, (y_probs >= thresh).astype(int))
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

def main():
    print("=== Loading & Cleaning Data ===")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    train = denoise_train_labels(train_raw)
    
    # Text cleaning
    train['clean_text'] = train['text'].apply(clean_text)
    test['clean_text'] = test['text'].apply(clean_text)
    
    # Incorporate keyword into text representation
    train_input = train.apply(lambda r: f"keyword {r['keyword']}: {r['clean_text']}" if pd.notnull(r['keyword']) else r['clean_text'], axis=1)
    test_input = test.apply(lambda r: f"keyword {r['keyword']}: {r['clean_text']}" if pd.notnull(r['keyword']) else r['clean_text'], axis=1)
    
    y = train['target'].values
    
    print("=== Extracting Multi-Granularity TF-IDF ===")
    word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=25000, sublinear_tf=True, min_df=2)
    char_vec = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', max_features=35000, sublinear_tf=True, min_df=2)
    
    X_word_tr = word_vec.fit_transform(train_input)
    X_word_te = word_vec.transform(test_input)
    
    X_char_tr = char_vec.fit_transform(train_input)
    X_char_te = char_vec.transform(test_input)
    
    meta_tr = extract_meta_features(train)
    meta_te = extract_meta_features(test)
    
    X_train = hstack([X_word_tr, X_char_tr, csr_matrix(meta_tr)]).tocsr()
    X_test = hstack([X_word_te, X_char_te, csr_matrix(meta_te)]).tocsr()
    print(f"Feature Dimension: {X_train.shape[1]}")
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_lr = np.zeros(len(train))
    test_lr = np.zeros(len(test))
    
    oof_nb = np.zeros(len(train))
    test_nb = np.zeros(len(test))
    
    oof_sgd = np.zeros(len(train))
    test_sgd = np.zeros(len(test))
    
    oof_lgb = np.zeros(len(train))
    test_lgb = np.zeros(len(test))
    
    print("\n=== Training 5-Fold Diverse Models ===")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[val_idx], y[val_idx]
        
        # 1. Logistic Regression
        clf_lr = LogisticRegression(C=2.0, max_iter=1000, random_state=42, solver='saga')
        clf_lr.fit(X_tr, y_tr)
        p_lr = clf_lr.predict_proba(X_va)[:, 1]
        oof_lr[val_idx] = p_lr
        test_lr += clf_lr.predict_proba(X_test)[:, 1] / n_splits
        
        # 2. Complement Naive Bayes
        X_tr_text = hstack([X_word_tr[tr_idx], X_char_tr[tr_idx]]).tocsr()
        X_va_text = hstack([X_word_tr[val_idx], X_char_tr[val_idx]]).tocsr()
        X_te_text = hstack([X_word_te, X_char_te]).tocsr()
        
        cnb = ComplementNB(alpha=0.6, norm=True)
        cnb.fit(X_tr_text, y_tr)
        p_nb = cnb.predict_proba(X_va_text)[:, 1]
        oof_nb[val_idx] = p_nb
        test_nb += cnb.predict_proba(X_te_text)[:, 1] / n_splits
        
        # 3. SGDClassifier with modified Huber loss
        clf_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=1e-5, l1_ratio=0.15, max_iter=1000, random_state=42)
        clf_sgd.fit(X_tr, y_tr)
        p_sgd = clf_sgd.predict_proba(X_va)[:, 1]
        oof_sgd[val_idx] = p_sgd
        test_sgd += clf_sgd.predict_proba(X_test)[:, 1] / n_splits
        
        # 4. Ridge Classifier (Sigmoid calibrated)
        clf_ridge = RidgeClassifier(alpha=1.5, random_state=42)
        clf_ridge.fit(X_tr, y_tr)
        p_ridge = expit(clf_ridge.decision_function(X_va))
        oof_lgb[val_idx] = p_ridge
        test_lgb += expit(clf_ridge.decision_function(X_test)) / n_splits
        
        print(f"Fold {fold+1}: LR F1={f1_score(y_va, (p_lr>=0.5).astype(int)):.4f} | "
              f"NB F1={f1_score(y_va, (p_nb>=0.5).astype(int)):.4f} | "
              f"SGD F1={f1_score(y_va, (p_sgd>=0.5).astype(int)):.4f} | "
              f"Ridge F1={f1_score(y_va, (p_ridge>=0.5).astype(int)):.4f}")
        
    print("\n=== Model Comparison on OOF ===")
    for name, probs in [("Logistic Regression", oof_lr), ("Complement NB", oof_nb), 
                        ("SGDClassifier", oof_sgd), ("Ridge Classifier", oof_lgb)]:
        th, best_f1 = find_best_threshold(y, probs)
        print(f"{name:20s}: Default (0.50) F1 = {f1_score(y, (probs>=0.5).astype(int)):.4f} | Optimal ({th:.3f}) F1 = {best_f1:.4f}")
        
    # Multi-model Ensemble Optimization (Grid search weights)
    print("\n=== Grid Searching Optimal Ensemble Weights ===")
    best_weights = None
    best_ensemble_f1 = 0.0
    best_th = 0.5
    
    # Candidate weights
    weights_list = [
        (0.40, 0.20, 0.15, 0.25),
        (0.35, 0.25, 0.15, 0.25),
        (0.45, 0.25, 0.10, 0.20),
        (0.50, 0.20, 0.10, 0.20),
        (0.30, 0.30, 0.20, 0.20),
    ]
    
    for w in weights_list:
        blend_oof = w[0]*oof_lr + w[1]*oof_nb + w[2]*oof_sgd + w[3]*oof_lgb
        th, f1 = find_best_threshold(y, blend_oof)
        if f1 > best_ensemble_f1:
            best_ensemble_f1 = f1
            best_weights = w
            best_th = th
            
    print(f"Optimal Weights (LR, NB, SGD, LGB): {best_weights}")
    print(f"Ensemble Optimal Threshold: {best_th:.3f}")
    print(f"Ensemble Best OOF F1: {best_ensemble_f1:.4f}")
    
    # Calculate Test Predictions with Optimal Blend
    test_final_prob = (best_weights[0]*test_lr + 
                       best_weights[1]*test_nb + 
                       best_weights[2]*test_sgd + 
                       best_weights[3]*test_lgb)
    
    test_preds = (test_final_prob >= best_th).astype(int)
    
    # Exact Match Post-Processing
    # For test texts that exactly exist in train, override with the train ground truth!
    exact_matches = 0
    train_text_dict = train.groupby('text')['target'].first().to_dict()
    for idx, row in test.iterrows():
        t = row['text']
        if t in train_text_dict:
            exact_matches += 1
            test_preds[idx] = train_text_dict[t]
            
    print(f"\nApplied Exact Match Post-Processing to {exact_matches} test samples.")
    
    sub = pd.DataFrame({
        'id': test['id'],
        'target': test_preds
    })
    sub_path = os.path.join(SUB_DIR, "submission_improved_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved improved ensemble submission to: {sub_path}")
    print(f"Test Positive Ratio: {sub['target'].mean():.4f}")

if __name__ == "__main__":
    main()
