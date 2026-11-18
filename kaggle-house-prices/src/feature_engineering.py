"""
House Prices: Advanced Regression Techniques
Enhanced Feature Engineering Pipeline v2
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from scipy.stats import skew

def prepare_data_v2(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    
    # 1. Remove extreme outliers as recommended by Dean De Cock
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
    train_len = len(train)
    
    # Log-transform target
    y_train = np.log1p(train['SalePrice'].values)
    test_id = test['Id']
    
    # Neighborhood Tier based on train median price
    neigh_median = train.groupby('Neighborhood')['SalePrice'].median()
    tier_map = {}
    for n, p in neigh_median.items():
        if p >= 280000:
            tier_map[n] = 5
        elif p >= 200000:
            tier_map[n] = 4
        elif p >= 160000:
            tier_map[n] = 3
        elif p >= 130000:
            tier_map[n] = 2
        elif p >= 100000:
            tier_map[n] = 1
        else:
            tier_map[n] = 0
            
    # Concatenate features
    features = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), test.drop(['Id'], axis=1)], ignore_index=True)
    
    # 2. Impute Categorical 'None' features
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        features[col] = features[col].fillna('None')
        
    # 3. Impute Numerical '0' features
    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        features[col] = features[col].fillna(0)
        
    # 4. Neighborhood-based LotFrontage imputation
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].median())
    
    # 5. Impute other features with contextual domain mode
    # Specific contextual test imputations
    features.loc[features['Neighborhood'] == 'IDOTRR', 'MSZoning'] = features.loc[features['Neighborhood'] == 'IDOTRR', 'MSZoning'].fillna('RM')
    features.loc[features['Neighborhood'] == 'Mitchel', 'MSZoning'] = features.loc[features['Neighborhood'] == 'Mitchel', 'MSZoning'].fillna('RL')
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'RL'))
    
    # GarageArea & Cars for Detached Garage (Id 2577)
    features.loc[(features['GarageType'] == 'Detchd') & (features['GarageArea'] == 0), 'GarageArea'] = 400.0
    features.loc[(features['GarageType'] == 'Detchd') & (features['GarageCars'] == 0), 'GarageCars'] = 2.0
    
    mode_cols = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities', 'Functional']
    for col in mode_cols:
        features[col] = features[col].fillna(features[col].mode()[0])
        
    # 6. Convert numerical codes that are actually categorical
    features['MSSubClass'] = features['MSSubClass'].astype(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    
    # 7. Map Ordinal Quality Features to Numeric Scales
    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in ordinal_cols:
        features[col] = features[col].map(qual_map).fillna(0)
        
    bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmt_fin_map).fillna(0)
    features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmt_fin_map).fillna(0)
    
    bsmt_exp_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    features['BsmtExposure'] = features['BsmtExposure'].map(bsmt_exp_map).fillna(0)
    
    # 8. Domain Specific Feature Engineering
    features['Neighborhood_Tier'] = features['Neighborhood'].map(tier_map).fillna(2)
    
    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
    features['TotalBath'] = features['FullBath'] + 0.5 * features['HalfBath'] + features['BsmtFullBath'] + 0.5 * features['BsmtHalfBath']
    features['TotalPorch'] = features['OpenPorchSF'] + features['EnclosedPorch'] + features['3SsnPorch'] + features['ScreenPorch'] + features['WoodDeckSF']
    
    features['HouseAge'] = features['YrSold'].astype(int) - features['YearBuilt']
    features['HouseAge'] = features['HouseAge'].apply(lambda x: max(0, x))
    features['RemodAge'] = features['YrSold'].astype(int) - features['YearRemodAdd']
    features['RemodAge'] = features['RemodAge'].apply(lambda x: max(0, x))
    
    features['IsNew'] = (features['YrSold'].astype(int) == features['YearBuilt']).astype(int)
    features['IsRemodeled'] = (features['YearRemodAdd'] != features['YearBuilt']).astype(int)
    
    # Interactions & Higher-order terms
    features['OverallQual_TotalSF'] = features['OverallQual'] * features['TotalSF']
    features['OverallQual_GrLivArea'] = features['OverallQual'] * features['GrLivArea']
    features['OverallQual_YearBuilt'] = features['OverallQual'] * features['YearBuilt']
    features['OverallGrade'] = features['OverallQual'] * features['OverallCond']
    features['ExterGrade'] = features['ExterQual'] * features['ExterCond']
    features['KitchenScore'] = features['KitchenQual'] * features['KitchenAbvGr']
    features['GarageScore'] = features['GarageCars'] * features['GarageArea']
    features['Bath_Bed_Ratio'] = features['TotalBath'] / (features['BedroomAbvGr'] + 1)
    features['AvgRoomSize'] = features['GrLivArea'] / (features['TotRmsAbvGrd'] + 1)
    
    # Area Proportions
    features['FinishedBsmtRatio'] = features['BsmtFinSF1'] / (features['TotalBsmtSF'] + 1)
    features['FirstFloorRatio'] = features['1stFlrSF'] / (features['TotalSF'] + 1)
    features['SecondFloorRatio'] = features['2ndFlrSF'] / (features['TotalSF'] + 1)
    features['PorchRatio'] = features['TotalPorch'] / (features['TotalSF'] + 1)
    
    # Binary Flags
    features['HasPool'] = (features['PoolArea'] > 0).astype(int)
    features['Has2ndFloor'] = (features['2ndFlrSF'] > 0).astype(int)
    features['HasGarage'] = (features['GarageArea'] > 0).astype(int)
    features['HasBsmt'] = (features['TotalBsmtSF'] > 0).astype(int)
    features['HasFireplace'] = (features['Fireplaces'] > 0).astype(int)
    features['HasWoodDeck'] = (features['WoodDeckSF'] > 0).astype(int)
    features['HasOpenPorch'] = (features['OpenPorchSF'] > 0).astype(int)
    features['HasEnclosedPorch'] = (features['EnclosedPorch'] > 0).astype(int)
    
    # 9. Handle Skewed Numeric Features with log1p
    numeric_feats = features.select_dtypes(include=[np.number]).columns
    skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
    
    for feat in high_skew:
        if (features[feat] < 0).any():
            continue
        features[feat] = np.log1p(features[feat])
        
    # 10. One-Hot Encoding
    features = pd.get_dummies(features, drop_first=True)
    
    X_train = features.iloc[:train_len].copy()
    X_test = features.iloc[train_len:].copy()
    
    print(f"Features created: {X_train.shape[1]}")
    return X_train, X_test, y_train, test_id
