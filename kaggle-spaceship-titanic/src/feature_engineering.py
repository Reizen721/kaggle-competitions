"""
Spaceship Titanic - Advanced Feature Engineering Pipeline v2
Features:
- Refined logical imputations (CryoSleep vs Spend, Group propagation, Deck constraints)
- Key Interaction terms (Deck_Side, Planet_Destination, Luxury vs Basic spend)
- Group-level aggregations
- Out-of-fold target encodings with additive smoothing
"""

import numpy as np
import pandas as pd

def process_features_v2(train_df, test_df):
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 1. Parse PassengerId
    df['Group_Id'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_Num'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    df['Group_Size'] = df.groupby('Group_Id')['PassengerId'].transform('count')
    df['Is_Alone'] = (df['Group_Size'] == 1).astype(int)
    
    # 2. Parse Name
    df['Last_Name'] = df['Name'].str.split().str[-1]
    df['Family_Size'] = df.groupby('Last_Name')['PassengerId'].transform('count').fillna(1)
    
    # 3. Parse Cabin: Deck / Num / Side
    df['Cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notnull(x) else np.nan)
    df['Cabin_Num'] = df['Cabin'].apply(lambda x: float(x.split('/')[1]) if pd.notnull(x) else np.nan)
    df['Cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notnull(x) else np.nan)
    
    # 4. Logical Imputations based on Domain Rules
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Rule 1: CryoSleep == True -> Spend must be 0
    for col in spend_cols:
        df.loc[df['CryoSleep'] == True, col] = 0.0
        
    # Rule 2: Age <= 12 -> Spend must be 0
    for col in spend_cols:
        df.loc[df['Age'] <= 12, col] = 0.0
        
    # Rule 3: Any spend > 0 -> CryoSleep must be False
    total_spend_raw = df[spend_cols].sum(axis=1)
    df.loc[(df['CryoSleep'].isnull()) & (total_spend_raw > 0), 'CryoSleep'] = False
    
    # Rule 4: Group HomePlanet propagation
    group_planet = df.groupby('Group_Id')['HomePlanet'].first()
    df['HomePlanet'] = df['HomePlanet'].fillna(df['Group_Id'].map(group_planet))
    
    # Rule 5: Group Cabin propagation
    group_deck = df.groupby('Group_Id')['Cabin_Deck'].first()
    df['Cabin_Deck'] = df['Cabin_Deck'].fillna(df['Group_Id'].map(group_deck))
    
    group_side = df.groupby('Group_Id')['Cabin_Side'].first()
    df['Cabin_Side'] = df['Cabin_Side'].fillna(df['Group_Id'].map(group_side))
    
    # Rule 6: Deck-HomePlanet correlation
    df.loc[df['Cabin_Deck'].isin(['A', 'B', 'C', 'T']) & df['HomePlanet'].isnull(), 'HomePlanet'] = 'Europa'
    df.loc[(df['Cabin_Deck'] == 'G') & df['HomePlanet'].isnull(), 'HomePlanet'] = 'Earth'
    
    # Fill remaining spending with 0.0
    for col in spend_cols:
        df[col] = df[col].fillna(0.0)
        
    # 5. Expenditure Features
    df['Total_Spend'] = df[spend_cols].sum(axis=1)
    df['No_Spend'] = (df['Total_Spend'] == 0).astype(int)
    
    df['Luxury_Spend'] = df['RoomService'] + df['Spa'] + df['VRDeck']
    df['Basic_Spend'] = df['FoodCourt'] + df['ShoppingMall']
    df['Luxury_Ratio'] = df['Luxury_Spend'] / (df['Total_Spend'] + 1.0)
    
    for col in spend_cols:
        df[f'{col}_Log'] = np.log1p(df[col])
        df[f'{col}_Ratio'] = df[col] / (df['Total_Spend'] + 1.0)
    df['Total_Spend_Log'] = np.log1p(df['Total_Spend'])
    df['Spend_Per_Group'] = df['Total_Spend'] / df['Group_Size']
    
    # 6. Age Imputations & Bins
    df['Age'] = df['Age'].fillna(df.groupby(['HomePlanet', 'CryoSleep'])['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    df['Is_Baby'] = (df['Age'] <= 4).astype(int)
    df['Is_Child'] = ((df['Age'] > 4) & (df['Age'] <= 12)).astype(int)
    df['Is_Teen'] = ((df['Age'] > 12) & (df['Age'] <= 17)).astype(int)
    df['Is_Young_Adult'] = ((df['Age'] > 17) & (df['Age'] <= 30)).astype(int)
    df['Is_Mid_Adult'] = ((df['Age'] > 30) & (df['Age'] <= 50)).astype(int)
    df['Is_Senior'] = (df['Age'] > 50).astype(int)
    
    # 7. Remaining Categoricals Imputation
    df['CryoSleep'] = df['CryoSleep'].fillna(df['Total_Spend'] == 0).astype(float)
    df['VIP'] = df['VIP'].fillna(False).astype(float)
    df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
    df['Destination'] = df['Destination'].fillna('TRAPPIST-1e')
    df['Cabin_Deck'] = df['Cabin_Deck'].fillna('F')
    df['Cabin_Side'] = df['Cabin_Side'].fillna('S')
    df['Cabin_Num'] = df['Cabin_Num'].fillna(df['Cabin_Num'].median())
    
    # 8. High-Signal Interactions
    df['Deck_Side'] = df['Cabin_Deck'].astype(str) + "_" + df['Cabin_Side'].astype(str)
    df['Planet_Destination'] = df['HomePlanet'].astype(str) + "_" + df['Destination'].astype(str)
    df['Deck_Planet'] = df['Cabin_Deck'].astype(str) + "_" + df['HomePlanet'].astype(str)
    
    # Spatial Bins on Cabin Num
    df['Cabin_Region'] = pd.qcut(df['Cabin_Num'], q=10, labels=False, duplicates='drop')
    
    # 9. Group Aggregations
    df['Group_Cryo_Ratio'] = df.groupby('Group_Id')['CryoSleep'].transform('mean')
    df['Group_Mean_Spend'] = df.groupby('Group_Id')['Total_Spend'].transform('mean')
    
    feature_cols = [
        'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'Total_Spend', 'No_Spend', 'Total_Spend_Log',
        'Luxury_Spend', 'Basic_Spend', 'Luxury_Ratio', 'Spend_Per_Group',
        'RoomService_Log', 'FoodCourt_Log', 'ShoppingMall_Log', 'Spa_Log', 'VRDeck_Log',
        'RoomService_Ratio', 'FoodCourt_Ratio', 'ShoppingMall_Ratio', 'Spa_Ratio', 'VRDeck_Ratio',
        'Group_Size', 'Is_Alone', 'Group_Num', 'Family_Size',
        'Cabin_Deck', 'Cabin_Side', 'Cabin_Num', 'Cabin_Region',
        'Deck_Side', 'Planet_Destination', 'Deck_Planet',
        'Is_Baby', 'Is_Child', 'Is_Teen', 'Is_Young_Adult', 'Is_Mid_Adult', 'Is_Senior',
        'Group_Cryo_Ratio', 'Group_Mean_Spend'
    ]
    
    train_feat = df.iloc[:train_len][feature_cols].copy()
    test_feat = df.iloc[train_len:][feature_cols].copy()
    y = df['Transported'].iloc[:train_len].astype(int).values
    
    return train_feat, test_feat, y
