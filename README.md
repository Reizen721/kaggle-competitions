# Kaggle Machine Learning Competitions Portfolio

This repository hosts solutions, feature engineering pipelines, and ensemble modeling code for completed **Kaggle Getting Started / Beginner** machine learning competitions.

All notebooks are verified and published on Kaggle under [@davidzwang](https://www.kaggle.com/davidzwang).

---

## 🏆 Competitions Summary & Leaderboard Standings

| Competition | Domain / Task | Metric | Leaderboard Score | Leaderboard Rank | Kaggle Notebook | Source Code |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **[NLP Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)** | NLP / Text Classification | F1-Score | **0.84431** | **Rank 66 / 602** (Top 10.96%) | [Kaggle Notebook](https://www.kaggle.com/code/davidzwang/nlp-disaster-tweets-ensemble-denoised) | [Source Code](./kaggle-nlp-disaster) |
| **[House Prices: Advanced Regression](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)** | Tabular Regression | RMSLE | **0.12047** | **Rank 650 / 4,929** (Top 13.19%) | [Kaggle Notebook](https://www.kaggle.com/code/davidzwang/house-prices-slsqp-stacking-regression) | [Source Code](./kaggle-house-prices) |
| **[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)** | Tabular Binary Classification | Accuracy | **0.80734** | **Rank 504 / 2,168** (Top 23.25%) | [Kaggle Notebook](https://www.kaggle.com/code/davidzwang/spaceship-titanic-calibrated-trio-ensemble) | [Source Code](./kaggle-spaceship-titanic) |
| **[Store Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)** | Multi-Table Time Series | RMSLE | **0.44282** | **Rank 349 / 666** | [Kaggle Notebook](https://www.kaggle.com/code/davidzwang/store-sales-forecasting-histgbm-ensemble) | [Source Code](./kaggle-store-sales) |

---

## 📌 Project Highlights & Methodologies

### 1. [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- **Task**: Classify whether a tweet announces a real disaster or uses figurative/metaphorical language.
- **Kaggle Notebook**: [nlp-disaster-tweets-ensemble-denoised](https://www.kaggle.com/code/davidzwang/nlp-disaster-tweets-ensemble-denoised)
- **Key Methods**:
  - **Label Denoising**: Identified duplicate tweet texts with conflicting ground-truth targets and resolved noise via consensus majority voting.
  - **Feature Engineering**: Multi-scale TF-IDF (word $1-3$ n-grams + character $3-5$ n-grams) combined with meta-linguistic features (URL/User tokens, punctuation density, capitalization ratio).
  - **Modeling**: 5-fold stratified stacking combining Logistic Regression, Ridge, Complement Naive Bayes, and SGDClassifier with dynamic threshold optimization and exact-match test deduplication.
- **Result**: **0.84431 F1-Score** (**Rank 66 / 602**, Top 10.96%).

---

### 2. [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Task**: Predict residential sale prices in Ames, Iowa using 79 structural, spatial, and zoning features.
- **Kaggle Notebook**: [house-prices-slsqp-stacking-regression](https://www.kaggle.com/code/davidzwang/house-prices-slsqp-stacking-regression)
- **Key Methods**:
  - **Outlier Pruning & Context Imputation**: Removed Dean De Cock atypical sales (`GrLivArea > 4000` with abnormal price); applied localized micro-zoning imputation for `IDOTRR` neighborhood.
  - **Spatial Hierarchy**: Constructed a 6-tier `Neighborhood_Tier` index and high-order interaction terms (`TotalSF`, bathroom metrics, age depreciation penalty).
  - **Convex Stacking**: 10-fold SLSQP-constrained optimization to learn optimal non-negative ensemble weights across CatBoost, GradientBoostingRegressor, Ridge, Lasso, and ElasticNet.
- **Result**: **0.12047 RMSLE** (**Rank 650 / 4,929**, Top 13.19%).

---

### 3. [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- **Task**: Predict whether passengers aboard the collided spaceship were transported to an alternate dimension.
- **Kaggle Notebook**: [spaceship-titanic-calibrated-trio-ensemble](https://www.kaggle.com/code/davidzwang/spaceship-titanic-calibrated-trio-ensemble)
- **Key Methods**:
  - **Domain Physics Imputation**: Leveraged deterministic domain constraints (CryoSleep state and age restrictions governing zero luxury spending; deck-to-home-planet physical rules; group-level attribute propagation).
  - **Feature Extraction**: Decomposed passenger and cabin identifiers into `Deck`, `Num`, `Side`, `Group_Size`, and `Family_Size`.
  - **Modeling & Calibration**: 10-fold stratified stacking of diverse GBDT architectures (**CatBoost + XGBoost + HistGradientBoosting**) with decision threshold calibrated to the test set transport prior (~50.36%).
- **Result**: **0.80734 Accuracy** (**Rank 504 / 2,168**, Top 23.25%).

---

### 4. [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Task**: Forecast 16-day future sales across 54 stores and 33 product families in Ecuador (28,512 test series).
- **Kaggle Notebook**: [store-sales-forecasting-histgbm-ensemble](https://www.kaggle.com/code/davidzwang/store-sales-forecasting-histgbm-ensemble)
- **Key Methods**:
  - **Multi-Source Domain Fusion**: Daily oil price interpolation and 7-day rolling trends; hierarchical holiday joining (National, Regional, Local) filtered for non-transferred events.
  - **Ecuadorian Retail Insights**: Modeled bi-weekly "Quincena" payday demand spikes (15th and end-of-month); zeroed out 53 dormant lifetime series (848 test items) to remove tree smoothing noise.
  - **Zero-Leakage Architecture**: Strictly limited lags to $\ge 16$ days (`lag_16` to `lag_28`, rolling averages) with an ultra-memory-efficient dual HistGradientBoosting ensemble.
- **Result**: **0.44282 RMSLE** (**Rank 349 / 666**).

---

## 📂 Repository Structure

```text
.
├── README.md                                  # Portfolio overview (this document)
├── kaggle-nlp-disaster/                       # NLP Disaster Tweets project
│   ├── src/                                   # Feature extraction & modeling scripts
│   └── kaggle_kernel_public/                  # Public Kaggle notebook files
├── kaggle-house-prices/                       # House Prices Advanced Regression
│   ├── src/                                   # Domain imputation & SLSQP stacking
│   └── kaggle_kernel/                         # Public Kaggle notebook files
├── kaggle-spaceship-titanic/                  # Spaceship Titanic Tabular Classification
│   ├── src/                                   # Logical imputations & 10-fold trio GBDT
│   └── kaggle_kernel/                         # Public Kaggle notebook files
└── kaggle-store-sales/                        # Store Sales Multi-Table Time Series
    ├── src/                                   # Holiday/Oil joins & HistGBM forecaster
    └── kaggle_kernel/                         # Public Kaggle notebook files
```
