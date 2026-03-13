---
title: "Feature Engineering Guide: Transform Raw Data into Model-Ready Features"
description: "Practical feature engineering techniques for ML engineers — encoding, scaling, handling missing values, creating new features, and selecting the most useful ones."
date: "2026-03-10"
slug: "feature-engineering-guide"
keywords: ["feature engineering", "feature engineering machine learning", "ML feature engineering"]
---

## Learning Objectives

- Handle missing values, categorical variables, and outliers
- Apply the right encoding strategy for different categorical types
- Create new features that improve model performance
- Scale and normalize features correctly
- Select the most informative features and remove noise

---

## Why Feature Engineering Matters

In practice, feature engineering often contributes more to model performance than algorithm selection. A simple model on great features beats a complex model on poor features.

The feature engineering pipeline is: **raw data → cleaned data → encoded data → scaled data → selected features → model input**.

---

## Handling Missing Values

### Strategy 1: Drop
Drop rows or columns with missing values. Only safe if missingness is rare and random.

```python
df.dropna(subset=['important_column'], inplace=True)  # drop rows
df.drop(columns=['mostly_null_column'], inplace=True)  # drop columns
```

### Strategy 2: Impute with Statistics

```python
from sklearn.impute import SimpleImputer
import pandas as pd

# Numerical: fill with median (robust to outliers)
num_imputer = SimpleImputer(strategy='median')
df[['age', 'income']] = num_imputer.fit_transform(df[['age', 'income']])

# Categorical: fill with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['city']] = cat_imputer.fit_transform(df[['city']])
```

### Strategy 3: Add a Missingness Indicator
Missing data itself can be informative. Add a binary flag before imputing.

```python
df['age_was_missing'] = df['age'].isna().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)
```

### Strategy 4: Model-Based Imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iter_imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(iter_imputer.fit_transform(df), columns=df.columns)
```

---

## Encoding Categorical Variables

### Label Encoding
Assigns each category an integer. Only appropriate for ordinal categories where order matters.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
# "high school" → 0, "bachelor" → 1, "master" → 2
```

**Warning:** Don't use label encoding for nominal categories with tree-ensemble models — it implies a false ordering.

### One-Hot Encoding
Creates a binary column for each category. Use for nominal categories with low cardinality.

```python
df_encoded = pd.get_dummies(df, columns=['city', 'department'], drop_first=True)

# Or with scikit-learn:
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded = ohe.fit_transform(df[['city']])
```

### Ordinal Encoding
For ordered categories (low/medium/high, bronze/silver/gold).

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['risk_encoded'] = oe.fit_transform(df[['risk_level']])
```

### Target Encoding (Mean Encoding)
Replace each category with the mean of the target variable for that category. Powerful for high-cardinality features but prone to overfitting — use with cross-validation.

```python
# Manual implementation
target_mean = df.groupby('city')['churn'].mean()
df['city_target_encoded'] = df['city'].map(target_mean)
```

### Frequency Encoding
Replace each category with how often it appears in the dataset.

```python
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

---

## Scaling Numerical Features

### StandardScaler (Z-score normalization)
Centers to mean=0, std=1. Best for normally distributed features and algorithms sensitive to scale (linear models, SVM, neural networks).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use fit from training set only
```

### MinMaxScaler
Scales to a fixed range [0, 1]. Preserves zero values. Sensitive to outliers.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

### RobustScaler
Uses median and IQR instead of mean and std. Best when your data has many outliers.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
```

**Tree-based models (Random Forest, XGBoost) don't require scaling.** Neural networks, linear models, and SVMs do.

---

## Handling Outliers

### Detect Outliers

```python
# IQR method
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['income'] < lower) | (df['income'] > upper)]
print(f"Outliers: {len(outliers)}")
```

### Treatment Options
- **Cap (Winsorize):** Clip values to [lower, upper]
- **Log transform:** Compress skewed distributions
- **Remove:** Only if outliers are data entry errors

```python
# Winsorize
df['income'] = df['income'].clip(lower=lower, upper=upper)

# Log transform (for right-skewed data)
import numpy as np
df['income_log'] = np.log1p(df['income'])  # log1p handles zeros
```

---

## Creating New Features

### Date/Time Features

```python
df['date'] = pd.to_datetime(df['date'])
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
df['hour']        = df['date'].dt.hour
```

### Interaction Features

```python
df['income_per_dependent'] = df['income'] / (df['dependents'] + 1)
df['age_times_income']     = df['age'] * df['income']
```

### Binning / Discretization

```python
# Manual bins
df['age_group'] = pd.cut(df['age'],
    bins=[0, 18, 35, 50, 65, 100],
    labels=['teen', 'young_adult', 'adult', 'middle_age', 'senior'])

# Quantile bins (equal-frequency)
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4'])
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X[['age', 'income']])
```

---

## Feature Selection

### Filter Methods — Statistical Tests

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Select top 10 features by ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(selected_features.tolist())
```

### Wrapper Method — Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=10)
rfe.fit(X_train, y_train)
print(X.columns[rfe.support_].tolist())
```

### Embedded Method — Feature Importance

```python
import pandas as pd
model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.nlargest(15)
print(top_features)
```

### Permutation Importance (Most Reliable)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
print(perm_imp.head(15))
```

---

## Building a Feature Pipeline

Combine all preprocessing steps into a reproducible pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_features = ['age', 'income', 'tenure']
categorical_features = ['city', 'plan_type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier()),
])

full_pipeline.fit(X_train, y_train)
score = full_pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

---

## Troubleshooting

**Model performance doesn't improve after adding features**
→ Check correlation with target. Low correlation = low predictive power. Use `SelectKBest` to filter.

**One-hot encoding creates too many columns**
→ Use target encoding or frequency encoding for high-cardinality categoricals (>20 unique values).

**Train/test performance gap is large despite regularization**
→ Check for data leakage — ensure no future information sneaks into features.

---

## FAQ

**When should I do feature selection?**
After basic preprocessing. Use feature importance from a quick baseline model to identify candidates for removal. Always validate that removing a feature doesn't hurt performance.

**Does feature engineering matter for deep learning?**
Less so for raw data like images and text (deep learning learns features automatically). For tabular data, yes — good feature engineering still matters even with deep learning.

---

## What to Learn Next

- **Model evaluation** → [Model Evaluation and Metrics](/blog/model-evaluation-and-metrics/)
- **Python for ML** → python-for-machine-learning
- **Full roadmap** → [Machine Learning Roadmap](/blog/machine-learning-roadmap/)
