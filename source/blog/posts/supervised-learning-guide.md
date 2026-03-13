---
title: "Supervised Learning Guide: Algorithms, Workflows, and Best Practices"
description: "Deep dive into supervised learning — regression and classification algorithms, evaluation metrics, cross-validation, and how to pick the right model for your problem."
date: "2026-03-10"
slug: "supervised-learning-guide"
keywords: ["supervised learning", "supervised learning algorithms", "classification regression guide"]
---

## Learning Objectives

- Understand the difference between regression and classification tasks
- Know the most important supervised learning algorithms and when to use each
- Implement a full supervised learning pipeline with cross-validation
- Choose and interpret evaluation metrics correctly
- Diagnose and fix model performance problems

---

## Regression vs Classification

**Regression** predicts a continuous numeric output.
- House price prediction
- Sales forecasting
- Temperature prediction

**Classification** predicts a discrete category.
- Spam detection (binary: spam / not spam)
- Disease diagnosis (binary: positive / negative)
- Image recognition (multi-class: cat / dog / bird)

The key difference is your target variable: continuous → regression, categorical → classification.

---

## Core Algorithms

### Linear Regression
The simplest regression model. Fits a line (or hyperplane) through data.

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.3f}")
```

**Use when:** your relationship between features and target is approximately linear and you need interpretability.

### Logistic Regression
Despite the name, this is a classification algorithm. It outputs a probability between 0 and 1.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

**Use when:** you need a fast, interpretable baseline classifier or probability calibration matters.

### Decision Trees
Learns a tree of if/else rules. Highly interpretable.

```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
print(export_text(model, feature_names=list(load_breast_cancer().feature_names)))
```

**Use when:** you need explainability or are dealing with non-linear relationships.

### Random Forest
An ensemble of decision trees. Reduces overfitting through bagging.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = pd.Series(model.feature_importances_, index=load_breast_cancer().feature_names)
print(importances.sort_values(ascending=False).head(10))
```

**Use when:** you need a reliable, strong baseline without heavy tuning.

### Gradient Boosting (XGBoost / LightGBM)
Builds trees sequentially, each correcting the previous one's errors. State-of-the-art for tabular data.

```bash
pip install xgboost
```

```python
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
```

**Use when:** you want maximum performance on tabular/structured data.

---

## Evaluation Metrics

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean(|y - ŷ|) | Robust to outliers |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors more |
| R² | 1 - SS_res/SS_tot | 1.0 = perfect, 0 = predicts mean |

### Classification Metrics

| Metric | Formula | Use When |
|--------|---------|----------|
| Accuracy | correct/total | Balanced classes only |
| Precision | TP/(TP+FP) | False positives are costly |
| Recall | TP/(TP+FN) | False negatives are costly |
| F1 | 2×P×R/(P+R) | Imbalanced classes |
| AUC-ROC | Area under ROC curve | Ranking quality |

```python
from sklearn.metrics import roc_auc_score, f1_score

y_proba = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"F1:      {f1_score(y_test, model.predict(X_test)):.3f}")
```

**Rule of thumb:** Use F1 or AUC-ROC for imbalanced classification. Use RMSE for regression when large errors matter more.

---

## Cross-Validation

Never trust a single train/test split. Cross-validation gives you a reliable estimate of model performance.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**K-Fold CV:** Split data into K folds. Train on K-1 folds, test on 1. Repeat K times. Average the scores.

**Stratified K-Fold:** Preserves class proportions in each fold. Always use this for classification.

---

## Hyperparameter Tuning

### Grid Search
Exhaustive search over a specified parameter grid.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.3f}")
```

### Random Search
Sample random combinations — often finds good parameters faster than grid search.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [3, 5, 7, None],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)
```

---

## Feature Preprocessing

Most algorithms require scaled features. **Decision trees and tree ensembles are an exception** — they are scale-invariant.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

Always fit preprocessing on training data only. Apply the same transform to test data. Using a `Pipeline` makes this automatic.

---

## Troubleshooting

**Accuracy is high but the model is useless (imbalanced data)**
- Check class distribution: `y.value_counts()`
- Switch metrics to F1 or AUC-ROC
- Use `class_weight='balanced'`

**Model performs well on validation but poorly in production**
- Train/production distribution mismatch — check if live data looks different
- Temporal leakage — never train on future data to predict the past

**Feature importance shows unexpected results**
- Correlated features split importance between themselves
- Use permutation importance for a more reliable signal: `sklearn.inspection.permutation_importance`

---

## FAQ

**Should I normalize or standardize features?**
Standardize (zero mean, unit variance) for algorithms sensitive to scale: linear models, SVMs, neural networks. Tree-based models don't need it.

**How many training examples do I need?**
A rough rule: at least 10–30 examples per feature for linear models. Tree ensembles and neural networks typically need more. When in doubt, run a learning curve.

**What if I have missing values?**
Use `SimpleImputer` from scikit-learn (mean/median/most frequent) or `IterativeImputer` for more accuracy. Tree-based models like XGBoost handle missing values natively.

---

## What to Learn Next

- **Model evaluation in depth** → model-evaluation-and-metrics
- **Feature engineering** → feature-engineering-guide
- **Neural networks** → transformer-architecture-explained
- **Full ML roadmap** → [Machine Learning Roadmap](/blog/machine-learning-roadmap/)
