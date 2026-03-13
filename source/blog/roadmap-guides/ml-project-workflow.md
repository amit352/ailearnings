---
title: "ML Project Workflow: End-to-End Machine Learning in Practice"
description: "A practical guide to the full machine learning project lifecycle — problem framing, data collection, feature engineering, model selection, evaluation, and deployment — with Python code at every step."
date: "2026-03-10"
slug: "ml-project-workflow"
keywords: ["machine learning project workflow", "end to end ML pipeline", "ML project lifecycle"]
---

## The 7-Stage ML Project Lifecycle

Kaggle competitions skip 80% of the real work. Real ML projects follow this path:

1. Problem framing
2. Data collection and exploration
3. Feature engineering
4. Model training and selection
5. Evaluation and validation
6. Deployment
7. Monitoring

Let's walk through each with a real example: predicting customer churn.

---

## Stage 1: Problem Framing

Before writing a line of code, nail down:

- **What are you predicting?** (churn: yes/no → binary classification)
- **What counts as success?** (e.g., recall ≥ 80% on the churned class)
- **What's the baseline?** (naive model: always predict "no churn" → 85% accuracy but 0% recall on churners)
- **How will it be used?** (batch nightly? real-time API?)
- **What data is available?** (transaction history, support tickets, usage logs)

```python
# Document your problem definition
problem = {
    "task": "binary_classification",
    "target": "churned_in_30_days",
    "success_metric": "recall >= 0.80 on churned class",
    "baseline_accuracy": 0.85,
    "update_frequency": "daily_batch",
    "latency_requirement": "hours (not real-time)",
}
```

---

## Stage 2: Data Exploration (EDA)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("customer_data.csv")

# 1. Basic overview
print(df.shape)           # (10000, 25)
print(df.dtypes)
print(df.isnull().sum())  # check for missing values

# 2. Target distribution
print(df["churned"].value_counts())
# churned
# 0    8500  (85%)
# 1    1500  (15%)
# → class imbalance! need to handle this

# 3. Numeric feature distributions
df.describe()

# 4. Correlation with target
correlations = df.corr(numeric_only=True)["churned"].sort_values(ascending=False)
print(correlations.head(10))

# 5. Feature distributions by target
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ["monthly_spend", "support_tickets", "login_days", "contract_length", "plan_type_score"]
for ax, feat in zip(axes.flatten(), features):
    df.groupby("churned")[feat].hist(ax=ax, alpha=0.6, bins=30)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig("eda_distributions.png")
```

**EDA checklist:**
- [ ] Missing values — how many? random or systematic?
- [ ] Class imbalance — how severe?
- [ ] Outliers — real data or errors?
- [ ] Feature distributions — normal? skewed?
- [ ] Correlations — which features relate to target?

---

## Stage 3: Feature Engineering

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Handle missing values
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

df[num_cols] = imputer_num.fit_transform(df[num_cols])
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# 2. Encode categoricals
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 3. Create new features (domain knowledge)
df["spend_per_login"] = df["monthly_spend"] / (df["login_days"] + 1)
df["tickets_per_month"] = df["support_tickets"] / df["months_as_customer"]
df["is_month_to_month"] = (df["contract_type"] == "month-to-month").astype(int)
df["days_since_last_login_ratio"] = df["days_since_last_login"] / df["months_as_customer"]

# 4. Scale features
feature_cols = [c for c in df.columns if c != "churned"]
X = df[feature_cols].values
y = df["churned"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Feature engineering principles:**
- Domain knowledge beats algorithms — talk to business stakeholders
- Ratio features often work better than raw values
- Interaction features: `feature_a * feature_b`
- Time-based features: days since last purchase, frequency trends

---

## Stage 4: Model Training and Selection

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# Split data (stratified to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Try multiple models
models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=len(y[y==0])/len(y[y==1]), random_state=42),
}

# Cross-validate each
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
    results[name] = {"recall_mean": scores.mean(), "recall_std": scores.std()}
    print(f"{name}: recall = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Model selection guidelines:**
- Start simple (logistic regression) — good baseline, interpretable
- Random Forest: robust, handles missing values, feature importance
- XGBoost/LightGBM: usually best on tabular data
- Deep learning: only if you have > 100k samples and many features

---

## Stage 5: Evaluation

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)

# Train best model on full training set
best_model = xgb.XGBClassifier(
    scale_pos_weight=5.7,  # handle imbalance
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Full evaluation
print(classification_report(y_test, y_pred, target_names=["stayed", "churned"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
print(f"False Negatives (missed churners): {cm[1][0]}")  # costly!
print(f"False Positives (wrong alarms): {cm[0][1]}")

# Feature importance
import pandas as pd
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(feat_imp.head(10))
```

### Choosing the Right Threshold

By default, `predict()` uses 0.5. You can adjust:

```python
# For churn: we want high recall (catch all churners), accept more false positives
threshold = 0.30

y_pred_adjusted = (y_proba >= threshold).astype(int)
print(classification_report(y_test, y_pred_adjusted, target_names=["stayed", "churned"]))
```

---

## Stage 6: Deployment

```python
# Save model and preprocessors
import joblib

joblib.dump(best_model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

# Serve via FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")


class CustomerData(BaseModel):
    monthly_spend: float
    login_days: int
    support_tickets: int
    contract_type: str
    months_as_customer: int
    # ... other features


@app.post("/predict-churn")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    # Feature engineering (same as training)
    df["spend_per_login"] = df["monthly_spend"] / (df["login_days"] + 1)
    # ... apply same transformations
    X = scaler.transform(df[feature_cols])
    prob = model.predict_proba(X)[0][1]
    return {"churn_probability": round(float(prob), 3), "will_churn": prob >= 0.30}
```

---

## Stage 7: Monitoring

```python
# Track model performance over time
import sqlite3
from datetime import datetime

def log_prediction(customer_id, features, prediction, actual=None):
    with sqlite3.connect("model_log.db") as conn:
        conn.execute("""
            INSERT INTO predictions (customer_id, features, prediction, actual, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (customer_id, str(features), prediction, actual, datetime.now().isoformat()))

def check_data_drift(new_data, reference_data):
    """Detect if input distribution has shifted (simplified)."""
    from scipy.stats import ks_2samp
    drift_features = []
    for col in reference_data.columns:
        _, p_value = ks_2samp(reference_data[col], new_data[col])
        if p_value < 0.05:
            drift_features.append(col)
    return drift_features
```

**When to retrain:**
- Weekly/monthly on schedule
- When data drift detected
- When performance degrades below threshold

---

## What to Learn Next

- **Deploy your model** → [Deploying AI Applications](/blog/deploying-ai-applications/)
- **Feature engineering deep dive** → [Feature Engineering Guide](/blog/feature-engineering-guide/)
- **Model evaluation** → [Model Evaluation and Metrics](/blog/model-evaluation-and-metrics/)
