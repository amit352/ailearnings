---
title: "Statistics for Machine Learning: Practical Concepts for Developers"
description: "The statistical concepts that actually matter for ML practitioners — probability distributions, Bayes' theorem, hypothesis testing, and how to interpret model evaluation metrics correctly."
date: "2026-03-10"
slug: "statistics-for-machine-learning"
keywords: ["statistics for machine learning", "probability for AI", "ML statistics guide"]
---

## What You Actually Need

This isn't a statistics textbook. These are the concepts that come up repeatedly when you're building, evaluating, and debugging ML models.

---

## Probability Fundamentals

### Probability Distributions

A distribution describes how likely different values are. The most important ones:

**Normal (Gaussian) Distribution**
```python
import numpy as np
import matplotlib.pyplot as plt

# Many natural phenomena follow this bell curve
# Parameters: mean (μ) and standard deviation (σ)
data = np.random.normal(loc=0, scale=1, size=10000)

# ~68% of data falls within 1 std dev of mean
# ~95% within 2 std devs
# ~99.7% within 3 std devs (the "three-sigma rule")

plt.hist(data, bins=50, density=True)
plt.title("Normal Distribution")
plt.show()
```

**Why it matters for ML:** Model weights in neural networks are often initialized from normal distributions. Residuals (prediction errors) often follow normal distributions in regression.

**Bernoulli Distribution** — binary outcomes (0 or 1). Probability of a coin flip. Basis of binary classification: the model predicts P(y=1 | x).

**Softmax output** — converts raw scores to a probability distribution over classes. Each output is between 0–1 and all outputs sum to 1.

```python
import numpy as np

def softmax(logits):
    # Numerically stable softmax
    e = np.exp(logits - logits.max())
    return e / e.sum()

logits = np.array([2.0, 1.0, 0.5, -1.0])   # raw scores for 4 classes
probs = softmax(logits)
print(probs)          # [0.619, 0.228, 0.138, 0.019]
print(probs.sum())    # 1.0
```

---

## Expected Value and Variance

**Expected value (mean)** — the average outcome over many trials.

**Variance** — how spread out values are. High variance = high uncertainty.

**Standard deviation** — √variance, in the same units as your data.

```python
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

print(f"Mean:  {data.mean():.2f}")     # 5.0
print(f"Std:   {data.std():.2f}")      # 2.0
print(f"Var:   {data.var():.2f}")      # 4.0

# For model predictions
predictions = np.array([0.9, 0.7, 0.85, 0.6, 0.95])
print(f"Avg confidence: {predictions.mean():.2f}")
print(f"Confidence spread: {predictions.std():.2f}")
```

**Bias-variance tradeoff** — a fundamental ML concept:
- **High bias** (underfitting): model too simple, misses patterns
- **High variance** (overfitting): model too complex, memorizes training data
- Goal: minimize both

---

## Bayes' Theorem

The most important formula in probabilistic ML:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

In English: "Given that B happened, what's the probability of A?"

**Practical example — spam filter:**

```python
# Prior: 30% of emails are spam
p_spam = 0.30

# Likelihood: "free money" appears in 80% of spam, 2% of ham
p_word_given_spam = 0.80
p_word_given_ham  = 0.02

# P(word) = P(word|spam)*P(spam) + P(word|ham)*P(ham)
p_word = p_word_given_spam * p_spam + p_word_given_ham * (1 - p_spam)

# Posterior: P(spam | word)
p_spam_given_word = (p_word_given_spam * p_spam) / p_word
print(f"P(spam | 'free money'): {p_spam_given_word:.2%}")  # 94.5%
```

Bayesian thinking appears in many places: Naive Bayes classifiers, Bayesian optimization for hyperparameter tuning, uncertainty estimation.

---

## Correlation and Causation

**Correlation** measures the linear relationship between two variables (range: -1 to +1).

```python
import pandas as pd

df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "test_score":  [50, 55, 60, 65, 70, 75, 80, 85],
    "shoe_size":   [9, 8, 10, 7, 9, 8, 10, 9],  # irrelevant
})

print(df.corr(numeric_only=True))
# study_hours ↔ test_score: ~0.99 (strong positive correlation)
# shoe_size ↔ test_score: ~0.0 (no correlation)
```

**Critical point:** Correlation ≠ causation. Ice cream sales correlate with drowning deaths (both increase in summer). Never assume a correlated feature *causes* the outcome.

---

## Hypothesis Testing: Does Your Model Actually Improve?

When you change something (add a feature, change architecture), you need to know if the improvement is real or random noise.

```python
from scipy import stats

# Old model: 500 test examples, 75% accuracy
# New model: 500 test examples, 77% accuracy
# Is the 2% improvement real or noise?

old_correct = 375  # 75% of 500
new_correct = 385  # 77% of 500
n = 500

# Two-proportion z-test
from statsmodels.stats.proportion import proportions_ztest

counts = [new_correct, old_correct]
nobs = [n, n]
z, p_value = proportions_ztest(counts, nobs)

print(f"Z-statistic: {z:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Improvement is {'statistically significant' if p_value < 0.05 else 'NOT significant'}")
```

**P-value < 0.05** means there's less than 5% chance the result is due to random chance. A common but imperfect threshold.

**Practical rule:** With fewer than ~500 test examples, small improvements (1-2%) are rarely meaningful. Always test on large, held-out datasets.

---

## Information Theory: Entropy and Cross-Entropy

**Entropy** measures uncertainty/information content:

```python
import numpy as np

def entropy(probs):
    """Shannon entropy: lower = more certain."""
    return -sum(p * np.log2(p) for p in probs if p > 0)

# Completely uncertain (uniform distribution over 4 classes)
uniform = [0.25, 0.25, 0.25, 0.25]
print(f"Uniform entropy: {entropy(uniform):.2f} bits")  # 2.0 bits (max)

# Completely certain
certain = [1.0, 0.0, 0.0, 0.0]
print(f"Certain entropy: {entropy(certain):.2f} bits")   # 0.0 bits
```

**Cross-entropy loss** — the loss function used in almost all classification models:

```python
def cross_entropy_loss(y_true, y_pred):
    """
    y_true: true class probabilities (one-hot or soft labels)
    y_pred: model's predicted probabilities
    """
    eps = 1e-10  # prevent log(0)
    return -np.sum(y_true * np.log(y_pred + eps))

# Perfect prediction
y_true = np.array([0, 1, 0, 0])  # true class is index 1
y_pred = np.array([0.02, 0.90, 0.05, 0.03])
print(f"Good prediction loss: {cross_entropy_loss(y_true, y_pred):.3f}")  # ~0.105

# Bad prediction
y_pred_bad = np.array([0.25, 0.25, 0.25, 0.25])
print(f"Random prediction loss: {cross_entropy_loss(y_true, y_pred_bad):.3f}")  # ~1.386
```

Why it matters: when you see "loss" in training curves, it's usually cross-entropy. Lower is better.

---

## Evaluating Classification Models

Beyond accuracy, you need these metrics:

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]

# Precision: of predicted positives, what % are actually positive?
# (Low precision = many false alarms)
print(f"Precision: {precision_score(y_true, y_pred):.2f}")

# Recall: of actual positives, what % did we catch?
# (Low recall = missed many positives)
print(f"Recall: {recall_score(y_true, y_pred):.2f}")

# F1: harmonic mean of precision and recall
print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")
#       Predicted
#       0    1
# True 0 [TN, FP]
#      1 [FN, TP]
```

**Rule of thumb:**
- Medical diagnosis → prioritize recall (don't miss sick patients)
- Spam filter → prioritize precision (don't block real emails)
- Search ranking → use NDCG or MRR

---

## Statistical Concepts in LLM Development

**Perplexity** — measures how well a language model predicts text. Lower is better. A model with perplexity 20 is less "surprised" by text than one with perplexity 100.

**Temperature sampling** — not technically statistics, but uses the probability distribution output:
- `temperature=0` → argmax (most likely token always)
- `temperature=1` → sample from the distribution
- `temperature=2` → exaggerate low-probability tokens (more random)

**Top-p (nucleus) sampling** — sample only from the smallest set of tokens whose cumulative probability ≥ p. Prevents sampling from the "long tail" of unlikely tokens.

---

## What to Learn Next

- **Machine learning in practice** → [ML Project Workflow](/blog/roadmap-guides/ml-project-workflow/)
- **Model evaluation** → [Model Evaluation and Metrics](/blog/model-evaluation-and-metrics/)
- **Deep learning** → [Deep Learning Fundamentals](/blog/roadmap-guides/deep-learning-fundamentals/)
