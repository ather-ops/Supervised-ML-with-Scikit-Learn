# Spam Detection — Threshold Tuning with Logistic Regression

A real-world classification project where the model was technically working — but practically useless — until we understood threshold tuning. This README documents what went wrong, why, and how we fixed it.

---

## The Problem

We built a logistic regression model to detect spam. The full pipeline was already done:

- EDA and data exploration
- Missing value imputation
- Outlier detection
- Binning
- One-Hot Encoding
- Train-Test Split

Six solid steps of feature engineering. The model trained without errors. Then we looked at the confusion matrix.

**70+ spam emails were being predicted as "not spam."**

The model was missing the thing it existed to catch.

---

## Why It Happened — Default Threshold

Logistic regression doesn't output a hard Yes/No. It outputs a probability between 0 and 1. By default, scikit-learn converts that probability to a class using a threshold of **0.5** — anything below 0.5 becomes "no spam," anything above becomes "spam."

When your classes are imbalanced or the cost of a false negative is high (a spam email slipping through), 0.5 is often the wrong cutoff.

---

## The Fix — Threshold Tuning

Instead of accepting the default, we tested different thresholds manually and watched the confusion matrix change:

| Threshold | False Negatives (spam missed) |
|-----------|-------------------------------|
| Default (0.5) | 70+ |
| 0.40 | 65 |
| 0.50 | 45 |
| 0.60 | 23 |
| **0.65** | **11** |

Each step down in threshold made the model more aggressive at catching spam — and the confusion matrix showed it in real numbers.

---

## Before and After

**Before — default threshold, 70+ missed spam**

![Confusion Matrix Before](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn/blob/main/Assets/Python%203.13%206_10_2026%203_59_26%20PM.png?raw=true)

**After — threshold 0.65, 11 missed spam**

![Confusion Matrix After](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn/blob/main/Assets/Python%203.13%206_12_2026%208_47_14%20PM.png?raw=true)

---

## The Real-World Lesson

In a company setting, a spam filter missing 70 emails per batch is a liability. You would tune the threshold based on business cost:

- **High cost of missing spam** → lower threshold, catch more spam even if some legitimate mail gets flagged
- **High cost of false alarms** → raise threshold, be more conservative

The model doesn't decide this. You do. The threshold is a business decision expressed as a number.

---

## Full Pipeline

```
Load Data
    ↓
EDA — distributions, correlations, class balance
    ↓
Fill Missing Values
    ↓
Detect and Handle Outliers
    ↓
Binning — continuous features into categories
    ↓
One-Hot Encoding
    ↓
Train-Test Split
    ↓
Train Logistic Regression
    ↓
Evaluate — confusion matrix, precision, recall
    ↓
Tune threshold (0.4 → 0.5 → 0.6 → 0.65)
    ↓
Final evaluation at optimal threshold
```

---

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.13 | Core language |
| Pandas / NumPy | Data cleaning, feature engineering |
| Scikit-Learn | Logistic Regression, metrics |
| Matplotlib / Seaborn | EDA and confusion matrix plots |

---

## Key Takeaway

A trained model is not a finished model. Evaluation is part of the build. And threshold tuning is one of the simplest, highest-impact steps most beginners skip entirely.

---

## Repository

Part of [Supervised-ML-with-Scikit-Learn](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn) — a hands-on series covering the full supervised ML workflow from raw data to deployed model.
