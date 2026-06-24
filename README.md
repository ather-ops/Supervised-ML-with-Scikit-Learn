![ML with Scikit-Learn](https://raw.githubusercontent.com/ather-ops/Supervised-ML-with-Scikit-Learn/main/Assets/ml%20(1).png)

# Supervised Machine Learning with Scikit-Learn

A structured, 7-project curriculum applying classical ML algorithms through production-grade pipelines. This is the direct companion to [Machine-Learning-from-scratch](https://github.com/ather-ops/Machine-Learning-from-scratch) — together they form a complete two-repository learning system.

Decision Trees, Ensemble Methods (Random Forest / Gradient Boosting), and Neural Networks are covered in their own dedicated repositories — linked at the bottom of this README.

---

## The Two-Repository System

The from-scratch repository builds the mathematical foundation — every algorithm implemented in pure NumPy, no abstractions, no shortcuts. Once you understand why gradient descent works and what a Gini split is actually computing, this repository shows you how the industry applies those same concepts at speed and scale.

The separation is intentional. People who skip straight to sklearn end up treating it as a black box. People who only study from scratch never learn to build things fast. Both repositories together close that gap.

| Aspect | ML from Scratch | ML with Scikit-Learn |
|--------|-----------------|----------------------|
| Purpose | Understand the math | Use industry tools |
| Implementation | Pure NumPy, manual loops | Sklearn pipelines |
| Production ready | No | Yes |
| Speed | Slow | Optimized (C under the hood) |
| Best for | Learning deeply | Building and deploying |

---

## Projects

| # | Project | Algorithm | Difficulty | Status |
|---|---------|-----------|------------|--------|
| 01 | Student Scores Prediction | Linear Regression | Beginner | Complete |
| 02 | House Price Prediction | Multiple Linear Regression | Beginner | Complete |
| 03 | Iris Classification | Logistic Regression / KNN | Beginner | Complete |
| 04 | Fraud Detection | Logistic Regression | Intermediate | In Progress |
| 05 | Loan Approval | Logistic Regression | Intermediate | In Progress |
| 06 | Customer Segmentation | K-Means Clustering | Intermediate | Planned |
| 07 | Digits Visualization | PCA | Intermediate | Planned |

---

## The Pipeline Pattern

Every project in this repository uses `sklearn.pipeline.Pipeline`. This is not a style preference — it is the correct way to build ML workflows because it prevents data leakage from the test set into preprocessing steps.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Split first, then all preprocessing happens inside the pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"CV Score:  {cross_val_score(pipeline, X, y, cv=5).mean():.4f}")
print(classification_report(y_test, y_pred))

# Hyperparameter tuning — note the double underscore syntax for pipeline steps
param_grid = {
    'model__C':        [0.01, 0.1, 1, 10],
    'model__solver':   ['lbfgs', 'liblinear'],
    'model__max_iter': [100, 500, 1000]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best Params: {grid.best_params_}")
print(f"Best Score:  {grid.best_score_:.4f}")
```

The double-underscore syntax (`model__n_estimators`) is how `GridSearchCV` reaches inside a pipeline step to tune its parameters. Once this pattern clicks, tuning any model in any pipeline becomes mechanical.

---

## Key Concepts Across Modules

**Data leakage** is the most common mistake in ML projects. Fitting a scaler on the full dataset before splitting means test data influenced your preprocessing. Always split first, then fit transformers only on training data. The `Pipeline` object enforces this automatically.

**Cross-validation over a single train/test split.** A single split can be lucky or unlucky. Five-fold cross-validation gives you five estimates of generalization performance and a standard deviation. If the standard deviation is high, the model is sensitive to which data it sees — a signal to investigate further.

**The elbow method for K-Means.** Plot inertia against K and look for the point where the curve bends sharply. Beyond that point, adding clusters reduces inertia only marginally while making the model harder to interpret.

**Learning curves diagnose the problem before you tune.** If training score is high and validation score is low, you are overfitting — add regularization or more data. If both scores are low, you are underfitting — increase model complexity. Tuning hyperparameters without checking learning curves first is guesswork.

---

## Getting Started

```bash
git clone https://github.com/ather-ops/ML-with-Scikit-Learn.git
cd ML-with-Scikit-Learn
pip install -r requirements.txt
jupyter notebook
```

Open modules in order. If you are coming from the from-scratch repository, the sklearn equivalents will make immediate sense because you already know what each algorithm is computing.

If you are new to ML entirely, start at Module 1 and work forward. The preprocessing module is unglamorous but every project that follows depends on it.

---

## Related Repositories

These topics are covered in their own dedicated repositories and are intentionally excluded from this one.

| Repository | Topic |
|------------|-------|
| [decision-trees-forest](https://github.com/ather-ops/decision-trees-forest) | Decision Trees and Random Forest — from scratch and with sklearn |
| ather-ops/ensemble-methods | Gradient Boosting, AdaBoost, and stacking (coming soon) |
| ather-ops/neural-networks-sklearn | MLPClassifier, MLPRegressor, and the MNIST project (coming soon) |

---

## Dependencies

```text
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
scipy>=1.7.0
```

```bash
pip install -r requirements.txt
```

---

## Technologies

| Tool | Role |
|------|------|
| Python 3.8+ | Core language |
| Scikit-Learn | All ML algorithms, pipelines, and evaluation tools |
| Pandas | Data loading, cleaning, and feature engineering |
| NumPy | Numerical operations |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical visualizations |
| Jupyter | Interactive notebooks |

---

## License

MIT. Use freely.

---

## Author

[ather-ops](https://github.com/ather-ops)
