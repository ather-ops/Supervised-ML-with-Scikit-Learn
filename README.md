# Machine Learning with Scikit-Learn

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.0%2B-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-red)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [About](#about)
2. [Purpose](#purpose)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Projects Included](#projects-included)
6. [Tech Stack](#tech-stack)
7. [Dependencies](#dependencies)
8. [Related Repository](#related-repository)
9. [License](#license)
10. [Author](#author)

---

## About

This repository contains practical Machine Learning projects using **Scikit-Learn**. It complements my [Machine-Learning-from-scratch](https://github.com/ather-ops/Machine-Learning-from-scratch) repository by showing how to implement the same algorithms using industry-standard libraries.

> "First understand from scratch, then master the tools."

---

## Purpose

- Implement ML algorithms using Scikit-Learn
- Compare with from-scratch implementations
- Focus on real-world projects
- Learn best practices and pipelines
- Hyperparameter tuning and model selection
- Model deployment ready code

---

## Repository Structure
ML-with-Scikit-Learn/
│
├── 01-Data-Preprocessing/
│ ├── data_cleaning.ipynb
│ ├── feature_scaling.ipynb
│ └── outlier_detection.ipynb
│
├── 02-Linear-Regression/
│ ├── simple_linear_regression.ipynb
│ ├── multiple_linear_regression.ipynb
│ └── projects/
│ ├── house_price_prediction.ipynb
│ └── student_scores_prediction.ipynb
│
├── 03-Logistic-Regression/
│ ├── binary_classification.ipynb
│ ├── multiclass_classification.ipynb
│ └── projects/
│ ├── fraud_detection.ipynb
│ ├── loan_approval.ipynb
│ └── iris_classification.ipynb
│
├── 04-KNN/
│ ├── knn_classification.ipynb
│ ├── knn_regression.ipynb
│ └── projects/
│ └── wine_classification.ipynb
│
├── 05-Clustering/
│ ├── kmeans_clustering.ipynb
│ ├── hierarchical_clustering.ipynb
│ └── projects/
│ └── customer_segmentation.ipynb
│
├── 06-Decision-Trees/
│ ├── decision_tree_classifier.ipynb
│ ├── decision_tree_regressor.ipynb
│ └── projects/
│ └── titanic_survival.ipynb
│
├── 07-Ensemble-Methods/
│ ├── random_forest.ipynb
│ ├── gradient_boosting.ipynb
│ ├── ada_boost.ipynb
│ └── projects/
│ └── spam_classifier.ipynb
│
├── 08-Model-Evaluation/
│ ├── cross_validation.ipynb
│ ├── learning_curves.ipynb
│ ├── confusion_matrix.ipynb
│ └── roc_auc_curves.ipynb
│
├── 09-Dimensionality-Reduction/
│ ├── pca.ipynb
│ └── projects/
│ └── digits_visualization.ipynb
│
├── 10-Neural-Networks/
│ ├── mlp_classifier.ipynb
│ ├── mlp_regressor.ipynb
│ └── projects/
│ └── mnist_classification.ipynb
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── dataset_info.md
│
├── notebooks/
│ └── template.ipynb
│
├── utils/
│ └── helpers.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

---

## Quick Start

### Step 1: Clone the Repository

git clone https://github.com/ather-ops/ML-with-Scikit-Learn.git
cd ML-with-Scikit-Learn
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Run a Project
# For Python script
python "02-Linear-Regression/projects/house_price_prediction.py"

# For Jupyter Notebook
jupyter notebook "02-Linear-Regression/projects/house_price_prediction.ipynb"
Projects Included
#	Project Name	Algorithm Used	Folder Location	Difficulty
1	Student Scores Prediction	Linear Regression	02-Linear-Regression/projects/	⭐ Beginner
2	House Price Prediction	Multiple Linear Regression	02-Linear-Regression/projects/	⭐ Beginner
3	Fraud Detection	Logistic Regression	03-Logistic-Regression/projects/	⭐⭐ Intermediate
4	Loan Approval	Logistic Regression	03-Logistic-Regression/projects/	⭐⭐ Intermediate
5	Iris Classification	KNN / Logistic Regression	03-Logistic-Regression/projects/	⭐ Beginner
6	Wine Quality	Random Forest	07-Ensemble-Methods/projects/	⭐⭐ Intermediate
7	Titanic Survival	Decision Trees	06-Decision-Trees/projects/	⭐⭐ Intermediate
8	Customer Segmentation	K-Means Clustering	05-Clustering/projects/	⭐⭐ Intermediate
9	Spam Detection	Ensemble Methods	07-Ensemble-Methods/projects/	⭐⭐⭐ Advanced
10	MNIST Classification	Neural Networks	10-Neural-Networks/projects/	⭐⭐⭐ Advanced
Project Status
Status	Meaning
✅ Completed	Project is finished and working
🔄 In Progress	Currently working on this project
⬜ Planned	Will start soon
Tech Stack
Technology	Version	Purpose
Python	3.8+	Core programming language
Scikit-Learn	1.0+	Machine learning algorithms
Pandas	1.3+	Data manipulation and analysis
NumPy	1.21+	Numerical computations
Matplotlib	3.4+	Basic visualizations
Seaborn	0.11+	Statistical visualizations
Jupyter	1.0+	Interactive notebooks
Dependencies
Create a requirements.txt file with:

# Core ML
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Jupyter Environment
jupyter>=1.0.0
notebook>=6.4.0

# Utilities
scipy>=1.7.0
Install all dependencies:

bash
pip install -r requirements.txt
Related Repository
Repository	Link	Description
Machine-Learning-from-scratch	GitHub Link	: Same algorithms implemented from scratch using only NumPy
Comparison
Aspect	Scratch Repo	Sklearn Repo
Purpose: Understand mathematics	Learn industry tools
Code	from scratch, using libraries
Difficulty	Hard	Easy
Production Ready	No	Yes
Speed	Slow	Fast
Folder Descriptions
Folder	Contents
01-Data-Preprocessing/	Data cleaning, scaling, encoding, outlier detection
02-Linear-Regression/	Simple and multiple linear regression projects
03-Logistic-Regression/	Binary and multi-class classification
04-KNN/	K-Nearest Neighbours algorithms
05-Clustering/	Unsupervised learning algorithms
06-Decision-Trees/	Decision tree classifiers and regressors
07-Ensemble-Methods/	Random Forest, Gradient Boosting, AdaBoost
08-Model-Evaluation/	Cross-validation, metrics, curves
09-Dimensionality-Reduction/	PCA and feature reduction
10-Neural-Networks/	Multi-layer perceptron models
data/	All datasets (raw and processed)
notebooks/	Template notebooks
utils/	Helper functions
How to Contribute
Fork the repository

Create a new branch (git checkout -b feature/new-project)

Add your project in the appropriate folder

Commit changes (git commit -m "Add new project")

Push to branch (git push origin feature/new-project)

Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

MIT License

Copyright (c) 2026 Ather-ops

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
Author
Ather-ops

GitHub: @ather-ops

First Repository: Machine-Learning-from-scratch

Second Repository: ML-with-Scikit-Learn

Support
If you find this repository helpful:

⭐ Give it a star

📢 Share with friends

🐛 Report issues

🤝 Contribute projects

Acknowledgments
Scikit-Learn Documentation

Kaggle Datasets

Open Source Community

Changelog
Date	Version	Changes
March 2026	v1.0	Initial release with House Price Prediction project
Added repository structure
Created README documentation
Contact
For questions, suggestions, or feedback:

Open an Issue

Reach out on GitHub
