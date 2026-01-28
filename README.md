# Customer Payment Default Prediction Analysis

## Overview

Ever wonder why some customers pay their bills while others don't? This project dives into payment behavior data to build a machine learning system that predicts customer defaults before they happen.

Using decision trees and custom financial scoring, I developed a risk assessment model that increased profitability by 64% compared to offering credit to all customers indiscriminately.

## The Problem

A company offers "pay later" options to customers but loses money when customers default. The challenge: identify high-risk customers without losing good customers who would actually pay.

**The twist?** Traditional accuracy metrics don't work here. Missing a paying customer costs $80 in lost profit, but incorrectly trusting a defaulter loses $60. The model needed to optimize for dollars, not just accuracy.

## What I Built

A decision tree classifier with custom payoff-based scoring that:
- Predicts payment defaults with 85%+ accuracy
- Optimizes for expected financial value, not just correctness
- Identifies unemployment rate as the #1 risk factor (56% importance)
- Segments customers into actionable risk categories

## Key Results

- **$53.47** expected payoff per customer (vs $32.53 baseline)
- **64% improvement** in profitability
- **Clear decision rules** based on unemployment, location, and demographics
- **Business-ready recommendations** for credit policy

## Technical Highlights

- **Pandas** for data wrangling and feature engineering
- **Scikit-learn** for model building (DecisionTreeClassifier, GridSearchCV)
- **Custom scoring function** translating predictions into financial outcomes
- **Hyperparameter tuning** across 45 parameter combinations
- **Feature importance analysis** revealing key risk drivers

## Quick Start
```python
# Load data
df = pd.read_csv('customer_payments.csv')

# Train model with custom payoff scoring
clf = DecisionTreeClassifier(max_depth=8, class_weight={0:1, 1:2})
clf.fit(X_train, y_train)

# Predict and calculate expected value
predictions = clf.predict(X_test)
payoff = calculate_expected_payoff(y_test, predictions)
```

## Repository Structure
```
customer-payment-prediction/
├── notebook/
│   └── Customer_Payment_Default_Prediction_Analysis.ipynb
├── data/
│   └── README.md
├── images/
│   ├── decision_tree.png
│   ├── feature_importance.png
│   └── policy_comparison.png
└── README.md
```

## What Makes This Interesting

Most ML projects optimize for accuracy. This one optimizes for money. The decision tree doesn't just predict defaults - it calculates the expected dollar value of each decision, making it immediately applicable to real business strategy.

The model revealed that a customer's local unemployment rate matters 2.5x more than their income level. Counter-intuitive? Maybe. Actionable? Definitely.

## Technologies

Python • Pandas • Scikit-learn • Decision Trees • GridSearchCV • Custom Metrics • Matplotlib



*"The best model isn't the most accurate one - it's the one that makes the best decisions."*
