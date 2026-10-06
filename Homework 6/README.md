# Homework 6 – Logistic Regression with Scikit-learn

This assignment applies logistic regression to a bank marketing dataset to predict whether an existing customer will accept a personal loan offer. 

The project was completed using Python, scikit-learn, VS Code, and GitHub.

## Objective

The goal of this assignment is to build and evaluate a binary classification model where:

- `Loan_Accepted = 1` means the customer accepted the loan offer
- `Loan_Accepted = 0` means the customer did not accept the loan offer

The assignment also evaluates predicted probabilities, classification thresholds, ROC performance, logistic regression coefficients, and odds ratios.

## Dataset

The analysis uses the `Customer_Data` sheet from:

`Homework6_Logistic_Regression_Customer_Data.xlsx`

The dataset includes the following predictors:

- Age
- Annual Income
- Credit Score
- Account Tenure
- Digital Engagement Score
- Prior Product Count
- Mortgage Status
- Offer Channel

`Customer_ID` was excluded from the predictor variables, as required.

## Methods

The workflow included:

- Loading and reviewing the dataset
- Separating predictors and target variable
- Encoding the categorical `Offer_Channel` variable
- Scaling numeric predictor variables
- Creating training and testing datasets
- Fitting a logistic regression model
- Generating predicted probabilities
- Creating class predictions using a 0.50 threshold
- Comparing thresholds of 0.30, 0.50, and 0.70
- Evaluating accuracy, precision, recall, F1 score, and ROC AUC
- Creating a confusion matrix
- Creating an ROC curve
- Extracting logistic regression coefficients
- Calculating odds ratios
- Generating probability predictions for new customer profiles

## Model Performance

At the default classification threshold of 0.50, the model achieved approximately:

- Accuracy: 74.17%
- Precision: 69.23%
- Recall: 58.70%
- F1 Score: 0.6353
- ROC AUC: 0.8093

The threshold comparison showed that lower thresholds increased recall, while higher thresholds increased precision.

## Files Included

- `Homework6_Keishana_Morsley_1001807020.py`
- `Homework6_Logistic_Regression_Customer_Data.xlsx`
- `logistic_predictions.csv`
- `classification_metrics.csv`
- `confusion_matrix.csv`
- `threshold_comparison.csv`
- `logistic_coefficients.csv`
- `business_probability_predictions.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `Code_Changes_Keishana_Morsley_1001807020.md`

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- VS Code
- GitHub
- GitHub Copilot

## Key Concepts

This assignment demonstrates:

- Binary classification
- Logistic regression
- Predicted probabilities
- Classification thresholds
- Confusion matrix analysis
- Precision and recall tradeoffs
- ROC curves
- ROC AUC
- Logistic coefficients
- Odds ratios

## Course

INSY 5339 – Special Topics in Artificial Intelligence  
University of Texas at Arlington
