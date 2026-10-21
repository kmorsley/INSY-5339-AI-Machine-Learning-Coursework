# Homework 8 - k-Nearest Neighbors and Association Rules

**Course:** INSY 5339 - AI and Machine Learning  
**Student:** Keishana Morsley  
**Topic:** k-Nearest Neighbors (k-NN), Distance-Based Classification, Association Rules, Apriori, and Python Implementation

## Overview

Homework 8 explores two data-mining techniques used to support business decisions:

- **Part A: k-Nearest Neighbors (k-NN)** to classify customers according to whether they are likely to adopt a premium service.
- **Part B: Association Rule Mining** to identify products frequently purchased together and evaluate those relationships using support, confidence, and lift.

The assignment emphasizes appropriate preprocessing, model selection, evaluation, and interpretation of results.

## Dataset

**Assignment dataset:** `Homework8_KNN_Association_Rules_Data.xlsx`

The workbook contains three worksheets:

| Worksheet | Purpose |
| --- | --- |
| `KNN_Customer_Data` | Customer records for premium-service adoption classification |
| `Market_Basket_Transactions` | Shopping baskets for association-rule analysis |
| `Data_Dictionary` | Variable definitions |

## Part A: k-Nearest Neighbors Classification

The classification target is `Premium_Service_Adopted`, with `1` indicating adoption and `0` indicating non-adoption.

The six predictors are age, annual income, monthly spending, digital engagement score, account tenure, and monthly website visits. `Customer_ID` is retained for identification but is **not** used as a predictor.

### Workflow

1. Inspect the dataset, data types, missing values, and target-class distribution.
2. Create stratified training (60%), validation (20%), and testing (20%) partitions using `random_state=42`.
3. Compare unscaled and standardized k-NN classifiers at `k=5`. Fit `StandardScaler` using only the training data.
4. Evaluate `k` values of 1, 3, 5, 7, 9, 11, 13, and 15 using validation accuracy and error rate. Select the highest-accuracy value, choosing the smallest `k` in a tie.
5. Evaluate the selected classifier on the untouched test set using accuracy, precision, recall, F1 score, and a confusion matrix.
6. Predict premium-service adoption for three new-customer scenarios, and identify the nearest training customers for Scenario B.

## Part B: Association Rules and Market Basket Analysis

Market basket analysis identifies combinations of items occurring together in transactions. These patterns can inform merchandising, product bundling, and recommendations.

### Workflow

1. Load the transaction worksheet and split comma-separated item lists.
2. Convert transactions into a binary item matrix using `TransactionEncoder`.
3. Find frequent itemsets using Apriori with `min_support=0.05` and `use_colnames=True`.
4. Generate rules using confidence with `min_threshold=0.40`, retain rules with `lift > 1.00`, and rank them by lift.
5. Visualize the ten highest-lift rules in a horizontal bar chart.
6. Independently verify the highest-lift rule's support, confidence, and lift against the original transaction data.

Association rules show **co-occurrence**, not causation.

## Technologies

- Python 3.11 (`insy5339` course environment)
- pandas, NumPy, scikit-learn, matplotlib
- mlxtend (`TransactionEncoder`, Apriori, association rules)
- Visual Studio Code, GitHub, GitHub Copilot

If required, install mlxtend with:

```bash
python -m pip install mlxtend
```

## Required Assignment Deliverables

The following file names reflect the assignment instructions. The source-code and Copilot-review filenames below are shown without a student ID for privacy; check course submission naming requirements separately.

### Python code and review

- `Homework8_Keishana_Morsley.py` - main Python analysis program
- `Code_Changes_Keishana_Morsley.md` - review of at least three meaningful changes to Copilot suggestions

### k-NN results

- `knn_scaling_comparison.csv`
- `knn_k_comparison.csv`
- `knn_test_metrics.csv`
- `knn_predictions.csv`
- `knn_confusion_matrix.csv`
- `knn_new_customer_predictions.csv`
- `nearest_neighbors_scenario_B.csv`
- `knn_validation_accuracy.png`
- `knn_confusion_matrix.png`

### Association-rule results

- `basket_binary_matrix.csv`
- `frequent_itemsets.csv`
- `association_rules.csv`
- `association_rule_manual_check.csv`
- `top_10_rules_by_lift.png`

## Running the Analysis

1. Open the `Homework 8` folder in VS Code.
2. Activate the course's Python 3.11 `insy5339` environment and ensure required libraries are installed.
3. Place the assignment dataset where the Python script expects to read it.
4. Run the Homework 8 Python script.
5. Review the generated tables and plots and verify that all required outputs were produced.

## Submission Notes

The assignment instructions require a `Homework8` folder in the course repository, at least three meaningful Git commits, and a ZIP submission through Canvas. The ZIP must include all required deliverables. The original instructions specify student-ID-based names for certain submission files; the README intentionally omits the ID and does not change the professor's stated naming requirements.

## Learning Objectives

- Explain why feature scaling affects distance-based classifiers.
- Select k without using the test set for model tuning.
- Evaluate classification quality using multiple metrics.
- Interpret association rules with support, confidence, and lift.
- Verify algorithm-generated findings with independent calculations.
- Document and review AI-assisted code changes.
