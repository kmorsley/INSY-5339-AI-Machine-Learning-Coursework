# INSY 5339: Homework 8 Code Review and Improvements

**Student:** Keishana Morsley  
**UTA Student ID:** 1001807020  
**Course:** INSY 5339  
**Assignment:** Homework 8: k-Nearest Neighbors and Association Rules

## Review Scope

The following are substantive code decisions implemented and verified in the completed Python program. They are not represented as records of an actual GitHub Copilot interaction. The student should compare them with their own Copilot suggestions and document genuine prompts and revisions before presenting this file as evidence of the required Copilot workflow.

### Improvement 1: Stratified Partitioning and Leakage-Free Standardization

**Problem addressed:** A generic random partition can change the target-class proportions, and scaling all records before splitting would leak validation and test information into the model.

**Implemented improvement:** Split the original data first into 60 percent training and 40 percent temporary data, then split the temporary data evenly between validation and testing. Both steps use `random_state=42` and `stratify`. Fit `StandardScaler` on training predictors only and transform both held-out partitions with the original scaler. Customer_ID is retained for auditing, not used as a predictor.

**Verification:** The executed program reports separate split counts and generates `knn_scaling_comparison.csv` from the validation partition.

### Improvement 2: Select k Only from Validation Accuracy

**Problem addressed:** Evaluating multiple values of k on the test data would make the final test estimate optimistic, and an arbitrary tie-breaking convention could reduce reproducibility.

**Implemented improvement:** Evaluate only the specified odd k values (1 through 15) using standardized training and validation data. Select the highest validation accuracy and choose the smallest k in a tie. Evaluate the selected model on the untouched test partition only after selection.

**Verification:** `knn_k_comparison.csv` and `knn_validation_accuracy.png` document the selection. The selected value is **k = 15**, and the final test accuracy is **0.644231**.

### Improvement 3: Transparent Basket Encoding and Rule Filtering

**Problem addressed:** Untidy item strings can produce duplicate product labels, and confidence by itself does not establish positive association.

**Implemented improvement:** Split comma-separated basket contents into lists, strip surrounding whitespace, encode product membership in a binary matrix, use Apriori with minimum support 0.05, and retain rules only when confidence is at least 0.40 and lift exceeds 1.00. Antecedents and consequents are converted to readable text before export. The program prioritizes `mlxtend` and provides a mathematically equivalent offline fallback when the package is unavailable.

**Verification:** The run produced 53 frequent itemsets and 29 qualifying association rules.

### Improvement 4: Independent Manual Verification of the Highest-Lift Rule

**Problem addressed:** Library values should be independently checked against underlying transactions.

**Implemented improvement:** For the strongest rule, count all transactions, antecedent transactions, consequent transactions, and joint transactions directly from the original item lists. Compute support, confidence, and lift from the counts rather than the frequent-itemset output. Compare the manual values with the algorithm-generated values using `numpy.allclose`.

**Verification:** The top rule is **Baby_Wipes -> Diapers**. Counts are 1,000 transactions overall, 110 with Baby_Wipes, 101 with Diapers, and 65 with both. Support = 0.065, confidence = 0.590909, and lift = 5.850585. The independently computed metrics agree with the generated metrics.

## Student Workflow Evidence Still Required

This package was generated and executed for the assignment, but it cannot itself establish that GitHub Copilot was used or that three meaningful commits were made in the assigned private repository. Complete and accurately document those steps using your actual VS Code and GitHub activity, then push the Homework8 folder as the assignment requires. Do not replace this evidence with invented Copilot interactions or commit histories.
