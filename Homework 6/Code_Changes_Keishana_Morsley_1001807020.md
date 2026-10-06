# Homework 6 - GitHub Copilot Review

## File Reviewed
`Homework6_Keishana_Morsley_UTAID.py`

## Suggested Copilot Prompt
Review this logistic regression homework program for correctness, readability, reproducibility, and scikit-learn best practices. Check the train/test split, preprocessing, probability predictions, classification thresholds, ROC/AUC calculation, confusion matrix, coefficient/odds-ratio extraction, file exports, and handling of the categorical Offer_Channel predictor. Do not change the assignment requirements.

## Review Checklist
Use GitHub Copilot in VS Code and confirm each item below after its review.

- [ ] `Customer_ID` is excluded from the predictor variables.
- [ ] `Loan_Accepted` is used as the binary target.
- [ ] `Offer_Channel` is one-hot encoded.
- [ ] Numeric predictors are standardized.
- [ ] The train/test split is reproducible and stratified.
- [ ] Logistic regression is fitted only on the training data.
- [ ] Predicted probabilities use `predict_proba()`.
- [ ] The default classification threshold is 0.50.
- [ ] Thresholds 0.30, 0.50, and 0.70 are compared.
- [ ] Accuracy, precision, recall, F1, and ROC AUC are calculated.
- [ ] The confusion matrix exports TN, FP, FN, and TP.
- [ ] The ROC curve is saved as a PNG.
- [ ] Coefficients and odds ratios are exported.
- [ ] Three new-customer probability predictions are exported.
- [ ] All required output files are written to the Homework6 folder.

## Changes Made After Copilot Review
Complete this section after running the prompt above in GitHub Copilot. Only record changes that Copilot actually suggested and that you actually made.

1. **Suggestion:** 
   **Decision:** Accepted / Rejected
   **Change made or reason rejected:** 

2. **Suggestion:** 
   **Decision:** Accepted / Rejected
   **Change made or reason rejected:** 

3. **Suggestion:** 
   **Decision:** Accepted / Rejected
   **Change made or reason rejected:** 

## Final Verification
After reviewing any Copilot suggestions, run the Python file again and confirm that all CSV and PNG outputs are recreated without errors.
