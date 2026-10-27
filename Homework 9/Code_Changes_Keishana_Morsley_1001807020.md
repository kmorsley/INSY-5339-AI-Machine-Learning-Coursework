# Homework 9 | Code Review and Changes

**Course:** INSY 5339  
**Student:** Keishana Morsley  
**UTA ID:** 1001807020

## What I checked and changed in the final code

The points below describe concrete choices implemented in the submitted script. They are useful checkpoints for the required GitHub Copilot review, but they should not be presented as a record of a Copilot interaction unless that interaction actually occurred.

### 1. Kept customer IDs out of the model

I used a separate list of the eight categorical predictors instead of passing every spreadsheet column into the classifier. `Customer_ID` identifies a customer, but it does not describe their behavior. Including it could make the model learn meaningless patterns tied to record labels. The ID remains in the output CSV so predictions can still be traced to the original customers.

### 2. Put one-hot encoding inside the pipeline

I placed `OneHotEncoder(handle_unknown="ignore")` before `BernoulliNB` in a scikit-learn `Pipeline`. This makes training, testing, and new customer predictions use the same encoding logic. I also fit the pipeline only on the training split so the test set does not influence the learned preprocessing.

### 3. Used probabilities rather than only the model's default class labels

I used `predict_proba()` to retrieve the acceptance probability for class 1, then applied explicit cutoff values of 0.30, 0.50, and 0.70. This lets the business compare how many customers it would contact under a more inclusive or selective targeting rule. I checked that the 0.50 results match the main classification metrics.

### 4. Compared Laplace smoothing without changing the data split

For alpha values of 0.10, 1.00, and 5.00, I refit the same pipeline structure on the same training rows and evaluated on the same testing rows. This keeps the comparison focused on smoothing rather than differences caused by random samples.

### 5. Checked the model against a simple baseline

I calculated class priors from the training set and compared the model's test accuracy to always predicting the training majority class. This was important because accuracy alone can look good when one outcome is more common than another.

## Short reflection

The biggest lesson from this assignment is that choosing a classification cutoff changes how the model would be used in practice. A lower cutoff catches more potential acceptors, but it can also mean contacting more customers who would not accept. I would compare the cost of outreach with the value of a successful acceptance before choosing a final cutoff.

**Before submission:** The assignment specifically asks for changes made to *Copilot-generated or Copilot-suggested* code. The review above documents the finished implementation, not a verified Copilot session. If you use Copilot in VS Code, verify and update the statements to describe three suggestions you actually reviewed and modified.
