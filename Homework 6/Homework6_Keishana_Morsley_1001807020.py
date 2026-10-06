from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# Part 1 - Load and Understand the Data
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Homework6_Logistic_Regression_Customer_Data.xlsx"

customer_data = pd.read_excel(DATA_FILE, sheet_name="Customer_Data")

print("Dataset shape:", customer_data.shape)
print("\nFirst five rows:")
print(customer_data.head())
print("\nColumn data types:")
print(customer_data.dtypes)
print("\nMissing values by column:")
print(customer_data.isnull().sum())
print("\nTarget distribution:")
print(customer_data["Loan_Accepted"].value_counts())


# ------------------------------------------------------------
# Part 2 - Prepare the Predictor Variables
# ------------------------------------------------------------
# Customer_ID is an identifier only and is intentionally excluded.
X = customer_data.drop(columns=["Customer_ID", "Loan_Accepted"])
y = customer_data["Loan_Accepted"]

numeric_features = [
    "Age",
    "Annual_Income_K",
    "Credit_Score",
    "Account_Tenure_Years",
    "Digital_Engagement_Score",
    "Prior_Product_Count",
    "Has_Mortgage",
]

categorical_features = ["Offer_Channel"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
    ]
)


# ------------------------------------------------------------
# Part 3 - Create Training and Testing Data
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ------------------------------------------------------------
# Part 4 - Build the Logistic Regression Model
# ------------------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

model.fit(X_train, y_train)


# ------------------------------------------------------------
# Part 5 - Generate Probabilities and Predictions
# ------------------------------------------------------------
y_probability = model.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.50).astype(int)

prediction_output = pd.DataFrame(
    {
        "Customer_ID": customer_data.loc[X_test.index, "Customer_ID"].values,
        "Actual_Loan_Accepted": y_test.values,
        "Predicted_Probability": y_probability,
        "Predicted_Class_0.50": y_prediction,
    }
)

prediction_output.to_csv(BASE_DIR / "logistic_predictions.csv", index=False)


# ------------------------------------------------------------
# Part 6 - Evaluate Model Performance
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction, zero_division=0)
recall = recall_score(y_test, y_prediction, zero_division=0)
f1 = f1_score(y_test, y_prediction, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

metrics_output = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        "Value": [accuracy, precision, recall, f1, roc_auc],
    }
)
metrics_output.to_csv(BASE_DIR / "classification_metrics.csv", index=False)

print("\nClassification metrics at threshold 0.50:")
print(metrics_output.to_string(index=False))

# Confusion matrix values: TN, FP, FN, TP
cm = confusion_matrix(y_test, y_prediction)
tn, fp, fn, tp = cm.ravel()

confusion_output = pd.DataFrame(
    {
        "TN": [tn],
        "FP": [fp],
        "FN": [fn],
        "TP": [tp],
    }
)
confusion_output.to_csv(BASE_DIR / "confusion_matrix.csv", index=False)

# Confusion matrix visualization
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)
ax.set_title("Logistic Regression Confusion Matrix (Threshold = 0.50)")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Part 7 - Compare Different Classification Thresholds
# ------------------------------------------------------------
threshold_results = []

for threshold in [0.30, 0.50, 0.70]:
    threshold_prediction = (y_probability >= threshold).astype(int)
    threshold_cm = confusion_matrix(y_test, threshold_prediction)
    threshold_tn, threshold_fp, threshold_fn, threshold_tp = threshold_cm.ravel()

    threshold_results.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, threshold_prediction),
            "Precision": precision_score(y_test, threshold_prediction, zero_division=0),
            "Recall": recall_score(y_test, threshold_prediction, zero_division=0),
            "F1": f1_score(y_test, threshold_prediction, zero_division=0),
            "TN": threshold_tn,
            "FP": threshold_fp,
            "FN": threshold_fn,
            "TP": threshold_tp,
            "Predicted_Positive_Count": int(threshold_prediction.sum()),
        }
    )

threshold_comparison = pd.DataFrame(threshold_results)
threshold_comparison.to_csv(BASE_DIR / "threshold_comparison.csv", index=False)

print("\nThreshold comparison:")
print(threshold_comparison.to_string(index=False))


# ------------------------------------------------------------
# Part 8 - ROC Curve
# ------------------------------------------------------------
fpr, tpr, roc_thresholds = roc_curve(y_test, y_probability)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loan Acceptance")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(BASE_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Part 9 - Logistic Coefficients and Odds Ratios
# ------------------------------------------------------------
feature_names = model.named_steps["preprocess"].get_feature_names_out()
coefficients = model.named_steps["logistic_regression"].coef_[0]

coefficient_output = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Odds_Ratio": np.exp(coefficients),
    }
).sort_values(by="Odds_Ratio", ascending=False)

coefficient_output.to_csv(BASE_DIR / "logistic_coefficients.csv", index=False)

print("\nLogistic coefficients and odds ratios:")
print(coefficient_output.to_string(index=False))


# ------------------------------------------------------------
# Business Probability Predictions - Three New Customers
# ------------------------------------------------------------
# The homework names this required file but does not specify three customer profiles.
# These three illustrative profiles represent lower, moderate, and stronger engagement.
new_customers = pd.DataFrame(
    [
        {
            "Age": 28,
            "Annual_Income_K": 48.0,
            "Credit_Score": 655,
            "Account_Tenure_Years": 2.0,
            "Digital_Engagement_Score": 35,
            "Prior_Product_Count": 1,
            "Has_Mortgage": 0,
            "Offer_Channel": "Email",
        },
        {
            "Age": 42,
            "Annual_Income_K": 92.0,
            "Credit_Score": 720,
            "Account_Tenure_Years": 8.0,
            "Digital_Engagement_Score": 67,
            "Prior_Product_Count": 3,
            "Has_Mortgage": 1,
            "Offer_Channel": "Mobile",
        },
        {
            "Age": 55,
            "Annual_Income_K": 145.0,
            "Credit_Score": 775,
            "Account_Tenure_Years": 18.0,
            "Digital_Engagement_Score": 88,
            "Prior_Product_Count": 5,
            "Has_Mortgage": 1,
            "Offer_Channel": "Web",
        },
    ]
)

business_probabilities = model.predict_proba(new_customers)[:, 1]
business_predictions = (business_probabilities >= 0.50).astype(int)

business_output = new_customers.copy()
business_output.insert(0, "New_Customer", ["Customer_A", "Customer_B", "Customer_C"])
business_output["Predicted_Probability"] = business_probabilities
business_output["Predicted_Class_0.50"] = business_predictions
business_output.to_csv(BASE_DIR / "business_probability_predictions.csv", index=False)

print("\nThree new-customer probability predictions:")
print(business_output.to_string(index=False))


# ------------------------------------------------------------
# Part 10 - GitHub Copilot Review
# ------------------------------------------------------------
# Complete the separate Code_Changes_FirstName_LastName_UTAID.md file after
# reviewing this program with GitHub Copilot. Record at least the prompt(s),
# suggestions received, and which suggestions you accepted or rejected.

print("\nAll required CSV and PNG outputs were created in:")
print(BASE_DIR)
