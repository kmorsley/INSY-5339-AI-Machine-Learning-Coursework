# Homework 9: Naïve Bayes Classification

**INSY 5339 | Keishana Morsley**

## Purpose

This project predicts whether customers will accept a financial product offer using eight categorical characteristics. It applies one-hot encoding and a Bernoulli Naïve Bayes classifier, then compares cutoff probabilities and Laplace smoothing settings.

## Run it

Open the `Homework9` folder in VS Code, select the course's `insy5339` Python 3.11 environment, and run:

```bash
python Homework9_Keishana_Morsley_1001807020.py
```

Needed packages: `pandas`, `openpyxl`, `numpy`, `scikit-learn`, and `matplotlib`. The supplied Excel workbook must stay in the same folder as the Python script. Running the script recreates the CSV outputs and PNG visualizations.

## Results at a glance

- Dataset: 800 customers, 10 columns, no missing values.
- Train/test data: 640/160 rows, using a stratified 80/20 split with random state 42.
- Majority-class test baseline accuracy: **64.38%**.
- BernoulliNB (`alpha=1.0`) test accuracy: **71.25%**.
- Precision: **64.86%**; recall: **42.11%**; F1: **0.5106**; ROC AUC: **0.7347**.
- Manual three-characteristic profile, estimated P(accepted): **76.82%**. This is a separate, unsmoothed demonstration, not the full-model result.
- Scenario A: **0.53%**, class 0; scenario B: **97.43%**, class 1; scenario C: **98.96%**, class 1.

## File guide

The `.py` file reproduces the analysis. The input `.xlsx` file contains `Customer_Data` and `Data_Dictionary`. The eight CSVs contain class priors, manual calculations, test-set predictions, main classification metrics, confusion-matrix totals, cutoff comparisons, smoothing comparisons, and predictions for three new customer profiles. The two PNGs visualize the confusion matrix and probability distribution. The `Code_Changes` Markdown file describes code review and implementation decisions.

## Interpretation

The classifier improves on the majority-class baseline, but recall is lower than precision at the 0.50 cutoff. In a marketing setting, that means it overlooks some customers who would accept an offer. A lower cutoff may find more of them, but increases false positives. The appropriate decision threshold depends on campaign costs and expected returns.

**Submission note:** The repository sync, GitHub Copilot interaction, three separate commits, and GitHub push have to be carried out in your own course repository and VS Code environment. This ZIP contains the project files; it is not evidence that those platform steps occurred.
