"""INSY 5339 | Homework 9: Naive Bayes classification.

Run from the Homework9 directory: python Homework9_Keishana_Morsley_1001807020.py
Dependencies: pandas, openpyxl, numpy, matplotlib, scikit-learn.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'Homework9_Naive_Bayes_Customer_Data.xlsx'
TARGET = 'Offer_Accepted'
ID = 'Customer_ID'
PREDICTORS = ['Age_Group', 'Income_Band', 'Employment_Status', 'Credit_Profile', 'Digital_Engagement',
              'Prior_Product_Count_Band', 'Preferred_Channel', 'Recent_Campaign_Response']

def save_csv(data, filename):
    data.to_csv(HERE / filename, index=False, float_format='%.8f')

def pipeline(alpha=1.0):
    return Pipeline([('encode', OneHotEncoder(handle_unknown='ignore')),
                     ('naive_bayes', BernoulliNB(alpha=alpha))])

def metrics(actual, predicted, probability):
    return {'Accuracy': accuracy_score(actual, predicted),
            'Precision': precision_score(actual, predicted, zero_division=0),
            'Recall': recall_score(actual, predicted, zero_division=0),
            'F1_Score': f1_score(actual, predicted, zero_division=0),
            'ROC_AUC': roc_auc_score(actual, probability)}

def main():
    df = pd.read_excel(INPUT, sheet_name='Customer_Data')
    print('PART 1 | DATA OVERVIEW')
    print('First five observations:\n', df.head().to_string(index=False))
    print('\nRows, columns:', df.shape)
    print('Column names:', df.columns.tolist())
    print('Data types:\n', df.dtypes.to_string())
    print('Missing values:\n', df.isna().sum().to_string())
    for column in PREDICTORS:
        print(f'{column} categories:', sorted(df[column].dropna().unique().tolist()))
    print('Class counts:\n', df[TARGET].value_counts().sort_index().to_string())
    print('Class percentages:\n', (100 * df[TARGET].value_counts(normalize=True).sort_index()).round(2).to_string())
    if df[PREDICTORS + [TARGET, ID]].isna().any().any():
        raise ValueError('Input contains missing values; review before fitting the model.')
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError('Target must contain only 0 and 1.')

    # Only training records are used to estimate the class priors and manual probabilities.
    train_idx, test_idx = train_test_split(df.index, test_size=0.20,
                                           random_state=42, stratify=df[TARGET])
    training = df.loc[train_idx]
    testing = df.loc[test_idx]
    x_train, y_train = training[PREDICTORS], training[TARGET]
    x_test, y_test = testing[PREDICTORS], testing[TARGET]
    prior = y_train.value_counts(normalize=True).to_dict()
    majority = int(y_train.value_counts().idxmax())
    baseline_accuracy = float((y_test == majority).mean())
    prior_summary = pd.DataFrame([{'Class': klass,
                                   'Training_Count': int((y_train == klass).sum()),
                                   'Training_Prior_Probability': prior[klass],
                                   'Majority_Class': majority,
                                   'Majority_Class_Test_Accuracy': baseline_accuracy}
                                  for klass in [0, 1]])
    save_csv(prior_summary, 'class_prior_summary.csv')

    # This small demonstration is deliberately unsmoothed and uses only three variables.
    profile = [('Credit_Profile', 'Good'), ('Digital_Engagement', 'High'),
               ('Recent_Campaign_Response', 'Yes')]
    records = []
    scores = {}
    for klass in [0, 1]:
        group = training[training[TARGET] == klass]
        score = prior[klass]
        for field, value in profile:
            count = int((group[field] == value).sum())
            p = count / len(group)
            score *= p
            records.append({'Class': klass, 'Term': f'P({field}={value} | Offer_Accepted={klass})',
                            'Matching_Training_Rows': count, 'Class_Training_Rows': len(group),
                            'Probability': p})
        scores[klass] = score
    denominator = scores[0] + scores[1]
    for klass in [0, 1]:
        records += [{'Class': klass, 'Term': f'Prior P(Offer_Accepted={klass})',
                     'Matching_Training_Rows': int((y_train == klass).sum()),
                     'Class_Training_Rows': len(training), 'Probability': prior[klass]},
                    {'Class': klass, 'Term': 'Unnormalized NB score', 'Matching_Training_Rows': '',
                     'Class_Training_Rows': '', 'Probability': scores[klass]}]
    records.append({'Class': 1, 'Term': 'Normalized P(Offer_Accepted=1 | profile)',
                    'Matching_Training_Rows': '', 'Class_Training_Rows': '',
                    'Probability': scores[1] / denominator if denominator else np.nan})
    save_csv(pd.DataFrame(records), 'manual_bayes_calculation.csv')

    model = pipeline(1.0)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    pred = (prob >= 0.50).astype(int)
    predictions = pd.DataFrame({'Customer_ID': testing[ID].to_numpy(),
                                'Actual_Offer_Accepted': y_test.to_numpy(),
                                'Predicted_Probability': prob,
                                'Predicted_Class_0.50': pred})
    save_csv(predictions, 'naive_bayes_predictions.csv')
    result_metrics = metrics(y_test, pred, prob)
    save_csv(pd.DataFrame([{'Cutoff': 0.50, **result_metrics}]), 'classification_metrics.csv')

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    save_csv(pd.DataFrame([{'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}]), 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(7.1, 5.6))
    matrix = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Predicted 0', 'Predicted 1'])
    ax.set_yticks([0, 1], labels=['Actual 0', 'Actual 1'])
    ax.set_title('Offer Acceptance | Confusion Matrix (Cutoff 0.50)', pad=15)
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{labels[i][j]}\n{matrix[i,j]}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='white' if matrix[i,j] > matrix.max()/2 else '#13253b')
    fig.tight_layout()
    fig.savefig(HERE / 'confusion_matrix.png', dpi=180)
    plt.close(fig)

    cutoff_rows = []
    for cutoff in [0.30, 0.50, 0.70]:
        p = (prob >= cutoff).astype(int)
        m = metrics(y_test, p, prob)
        cutoff_rows.append({'Cutoff': cutoff, 'Accuracy': m['Accuracy'],
                            'Precision': m['Precision'], 'Recall': m['Recall'],
                            'F1_Score': m['F1_Score'], 'Predicted_Positive_Count': int(p.sum())})
    save_csv(pd.DataFrame(cutoff_rows), 'cutoff_comparison.csv')

    smoothing_rows = []
    for alpha in [0.10, 1.00, 5.00]:
        alternate = pipeline(alpha)
        alternate.fit(x_train, y_train)
        p = alternate.predict_proba(x_test)[:, list(alternate.classes_).index(1)]
        smoothing_rows.append({'Alpha': alpha, **metrics(y_test, (p >= 0.50).astype(int), p)})
    save_csv(pd.DataFrame(smoothing_rows), 'smoothing_comparison.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(prob, bins=np.linspace(0, 1, 21), color='#4472a0', edgecolor='white')
    ax.axvline(0.50, color='#ac493a', linestyle='--', linewidth=1.7, label='0.50 cutoff')
    ax.set(title='Predicted Offer Acceptance Probabilities | Test Set',
           xlabel='Predicted probability of acceptance', ylabel='Number of customers', xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / 'predicted_probability_distribution.png', dpi=180)
    plt.close(fig)

    scenarios = pd.DataFrame([
        ['A', '18-29', 'Low', 'Student', 'Fair', 'Low', '0-1', 'Email', 'No'],
        ['B', '30-44', 'Upper-Middle', 'Salaried', 'Good', 'High', '2-3', 'Mobile', 'Yes'],
        ['C', '45-59', 'High', 'Self-Employed', 'Excellent', 'High', '4+', 'Web', 'Yes'],
    ], columns=['Scenario'] + PREDICTORS)
    scenario_p = model.predict_proba(scenarios[PREDICTORS])[:, list(model.classes_).index(1)]
    scenarios['Predicted_Probability'] = scenario_p
    scenarios['Predicted_Class_0.50'] = (scenario_p >= 0.5).astype(int)
    save_csv(scenarios, 'business_probability_predictions.csv')
    print('\nPART 2 | SPLIT: training', len(training), 'testing', len(testing))
    print('PART 3 | PRIORS:\n', prior_summary.to_string(index=False))
    print('PART 4 | MANUAL PROFILE P(ACCEPTED=1):', scores[1] / denominator)
    print('PART 8 | METRICS:', result_metrics)
    print('PART 9 | CUTOFFS:\n', pd.DataFrame(cutoff_rows).to_string(index=False))
    print('PART 10 | SMOOTHING:\n', pd.DataFrame(smoothing_rows).to_string(index=False))
    print('PART 12 | SCENARIOS:\n', scenarios[['Scenario', 'Predicted_Probability', 'Predicted_Class_0.50']].to_string(index=False))
    print('Saved result files to:', HERE)

if __name__ == '__main__':
    main()
