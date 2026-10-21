"""Homework 7: Classification and Regression Trees (scikit-learn).

Place this script beside Homework7_Classification_Regression_Trees_Data.xlsx and run:
    python Homework7_Keishana_Morsley_UTAID.py

Requires: pandas, numpy, matplotlib, scikit-learn, openpyxl.
Outputs are saved to the script's directory.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save charts even without a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_squared_error,
)

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'Homework7_Classification_Regression_Trees_Data.xlsx'
SEED = 42
FEATURES = [
    'Age', 'Annual_Income_K', 'Credit_Score', 'Family_Size',
    'Education_Level', 'Account_Tenure_Years',
    'Avg_Monthly_Card_Spend_K', 'Digital_Engagement_Score', 'Product_Count',
]
DEPTHS = [3, 5, 7, None]


def csv(name, data):
    data.to_csv(BASE / name, index=False, float_format='%.6f')
    print(f'Saved: {name}')


def scores(y_true, y_pred):
    return {
        'Testing_Accuracy': accuracy_score(y_true, y_pred),
        'Testing_Precision': precision_score(y_true, y_pred, zero_division=0),
        'Testing_Recall': recall_score(y_true, y_pred, zero_division=0),
        'Testing_F1': f1_score(y_true, y_pred, zero_division=0),
    }


def depth_label(depth):
    return 'None (Fully Grown)' if depth is None else str(depth)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(BASE / filename, dpi=190, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


def choose_practical(rows, metric, tolerance):
    """Within tolerance of best test score, choose smallest tree by nodes/leaves.

    This is an explicit performance-versus-complexity heuristic, not a claim
    that test-set selection provides an unbiased final generalization estimate.
    """
    best_score = max(r[metric] for r in rows)
    eligible = [r for r in rows if r[metric] >= best_score - tolerance]
    return min(eligible, key=lambda r: (r['Nodes'], r['Leaves']))


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing input workbook: {INPUT}\nPut the Excel workbook beside this .py file.')

    # PART 1: Understand the supplied customer dataset.
    df = pd.read_excel(INPUT, sheet_name='Customer_Data', engine='openpyxl')
    expected = ['Customer_ID', *FEATURES, 'Loan_Accepted', 'Annual_Bank_Revenue_K']
    missing_columns = set(expected).difference(df.columns)
    if missing_columns:
        raise ValueError(f'Required columns missing from workbook: {sorted(missing_columns)}')
    print('\n=== PART 1: DATA EXPLORATION ===')
    print('First five observations:\n', df.head().to_string(index=False))
    print(f'\nRows: {df.shape[0]} | Columns: {df.shape[1]}')
    print('\nColumn names and types:\n', df.dtypes.to_string())
    print('\nMissing values:\n', df.isna().sum().to_string())
    counts = df['Loan_Accepted'].value_counts().reindex([0, 1], fill_value=0)
    class_summary = pd.DataFrame({'Count': counts, 'Percent': counts / len(df) * 100})
    class_summary.index.name = 'Loan_Accepted'
    print('\nLoan acceptance distribution:\n', class_summary.to_string())
    print('\nAnnual bank revenue (thousands of dollars):\n', df['Annual_Bank_Revenue_K'].describe().to_string())
    if df[expected].isna().any().any():
        raise ValueError('Missing data detected: inspect Part 1 before fitting trees.')
    if df['Customer_ID'].duplicated().any():
        raise ValueError('Duplicate Customer_ID values detected.')

    # PART 2: Same nine features for both tasks; never include ID or targets in X.
    X = df[FEATURES].copy()
    y_class = df['Loan_Accepted'].astype(int)
    y_reg = df['Annual_Bank_Revenue_K'].astype(float)

    # PART 3: Stratified 80/20 classification split with stable original indices.
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=SEED, stratify=y_class)
    print(f'\n=== PART 3: CLASSIFICATION SPLIT ===\nTrain: {len(Xc_train)} | Test: {len(Xc_test)}')

    # PART 4: Unrestricted baseline Gini classification tree.
    baseline = DecisionTreeClassifier(criterion='gini', random_state=SEED)
    baseline.fit(Xc_train, yc_train)
    train_prediction = baseline.predict(Xc_train)
    test_prediction = baseline.predict(Xc_test)
    baseline_metrics = {'Training_Accuracy': accuracy_score(yc_train, train_prediction),
                        **scores(yc_test, test_prediction)}
    print('\n=== PART 4: BASELINE CLASSIFICATION ===')
    for key, value in baseline_metrics.items():
        print(f'{key}: {value:.4f}')
    print('Confusion matrix (rows actual 0/1, columns predicted 0/1):\n',
          confusion_matrix(yc_test, test_prediction, labels=[0, 1]))
    csv('classification_predictions.csv', pd.DataFrame({
        'Customer_ID': df.loc[Xc_test.index, 'Customer_ID'].to_numpy(),
        'Actual_Loan_Accepted': yc_test.to_numpy(),
        'Predicted_Loan_Accepted': test_prediction,
    }))
    csv('classification_metrics.csv', pd.DataFrame([baseline_metrics]))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yc_test, test_prediction, labels=[0, 1],
                                            cmap='Blues', values_format='d', ax=ax)
    ax.set_title('Baseline Classification Tree — Test Set Confusion Matrix')
    save_figure('confusion_matrix.png')

    # PART 5: Hold max_depth and random_state constant; vary Gini versus Entropy.
    comparison_rows = []
    for criterion in ('gini', 'entropy'):
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=SEED)
        tree.fit(Xc_train, yc_train)
        comparison_rows.append({'Criterion': criterion, 'Max_Depth': 5,
                                **scores(yc_test, tree.predict(Xc_test))})
    csv('criterion_comparison.csv', pd.DataFrame(comparison_rows))
    print('\n=== PART 5: GINI VS ENTROPY ===\n', pd.DataFrame(comparison_rows).to_string(index=False))

    # PART 6: Depth, training/testing accuracy, nodes/leaves, and pre-pruning.
    classification_models, depth_rows = {}, []
    for depth in DEPTHS:
        tree = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=SEED)
        tree.fit(Xc_train, yc_train)
        classification_models[depth_label(depth)] = tree
        depth_rows.append({
            'Max_Depth': depth_label(depth),
            'Training_Accuracy': accuracy_score(yc_train, tree.predict(Xc_train)),
            'Testing_Accuracy': accuracy_score(yc_test, tree.predict(Xc_test)),
            'Nodes': tree.tree_.node_count, 'Leaves': tree.get_n_leaves(),
        })
    csv('depth_comparison.csv', pd.DataFrame(depth_rows))
    picked_class = choose_practical(depth_rows, 'Testing_Accuracy', tolerance=0.01)
    selected_class = classification_models[picked_class['Max_Depth']]
    print('\n=== PART 6: CLASSIFICATION DEPTHS ===\n', pd.DataFrame(depth_rows).to_string(index=False))
    print('Selected practical classification tree:', picked_class)
    positions = list(range(len(depth_rows)))
    labels = [r['Max_Depth'] for r in depth_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(positions, [r['Training_Accuracy'] for r in depth_rows], 'o-', label='Training accuracy')
    ax.plot(positions, [r['Testing_Accuracy'] for r in depth_rows], 's-', label='Testing accuracy')
    ax.set_xticks(positions, labels)
    ax.set_xlabel('Maximum depth (None = fully grown)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Classification Tree Accuracy vs. Maximum Depth')
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure('depth_accuracy_plot.png')

    # PART 7: Readable chosen tree with splits, Gini, samples, values and class.
    fig, ax = plt.subplots(figsize=(max(16, min(38, 3.0 * selected_class.get_n_leaves())),
                                    max(9, selected_class.get_depth() * 2.2)))
    plot_tree(selected_class, feature_names=FEATURES, class_names=['No Loan', 'Accepted'],
              filled=True, rounded=True, impurity=True, proportion=False, fontsize=9, ax=ax)
    ax.set_title(f"Selected Classification Tree (max_depth={picked_class['Max_Depth']})", fontsize=15)
    save_figure('classification_tree.png')
    rules = export_text(selected_class, feature_names=FEATURES, decimals=3, show_weights=True)
    (BASE / 'tree_rules.txt').write_text(
        'Selected classification tree rules\n'
        f"Maximum depth: {picked_class['Max_Depth']}\n"
        'Leaf class is the class with the highest class count.\n\n' + rules,
        encoding='utf-8')
    print('Saved: tree_rules.txt')

    # PART 8: Importance is relative impurity reduction, not a causal effect.
    importance = pd.DataFrame({'Feature': FEATURES, 'Importance': selected_class.feature_importances_})
    importance = importance.sort_values('Importance', ascending=False).reset_index(drop=True)
    csv('feature_importance.csv', importance)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    ax.barh(importance['Feature'][::-1], importance['Importance'][::-1])
    ax.set_xlabel('Relative impurity-based importance')
    ax.set_title('Selected Classification Tree — Feature Importance')
    save_figure('feature_importance.png')

    # PART 9: New independent non-stratified regression 80/20 split.
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=SEED)
    regression_models, regression_rows = {}, []
    for depth in DEPTHS:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=SEED)
        tree.fit(Xr_train, yr_train)
        predictions = tree.predict(Xr_test)
        regression_models[depth_label(depth)] = tree
        regression_rows.append({
            'Max_Depth': depth_label(depth),
            'Training_R2': r2_score(yr_train, tree.predict(Xr_train)),
            'Testing_R2': r2_score(yr_test, predictions),
            'Testing_RMSE': float(np.sqrt(mean_squared_error(yr_test, predictions))),
            'Nodes': tree.tree_.node_count, 'Leaves': tree.get_n_leaves(),
        })
    csv('regression_tree_comparison.csv', pd.DataFrame(regression_rows))
    picked_reg = choose_practical(regression_rows, 'Testing_R2', tolerance=0.02)
    selected_reg = regression_models[picked_reg['Max_Depth']]
    regression_prediction = selected_reg.predict(Xr_test)
    print('\n=== PART 9: REGRESSION DEPTHS ===\n', pd.DataFrame(regression_rows).to_string(index=False))
    print('Selected practical regression tree:', picked_reg)
    csv('regression_predictions.csv', pd.DataFrame({
        'Customer_ID': df.loc[Xr_test.index, 'Customer_ID'].to_numpy(),
        'Actual_Annual_Bank_Revenue_K': yr_test.to_numpy(),
        'Predicted_Annual_Bank_Revenue_K': regression_prediction,
        'Residual': yr_test.to_numpy() - regression_prediction,
    }))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(yr_test, regression_prediction, alpha=0.7, edgecolors='none')
    bottom = min(float(yr_test.min()), float(regression_prediction.min()))
    top = max(float(yr_test.max()), float(regression_prediction.max()))
    ax.plot([bottom, top], [bottom, top], '--', color='black', linewidth=1.3, label='Perfect prediction')
    ax.set(xlabel='Actual Annual Bank Revenue ($ thousands)',
           ylabel='Predicted Annual Bank Revenue ($ thousands)',
           title='Selected Regression Tree — Actual vs. Predicted')
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure('regression_actual_vs_predicted.png')

    # PART 10: Business predictions (column order matches training X).
    scenarios = pd.DataFrame([
        ['A', 29, 48, 625, 2, 2, 1.5, 1.10, 35, 1],
        ['B', 45, 95, 725, 3, 2, 8.0, 3.20, 72, 4],
        ['C', 58, 145, 780, 4, 3, 15.0, 5.50, 88, 5],
    ], columns=['Scenario', *FEATURES])
    scenarios['Predicted_Loan_Accepted'] = selected_class.predict(scenarios[FEATURES]).astype(int)
    scenarios['Predicted_Annual_Bank_Revenue_K'] = selected_reg.predict(scenarios[FEATURES])
    csv('business_tree_predictions.csv', scenarios)
    print('\n=== PART 10: BUSINESS SCENARIOS ===\n', scenarios.to_string(index=False))
    print('\nAll required machine-generated artifacts completed. Review the separate Copilot reflection.')


if __name__ == '__main__':
    main()
