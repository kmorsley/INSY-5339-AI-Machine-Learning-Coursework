"""INSY 5339 | Homework 8 | Keishana Morsley | UTA ID 1001807020.

"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

HERE = Path(__file__).resolve().parent
DATA = HERE / 'Homework_8_Dataset.xlsx'
if not DATA.exists():
    DATA = HERE / 'Homework8_KNN_Association_Rules_Data.xlsx'
if not DATA.exists():
    raise FileNotFoundError('Place the provided dataset workbook alongside this script.')

PREDICTORS = ['Age','Annual_Income_K','Monthly_Spend','Digital_Engagement_Score',
              'Account_Tenure_Years','Website_Visits_Month']
TARGET = 'Premium_Service_Adopted'


def item_string(items):
    return ', '.join(sorted(items))


def fallback_apriori(matrix, min_support=0.05):
    """Apriori support pruning: retain itemsets whose support >= threshold."""
    transactions = [frozenset(matrix.columns[row]) for row in matrix.to_numpy(dtype=bool)]
    n = len(transactions)
    threshold = min_support * n
    singles = sorted(matrix.columns)
    levels = [frozenset((item,)) for item in singles]
    found = []
    k = 1
    while levels:
        counted = [(set_, sum(set_.issubset(tx) for tx in transactions)) for set_ in levels]
        valid = sorted([(s, c) for s,c in counted if c + 1e-12 >= threshold],
                       key=lambda pair: tuple(sorted(pair[0])))
        found.extend([(c/n, s) for s,c in valid])
        previous = {s for s,_ in valid}
        k += 1
        if len(previous) < k:
            break
        candidate_set = set()
        for a,b in combinations(sorted(previous, key=lambda x: tuple(sorted(x))),2):
            union = a | b
            if len(union) == k and all(frozenset(part) in previous for part in combinations(sorted(union), k-1)):
                candidate_set.add(union)
        levels = sorted(candidate_set, key=lambda x: tuple(sorted(x)))
    return pd.DataFrame(found, columns=['support','itemsets'])


def fallback_rules(itemsets, min_confidence=0.40):
    """Association measures use the same itemset-support identities as mlxtend."""
    supports = {frozenset(row.itemsets): float(row.support) for row in itemsets.itertuples()}
    records = []
    for itemset, joint in supports.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for left_tuple in combinations(sorted(itemset),size):
                left = frozenset(left_tuple)
                right = itemset-left
                conf = joint/supports[left]
                if conf + 1e-12 >= min_confidence:
                    records.append({'antecedents':left,'consequents':right,'support':joint,
                                    'confidence':conf,'lift':conf/supports[right]})
    return pd.DataFrame(records, columns=['antecedents','consequents','support','confidence','lift'])


def main():
    customers = pd.read_excel(DATA, sheet_name='KNN_Customer_Data')
    print('PART A: CUSTOMER CLASSIFICATION')
    print('First five customers:\n',customers.head().to_string(index=False))
    print('Shape:',customers.shape,'\nColumn types:\n',customers.dtypes.to_string())
    print('Missing values:\n',customers.isna().sum().to_string())
    counts = customers[TARGET].value_counts().sort_index()
    print('Class counts and percentages:\n',pd.DataFrame({'Count':counts,'Percent':100*counts/len(customers)}))
    if customers[PREDICTORS+[TARGET]].isna().any().any():
        raise ValueError('Missing modeling values must be addressed before analysis.')
    X = customers[PREDICTORS]
    y = customers[TARGET]
    ids = customers['Customer_ID']
    # 60/20/20 overall, stratified in both stages to preserve class composition.
    X_train,X_temp,y_train,y_temp,id_train,id_temp = train_test_split(
        X,y,ids,test_size=0.40,random_state=42,stratify=y)
    X_val,X_test,y_val,y_test,id_val,id_test = train_test_split(
        X_temp,y_temp,id_temp,test_size=0.50,random_state=42,stratify=y_temp)
    print('Split sizes:',len(X_train),len(X_val),len(X_test))
    scaler = StandardScaler().fit(X_train)  # Fit ONLY training rows.
    scaled_train = scaler.transform(X_train)
    scaled_val = scaler.transform(X_val)
    scaled_test = scaler.transform(X_test)

    raw_model = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
    scaled_model_5 = KNeighborsClassifier(n_neighbors=5).fit(scaled_train,y_train)
    scale_result = pd.DataFrame([
        {'Model':'Unscaled k-NN (k=5)','Validation_Accuracy':accuracy_score(y_val,raw_model.predict(X_val))},
        {'Model':'Standardized k-NN (k=5)','Validation_Accuracy':accuracy_score(y_val,scaled_model_5.predict(scaled_val))}])
    scale_result.to_csv(HERE/'knn_scaling_comparison.csv',index=False)

    k_rows = []
    for k in [1,3,5,7,9,11,13,15]:
        model = KNeighborsClassifier(n_neighbors=k).fit(scaled_train,y_train)
        acc = accuracy_score(y_val,model.predict(scaled_val))
        k_rows.append({'k':k,'Validation_Accuracy':acc,'Validation_Error_Rate':1-acc})
    k_table = pd.DataFrame(k_rows)
    k_table.to_csv(HERE/'knn_k_comparison.csv',index=False)
    best_k = int(k_table.sort_values(['Validation_Accuracy','k'],ascending=[False,True]).iloc[0]['k'])
    print('Validation selected k:',best_k)
    plt.figure(figsize=(8.4,5.2))
    plt.plot(k_table['k'],k_table['Validation_Accuracy'],marker='o',linewidth=2)
    plt.xticks(k_table['k'])
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.title('INSY 5339 Homework 8: k-NN Validation Accuracy')
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(HERE/'knn_validation_accuracy.png',dpi=180)
    plt.close()

    # Preserve the original training-only model and never tune on test labels.
    final = KNeighborsClassifier(n_neighbors=best_k).fit(scaled_train,y_train)
    predictions = final.predict(scaled_test)
    metrics = pd.DataFrame([{'Selected_k':best_k,
                            'Accuracy':accuracy_score(y_test,predictions),
                            'Precision':precision_score(y_test,predictions,zero_division=0),
                            'Recall':recall_score(y_test,predictions,zero_division=0),
                            'F1_Score':f1_score(y_test,predictions,zero_division=0)}])
    metrics.to_csv(HERE/'knn_test_metrics.csv',index=False)
    pd.DataFrame({'Customer_ID':id_test.to_numpy(),'Actual_Class':y_test.to_numpy(),
                  'Predicted_Class':predictions}).to_csv(HERE/'knn_predictions.csv',index=False)
    confusion = confusion_matrix(y_test,predictions,labels=[0,1])
    cm_frame = pd.DataFrame(confusion,index=['Actual_0','Actual_1'],columns=['Predicted_0','Predicted_1'])
    cm_frame.to_csv(HERE/'knn_confusion_matrix.csv',index_label='Actual_Class')
    fig,ax = plt.subplots(figsize=(6.3,5.4))
    img=ax.imshow(confusion,cmap='Blues')
    for (i,j),value in np.ndenumerate(confusion):
        ax.text(j,i,str(value),ha='center',va='center',fontsize=17,
                color='white' if value > confusion.max()/2 else 'black')
    ax.set_xticks([0,1],labels=['Not Adopted (0)','Adopted (1)'])
    ax.set_yticks([0,1],labels=['Not Adopted (0)','Adopted (1)'])
    ax.set_xlabel('Predicted Class');ax.set_ylabel('Actual Class')
    ax.set_title('INSY 5339 Homework 8: Test Confusion Matrix')
    fig.colorbar(img,ax=ax,shrink=.8)
    fig.tight_layout();fig.savefig(HERE/'knn_confusion_matrix.png',dpi=180);plt.close(fig)

    new = pd.DataFrame([[28,48,130,38,1.5,7],[41,92,310,78,8,18],
                        [58,125,260,52,15,10]],columns=PREDICTORS)
    new['Scenario']=['A','B','C']
    new_scaled = scaler.transform(new[PREDICTORS])
    new['Predicted_Class'] = final.predict(new_scaled)
    new[['Scenario']+PREDICTORS+['Predicted_Class']].to_csv(
        HERE/'knn_new_customer_predictions.csv',index=False)
    distances,neighbor_positions=final.kneighbors(new_scaled[1:2],n_neighbors=best_k)
    neighbors=pd.DataFrame({'Customer_ID':id_train.iloc[neighbor_positions[0]].to_numpy(),
                            'Distance':distances[0],
                            'Class':y_train.iloc[neighbor_positions[0]].to_numpy()})
    neighbors.to_csv(HERE/'nearest_neighbors_scenario_B.csv',index=False)

    print('\nPART B: MARKET BASKET ANALYSIS')
    baskets = pd.read_excel(DATA,sheet_name='Market_Basket_Transactions')
    print('First five transactions:\n',baskets.head().to_string(index=False))
    print('Unique transactions:',baskets['Transaction_ID'].nunique())
    if baskets['Transaction_ID'].duplicated().any():
        raise ValueError('Repeated Transaction_ID values need resolution before encoding.')
    transactions = [[item.strip() for item in value.split(',') if item.strip()]
                    for value in baskets['Items'].astype(str)]
    if HAS_MLXTEND:
        encoder = TransactionEncoder()
        encoded=encoder.fit(transactions).transform(transactions)
        matrix=pd.DataFrame(encoded,columns=encoder.columns_)
    else:
        universe=sorted({item for transaction in transactions for item in transaction})
        matrix=pd.DataFrame([{item:item in transaction for item in universe} for transaction in transactions])
    matrix.insert(0,'Transaction_ID',baskets['Transaction_ID'].to_numpy())
    matrix.to_csv(HERE/'basket_binary_matrix.csv',index=False)
    binary=matrix.drop(columns=['Transaction_ID']).astype(bool)
    itemsets=apriori(binary,min_support=.05,use_colnames=True) if HAS_MLXTEND else fallback_apriori(binary,.05)
    itemsets=itemsets.sort_values('support',ascending=False,kind='stable').reset_index(drop=True)
    pd.DataFrame({'Itemsets':itemsets['itemsets'].map(item_string),
                  'Support':itemsets['support']}).to_csv(HERE/'frequent_itemsets.csv',index=False)
    if HAS_MLXTEND:
        # num_itemsets is required by recent mlxtend versions; older versions may not accept it.
        try:
            rules=association_rules(itemsets,metric='confidence',min_threshold=.40,num_itemsets=len(binary))
        except TypeError:
            rules=association_rules(itemsets,metric='confidence',min_threshold=.40)
    else:
        rules=fallback_rules(itemsets,.40)
    rules=rules.loc[rules['lift']>1.00].sort_values('lift',ascending=False,kind='stable').reset_index(drop=True)
    rules_out=pd.DataFrame({'Antecedents':rules['antecedents'].map(item_string),
                            'Consequents':rules['consequents'].map(item_string),
                            'Support':rules['support'],
                            'Confidence':rules['confidence'],
                            'Lift':rules['lift']})
    rules_out.to_csv(HERE/'association_rules.csv',index=False)
    if rules.empty:
        raise ValueError('No rules satisfy confidence >= 0.40 and lift > 1.00.')
    top=rules_out.head(10).iloc[::-1].copy()
    labels=top['Antecedents']+' -> '+top['Consequents']
    fig,ax=plt.subplots(figsize=(11,6.5))
    ax.barh(labels,top['Lift'],color='#27658c')
    ax.set_xlabel('Lift');ax.set_ylabel('Association Rule')
    ax.set_title('INSY 5339 Homework 8: Top 10 Association Rules by Lift')
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'top_10_rules_by_lift.png',dpi=180);plt.close(fig)

    strongest=rules.iloc[0]
    a=set(strongest['antecedents']);b=set(strongest['consequents'])
    n=len(transactions)
    antecedent_count=sum(a.issubset(set(t)) for t in transactions)
    consequent_count=sum(b.issubset(set(t)) for t in transactions)
    joint_count=sum((a|b).issubset(set(t)) for t in transactions)
    manual_support=joint_count/n
    manual_confidence=joint_count/antecedent_count
    manual_lift=manual_confidence/(consequent_count/n)
    check=pd.DataFrame([{
        'Rule':item_string(a)+' -> '+item_string(b),
        'Total_Transactions':n,'Antecedent_Count':antecedent_count,
        'Consequent_Count':consequent_count,'Joint_Count':joint_count,
        'Package_Support':strongest['support'],'Manual_Support':manual_support,
        'Package_Confidence':strongest['confidence'],'Manual_Confidence':manual_confidence,
        'Package_Lift':strongest['lift'],'Manual_Lift':manual_lift}])
    check.to_csv(HERE/'association_rule_manual_check.csv',index=False)
    assert np.allclose([manual_support,manual_confidence,manual_lift],
                       [strongest['support'],strongest['confidence'],strongest['lift']])
    print('Association rules:',len(rules_out),'| Frequent itemsets:',len(itemsets))
    print('Manual verification:\n',check.to_string(index=False))
    print('Analysis complete. Output directory:',HERE)

if __name__=='__main__':
    main()
