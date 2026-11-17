"""Homework 10: Customer segmentation with hierarchical clustering and K-Means.

Run: python Homework10_Keishana_Morsley.py
Requirements: pandas, openpyxl, numpy, scipy, scikit-learn, matplotlib
The Excel input should be placed in the same directory as this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE = Path(__file__).resolve().parent
DATA = BASE / 'HW10_Dataset.xlsx'
FEATURES = [
    'Age', 'Annual_Income_K', 'Monthly_Spend',
    'Digital_Engagement_Score', 'Store_Visits_Month',
    'Loyalty_Years', 'Discount_Use_Pct'
]
PROFILE_VARIABLES = FEATURES + ['Campaign_Response_Pct']


def write_csv(frame, name):
    frame.to_csv(BASE / name, index=False, float_format='%.8f')


def main():
    # Part 1: The problem is unsupervised because no true segment label exists.
    df = pd.read_excel(DATA, sheet_name='Customer_Data')
    print('PART 1: DATA OVERVIEW')
    print('First five rows:\n', df.head().to_string(index=False))
    print(f'\nShape: {df.shape[0]} rows, {df.shape[1]} columns')
    print('Columns and data types:\n', df.dtypes.to_string())
    print('Missing values:\n', df.isna().sum().to_string())
    print('Numerical descriptive statistics:\n', df.describe().to_string())
    print('This is unsupervised learning: the input contains no ground-truth cluster labels.\n')
    if df['Customer_ID'].duplicated().any():
        raise ValueError('Customer_ID values must be unique.')
    if df[FEATURES].isna().any().any():
        raise ValueError('Missing values in the seven clustering features need to be resolved.')

    # Parts 2-3: Fit StandardScaler on exactly the seven specified features.
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[FEATURES])
    scaled = pd.DataFrame(scaled_values, columns=FEATURES, index=df.index)
    standardized_export = pd.concat([df[['Customer_ID']], scaled], axis=1)
    write_csv(standardized_export, 'standardized_customer_data.csv')
    print('PART 3: STANDARDIZATION CHECK (population standard deviations)')
    print(pd.DataFrame({'mean': scaled.mean(), 'std_ddof0': scaled.std(ddof=0)}).to_string())

    # Part 4: Same two customers and same seven features, before/after scaling.
    a = df.loc[df.Customer_ID.eq('CUST0001')].iloc[0]
    b = df.loc[df.Customer_ID.eq('CUST0002')].iloc[0]
    ia, ib = a.name, b.name
    raw_distance = euclidean(df.loc[ia, FEATURES].to_numpy(dtype=float),
                             df.loc[ib, FEATURES].to_numpy(dtype=float))
    scaled_distance = euclidean(scaled.loc[ia].to_numpy(), scaled.loc[ib].to_numpy())
    write_csv(pd.DataFrame([
        {'Customer_1': 'CUST0001', 'Customer_2': 'CUST0002',
         'Scale': 'Raw', 'Euclidean_Distance': raw_distance},
        {'Customer_1': 'CUST0001', 'Customer_2': 'CUST0002',
         'Scale': 'Standardized', 'Euclidean_Distance': scaled_distance}
    ]), 'distance_comparison.csv')
    print(f'\nPART 4: Raw Euclidean distance = {raw_distance:.4f}; standardized = {scaled_distance:.4f}')

    # Part 5: Reproducible sample and average-linkage Euclidean dendrogram.
    sampled = df.sample(n=60, random_state=42)
    sample_scaled = scaled.loc[sampled.index]
    Z = linkage(sample_scaled.to_numpy(), method='average', metric='euclidean')
    # Strictly between the merge producing 4 groups and the next merge producing 3.
    cut_height = float((Z[-4, 2] + Z[-3, 2]) / 2)
    h_labels = fcluster(Z, t=cut_height, criterion='distance')
    assert len(set(h_labels)) == 4
    hierarchy = pd.DataFrame({'Customer_ID': sampled['Customer_ID'].to_numpy(),
                              'Hierarchical_Cluster': h_labels})
    write_csv(hierarchy, 'hierarchical_cluster_assignments.csv')
    plt.figure(figsize=(19, 8))
    dendrogram(Z, labels=sampled['Customer_ID'].tolist(), leaf_rotation=90,
               leaf_font_size=7, color_threshold=cut_height)
    plt.axhline(cut_height, color='black', linestyle='--', linewidth=1.4,
                label=f'4-cluster cut at height {cut_height:.3f}')
    plt.title('Average-Linkage Hierarchical Clustering: 60 Customers')
    plt.xlabel('Customer ID')
    plt.ylabel('Euclidean distance between clusters (standardized features)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(BASE / 'hierarchical_dendrogram.png', dpi=200)
    plt.close()
    print(f'\nPART 5: Dendrogram 4-cluster cut height = {cut_height:.4f}')
    print('Merges at lower heights indicate closer customers/clusters.')

    # Parts 6-7: Evaluate candidates and fit the required final k=4.
    comparison = []
    models = {}
    for k in (2, 3, 4, 5, 6):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_values)
        comparison.append({'k': k, 'Inertia': model.inertia_,
                           'Silhouette_Score': silhouette_score(scaled_values, labels)})
        models[k] = model
    metric_frame = pd.DataFrame(comparison)
    write_csv(metric_frame, 'kmeans_k_comparison.csv')
    print('\nPART 6: K-MEANS COMPARISON\n', metric_frame.to_string(index=False))
    plt.figure(figsize=(8, 5))
    plt.plot(metric_frame['k'], metric_frame['Inertia'], marker='o', linewidth=2)
    plt.axvline(4, color='gray', linestyle='--', label='Assigned final k = 4')
    plt.xticks([2, 3, 4, 5, 6])
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster sum of squares (inertia)')
    plt.title('K-Means Elbow Plot')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(BASE / 'kmeans_elbow_plot.png', dpi=200)
    plt.close()

    final_model = models[4]  # Equivalent to KMeans(n_clusters=4, random_state=42, n_init=20).
    df['Cluster'] = final_model.labels_
    write_csv(df[['Customer_ID', 'Cluster']], 'kmeans_cluster_assignments.csv')
    centers = pd.DataFrame(final_model.cluster_centers_, columns=FEATURES)
    centers.insert(0, 'Cluster', np.arange(4))
    write_csv(centers, 'kmeans_centroids_standardized.csv')
    print('\nPART 7: CUSTOMER COUNTS\n', df.groupby('Cluster').size().to_string())

    # Part 8: Original-unit cluster profiles; campaign response is external validation only.
    profiles = df.groupby('Cluster')[PROFILE_VARIABLES].mean().reset_index()
    profiles.insert(1, 'Customer_Count', df.groupby('Cluster').size().to_numpy())
    write_csv(profiles, 'cluster_profile.csv')
    print('\nPART 8: ORIGINAL-UNIT CLUSTER PROFILES\n', profiles.round(2).to_string(index=False))
    business_labels = {
        0: 'Low-Engagement Light Spenders',
        1: 'Promotion-Responsive Frequent Shoppers',
        2: 'Established In-Store Loyalists',
        3: 'High-Value Digitally Engaged Customers',
    }
    profiles_with_labels = profiles.copy()
    profiles_with_labels.insert(1, 'Business_Label', profiles_with_labels['Cluster'].map(business_labels))
    write_csv(profiles_with_labels, 'cluster_profile_with_labels.csv')

    # Part 9: A 2D illustration of the seven-feature clustering result.
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in sorted(df['Cluster'].unique()):
        portion = df.loc[df.Cluster == cluster]
        ax.scatter(portion['Annual_Income_K'], portion['Monthly_Spend'],
                   s=33, alpha=0.68, label=f'Cluster {cluster} (n={len(portion)})')
    ax.set(xlabel='Annual household income ($ thousands)',
           ylabel='Average monthly spend ($)',
           title='Customer Segments: Original Income vs. Original Spend')
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / 'kmeans_customer_segments.png', dpi=200)
    plt.close(fig)

    # Part 10: Labels are arbitrary; ARI compares partitions without matching label IDs.
    matching = hierarchy.merge(df[['Customer_ID', 'Cluster']], on='Customer_ID', validate='one_to_one')
    matching = matching.rename(columns={'Cluster': 'KMeans_Cluster'})
    write_csv(matching, 'hierarchical_vs_kmeans.csv')
    ari = adjusted_rand_score(matching.Hierarchical_Cluster, matching.KMeans_Cluster)
    print(f'\nPART 10: Adjusted Rand Index for the 60-customer sample = {ari:.4f}')
    print('Numeric labels from different algorithms are not directly comparable.')

    # Extra console summary and persisted text file for Part 1 and interpretation.
    sil = metric_frame.set_index('k')['Silhouette_Score']
    with (BASE / 'Homework10_Results_Summary.txt').open('w', encoding='utf-8') as f:
        f.write('HOMEWORK 10: CUSTOMER SEGMENTATION\n\n')
        f.write('PART 1: LOAD AND UNDERSTAND\n')
        f.write(df.drop(columns='Cluster').head().to_string(index=False) + '\n\n')
        f.write(f'Rows: {len(df)}; columns: {df.shape[1]-1}\n')
        f.write('Columns and types:\n' + df.drop(columns='Cluster').dtypes.to_string() + '\n\n')
        f.write('Missing values:\n' + df.drop(columns='Cluster').isna().sum().to_string() + '\n\n')
        f.write('Numeric descriptive statistics:\n' + df.drop(columns='Cluster').describe().to_string() + '\n\n')
        f.write('Why unsupervised: No target variable or established customer segment labels are provided.\n\n')
        f.write('PART 3: SCALING CHECK\n' +
                pd.DataFrame({'mean': scaled.mean(), 'std_population': scaled.std(ddof=0)}).to_string() + '\n\n')
        f.write(f'PART 4: EUCLIDEAN DISTANCES\nRaw: {raw_distance:.6f}\nStandardized: {scaled_distance:.6f}\n')
        f.write('Raw distance is affected by differing units and ranges; standardized distance gives each feature comparable scale.\n\n')
        f.write(f'PART 5: HIERARCHICAL CLUSTERING\nAverage linkage; Euclidean; n=60 sampled with random_state=42.\n')
        f.write(f'Cut height: {cut_height:.6f}; groups: 4. Lower dendrogram merges reflect greater similarity.\n\n')
        f.write('PART 6: K SELECTION\n' + metric_frame.round(4).to_string(index=False) + '\n')
        f.write(f'Highest silhouette among tested k: {sil.idxmax()} ({sil.max():.4f}).\n')
        f.write('The task specifies final k=4 for consistent comparison; elbow and profile interpretability were also examined.\n\n')
        f.write('PARTS 7-8: FINAL K=4 AND ORIGINAL-UNIT PROFILES\n' + profiles.round(2).to_string(index=False) + '\n\n')
        f.write('INTERPRETATION OF THE FOUR SEGMENTS\n')
        for c in range(4):
            r = profiles.loc[profiles.Cluster.eq(c)].iloc[0]
            f.write(f'Cluster {c}: {business_labels[c]} (n={int(r.Customer_Count)}). '
                    f'Monthly spend ${r.Monthly_Spend:.2f}; digital score {r.Digital_Engagement_Score:.2f}; '
                    f'visits/month {r.Store_Visits_Month:.2f}; loyalty years {r.Loyalty_Years:.2f}; '
                    f'discount use {r.Discount_Use_Pct:.2f}%; campaign response {r.Campaign_Response_Pct:.2f}%.\n')
        f.write('Campaign response was assessed after fitting and never used as a clustering feature.\n\n')
        f.write('PART 9: VISUALIZATION\nIncome vs. spend is two-dimensional; the actual K-Means model uses all seven standardized variables.\n\n')
        f.write(f'PART 10: METHOD COMPARISON\nAdjusted Rand Index (sample): {ari:.4f}.\n')
        f.write('ARI assesses agreement without assuming cluster numbers correspond. The sample-only hierarchical\n')
        f.write('partition and full-data K-Means partition need not match exactly.\n')
    print('\nFiles written to', BASE)

if __name__ == '__main__':
    main()
