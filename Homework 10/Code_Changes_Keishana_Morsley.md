# Homework 10: Code Review and Changes

The assignment requests changes made after reviewing GitHub Copilot-generated or Copilot-suggested code. A Copilot session was not provided, so the following are three documented implementation and review decisions. They should not be represented as verified changes to Copilot code. Actual Copilot prompts and changes can be added after a review in VS Code.

## 1. Apply standardization before distance-based calculations

The seven specified numeric clustering features are standardized with `StandardScaler` before calculating standardized Euclidean distance, constructing the average-linkage dendrogram, or fitting K-Means. Raw distance is calculated separately only for the requested comparison. This prevents dollar-valued spending and income from dominating merely because of their numerical scales.

## 2. Make the hierarchical comparison reproducible and set the cut height from the linkage matrix

The dendrogram uses exactly 60 customers sampled with `random_state=42`, average linkage, and Euclidean distance. Rather than selecting an arbitrary threshold by eye, the script calculates a height between the merges yielding four and three groups, draws the corresponding horizontal line, and asserts that the resulting cut has four clusters. Lower merges mean more similar observations or groups.

## 3. Standardize K-Means settings and validate multiple choices of k

Each candidate k from 2 through 6 uses `random_state=42` and `n_init=20`. The script reports both inertia and silhouette score, then uses the assignment's required final k=4 for interpretation. The reported results independently support this choice because k=4 has the highest silhouette score among those tested.

## 4. Protect the validation measure from target leakage

`Campaign_Response_Pct` is intentionally excluded from all fitted clustering models and used only after segment assignments are complete. Cluster profiles use the original measurement units, making the resulting business interpretations more meaningful.

## 5. Compare groups without equating their numeric labels

The comparison joins hierarchical and K-Means results by `Customer_ID` and calculates Adjusted Rand Index, which measures agreement without assuming that cluster label numbers should match. The hierarchical model uses the sample of 60 while K-Means fits all 600.
