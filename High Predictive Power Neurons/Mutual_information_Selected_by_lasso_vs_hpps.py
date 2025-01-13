import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import base_functions as bf
from scipy.stats import mannwhitneyu

# Load the data for V1 area layers 2, 3, and 4 only ROP (reliable orientation preference) neurons
X, y = bf.import_data('L234', rop=True, area='V1')

# Load important features
important_features_path = f'{bf.results_path}/feature_importances_lasso.feather'
important_features = pd.read_feather(important_features_path).set_index('Neuron')['Importance']
kept_features = important_features[-37:].index

# Prepare data with selected features
X_kept = X[kept_features]

# Prepare data with HPP features
allhpps = {
    3: ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920']
}
X_hpps = X[allhpps[3]]

# Calculate mutual information for selected features
mi_kept = mutual_info_classif(X_kept, y, discrete_features=True)
mi_kept_series = pd.Series(mi_kept, index=kept_features)

# Calculate mutual information for HPP features
mi_hpps = mutual_info_classif(X_hpps, y, discrete_features=True)
mi_hpps_series = pd.Series(mi_hpps, index=allhpps[3])

# Normalize important features
important_features_normalized = (important_features - important_features.min()) / (important_features.max() - important_features.min())

# Ensure the sizes match for plotting
normalized_kept = important_features_normalized.loc[kept_features]
normalized_hpps = important_features_normalized.loc[allhpps[3]]
# Prepare data with all features
all_features = X.columns

# Calculate mutual information for all features
mi_all = mutual_info_classif(X, y, discrete_features=True)
mi_all_series = pd.Series(mi_all, index=all_features)

# Normalize all features
normalized_all = important_features_normalized.reindex(all_features).fillna(0)

# take intersection of HPP and selected neurons
intersection = list(set(X_hpps.columns) & set(X_kept.columns))

# Plot all features
plt.figure(figsize=(12, 7))
plt.scatter(normalized_all, mi_all_series, color='gray', alpha=0.5, label='All Features')
plt.scatter(normalized_kept, mi_kept_series, color='blue', label='Selected Features')
plt.scatter(normalized_hpps, mi_hpps_series, color='red', label='HPP Features')
# Plot mean horizontal lines for each category
plt.axhline(y=mi_all_series.mean(), color='gray', linestyle='--', linewidth=1, label='Mean All Features')
plt.axhline(y=mi_kept_series.mean(), color='blue', linestyle='--', linewidth=1, label='Mean Selected Features')
plt.axhline(y=mi_hpps_series.mean(), color='red', linestyle='--', linewidth=1, label='Mean HPP Features')
# Plot intersection features
if intersection:
    normalized_intersection = important_features_normalized.loc[intersection]
    mi_intersection_series = mi_kept_series.loc[intersection]
    plt.scatter(normalized_intersection, mi_intersection_series, color='green', label='Intersection Features')
    plt.axhline(y=mi_intersection_series.mean(), color='green', linestyle='--', linewidth=1, label='Mean Intersection Features')
    normalized_intersection = important_features_normalized.loc[intersection]
    mi_intersection_series = mi_kept_series.loc[intersection]
    plt.scatter(normalized_intersection, mi_intersection_series, color='green', label='Intersection Features')
plt.xlabel('Normalized Importance', fontsize=14, fontweight='bold')
plt.ylabel('Mutual Information', fontsize=14, fontweight='bold')
plt.title('Normalized Importance vs Mutual Information', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.show()

