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

# Plot mutual information for selected features
plt.figure(figsize=(12, 7))
mi_kept_series.sort_values(ascending=False).plot(kind='bar', color='blue', alpha=0.7)
plt.xlabel('Neuron', fontsize=14, fontweight='bold')
plt.ylabel('Mutual Information', fontsize=14, fontweight='bold')
plt.title('Mutual Information of Selected Features with Output', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.show()

# Plot mutual information for HPP features
plt.figure(figsize=(12, 7))
mi_hpps_series.sort_values(ascending=False).plot(kind='bar', color='red', alpha=0.7)
plt.xlabel('Neuron', fontsize=14, fontweight='bold')
plt.ylabel('Mutual Information', fontsize=14, fontweight='bold')
plt.title('Mutual Information of HPP Features with Output', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.show()

# Plot histogram comparing mutual information distributions
plt.figure(figsize=(12, 7))
plt.hist(mi_kept, bins=20, alpha=0.7, label='Selected by Lasso', color='blue')
plt.hist(mi_hpps, bins=20, alpha=0.7, label='HPP Features', color='red')
plt.xlabel('Mutual Information', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Histogram of Mutual Information Distributions', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.show()

# print the means
print(f'Mean of mutual information for selected features: {np.mean(mi_kept)}')
print(f'Mean of mutual information for HPP features: {np.mean(mi_hpps)}')

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(mi_kept, mi_hpps, alternative='two-sided')
print(f'Mann-Whitney U test statistic: {stat}, p-value: {p_value}')

# Generate null distribution by sampling random features
n_iterations = 10
null_distributions = []

for _ in range(n_iterations):
    random_features = np.random.choice(X.columns, size=37, replace=False)
    X_random = X[random_features]
    mi_random = mutual_info_classif(X_random, y)
    null_distributions.append(mi_random)

null_distributions = np.array(null_distributions).flatten()

# Plot histogram comparing mutual information distributions with null distribution
plt.figure(figsize=(12, 7))
bincnt = 10
weights_kept = np.ones_like(mi_kept) / len(mi_kept)
weights_hpps = np.ones_like(mi_hpps) / len(mi_hpps)
weights_null = np.ones_like(null_distributions) / len(null_distributions)

plt.hist(mi_kept, bins=bincnt, alpha=0.7, label='Selected by Lasso', color='blue', weights=weights_kept)
plt.hist(mi_hpps, bins=bincnt, alpha=0.7, label='HPP Features', color='red', weights=weights_hpps)
plt.hist(null_distributions, bins=bincnt, alpha=0.7, label='Null Distribution', color='green', weights=weights_null)
#plt.hist(mi_hpps, bins=bincnt, alpha=0.7, label='HPP Features', color='red')
#plt.hist(null_distributions, bins=bincnt, alpha=0.7, label='Null Distribution', color='green')
plt.xlabel('Mutual Information', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Histogram of Mutual Information Distributions with Null', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.show()