import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import sys
import os

sys.path.append(os.getcwd() + '/code')
import utils.data_explorer as de

# OG Hpps 
allhpps = {
    3: ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920'],
    4: ['V166', 'V195', 'V223', 'V299', 'V306', 'V341', 'V351', 'V357', 'V369', 'V422', 'V448', 'V483', 'V486', 'V724', 'V960', 'V977', 'V1055', 'V1067', 'V1070', 'V1077', 'V1792', 'V1880', 'V1997', 'V2043', 'V2077', 'V2164', 'V2423', 'V3293', 'V3449', 'V3460', 'V3517', 'V3768', 'V4994', 'V5052', 'V5172', 'V5326', 'V5376', 'V5381', 'V5465', 'V5528', 'V5535', 'V5549', 'V5621', 'V5625', 'V5664', 'V6643', 'V6660', 'V6685', 'V6687', 'V6766', 'V6787', 'V6921', 'V7039', 'V7163'],
    # Other layers truncated for brevity
}

# Function to import neuron data
def import_data(layer, rop=None, area='V1', hpp=None):
    neurons = de.get_generic_filtered_neurons('3', False, area, layer, rop)
    print(f"Number of neurons: {len(neurons)}")
    y = de.get_angles_per_frame('3')
    X = de.get_spiketrains('3', False).iloc[y['2pf']][de.st_list(neurons)]
    return X, y['class']

# Load the data
X, y = import_data('L234', rop=True, area='V1')

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression
logistic = LogisticRegression(max_iter=10, multi_class='multinomial', solver='lbfgs')
params = {'C': [0.01, 0.1, 1.0, 5.0], 'penalty': ['l1','l2'], 'solver': ['liblinear', 'saga']}
grid_search = GridSearchCV(logistic, param_grid=params, cv=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Optimal model
best_logistic = grid_search.best_estimator_

# Train the best model
best_logistic.fit(X_train, y_train)

# Predictions
y_pred = best_logistic.predict(X_test)

# Evaluation
print("Best Logistic Regression Hyperparameters:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Display
ConfusionMatrixDisplay.from_estimator(best_logistic, X_test, y_test, cmap='Blues')
plt.show()
feature_importances = 0
# Feature Importance
if 'l1' in grid_search.best_params_['penalty']:
    feature_importances = pd.Series(best_logistic.coef_.flatten(), index=X.columns).abs()
else:
    print("Feature importance analysis is not supported for non-sparse penalties (e.g., L2).")

important_features = feature_importances[feature_importances > 0].sort_values()

# Normalization
important_features_normalized = (important_features - important_features.min()) / (important_features.max() - important_features.min())

# Plot
plt.figure(figsize=(12, 7))
plt.plot(important_features_normalized.index, important_features_normalized, marker='o', linestyle='-', label='Normalized Importance', color='blue')
plt.axhline(important_features_normalized.mean(), color='red', linestyle='--', label='Mean Importance')
plt.axhline(important_features_normalized.mean() + important_features_normalized.std(), color='green', linestyle='--', label='Mean + 1 Std')
plt.axhline(important_features_normalized.mean() - important_features_normalized.std(), color='green', linestyle='--', label='Mean - 1 Std')

# Highlighting original HPPs
hpps_indices = [i for i, neuron in enumerate(important_features_normalized.index) if neuron in allhpps[3]]
plt.scatter(hpps_indices, important_features_normalized.iloc[hpps_indices], color='red', label='Original HPPs', zorder=5)

# Customization
plt.xlabel("Neuron (sorted by importance)", fontsize=14, fontweight='bold')
plt.ylabel("Normalized Importance", fontsize=14, fontweight='bold')
plt.title("Feature Importance After Hyperparameter Optimization", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.savefig('/home/psilou/code/misc/tony/data/feature_importance/regr4.png')
