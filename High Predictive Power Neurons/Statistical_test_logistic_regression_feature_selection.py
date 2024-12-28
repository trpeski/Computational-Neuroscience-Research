import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import base_functions as bf

import os,sys
sys.path.append(os.getcwd())
import data_explorer as de
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter

frps_data = True # frps is True, else spikes

frps_data_flag = '_frps' if frps_data else ''

# OG Hpps 
allhpps = {
    3: ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920'],
    4: ['V166', 'V195', 'V223', 'V299', 'V306', 'V341', 'V351', 'V357', 'V369', 'V422', 'V448', 'V483', 'V486', 'V724', 'V960', 'V977', 'V1055', 'V1067', 'V1070', 'V1077', 'V1792', 'V1880', 'V1997', 'V2043', 'V2077', 'V2164', 'V2423', 'V3293', 'V3449', 'V3460', 'V3517', 'V3768', 'V4994', 'V5052', 'V5172', 'V5326', 'V5376', 'V5381', 'V5465', 'V5528', 'V5535', 'V5549', 'V5621', 'V5625', 'V5664', 'V6643', 'V6660', 'V6685', 'V6687', 'V6766', 'V6787', 'V6921', 'V7039', 'V7163'],
    # Other layers truncated for brevity
}

# Load the data for V1 area layers 2, 3, and 4 only ROP(reliable orientation prefrence) neurons 
X, y = bf.import_data('L234', rop=True, area='V1', frps=frps_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression
maxiter = 1000
logistic = LogisticRegression(max_iter=maxiter, multi_class='multinomial')
params = {'C': [1/150], 'penalty': ['l1'], 'solver': ['saga']}

# Store the top 37 features for each iteration
top_features_list = []

for i in range(5):
    print(f"Iteration {i+1}/50")
    grid_search = GridSearchCV(logistic, param_grid=params, cv=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Optimal model
    best_logistic = grid_search.best_estimator_

    # Train the best model
    best_logistic.fit(X_train, y_train)

    # Predictions
    y_pred = best_logistic.predict(X_test)

    # Feature Importances (average of absolute value of coefficients across classes for each feature)
    feature_importances = pd.Series(np.mean(np.abs(best_logistic.coef_), axis=0), index=X.columns)
    important_features = feature_importances.abs().sort_values(ascending=True)

    # Get the top 37 features
    top_37_features = important_features.index[-37:].tolist()
    top_features_list.append(top_37_features)
    
    # Print the most important features for this iteration
    print(f"Top 37 features for iteration {i+1}: {top_37_features}")

# Check how many times the top 37 features are the same

# Flatten the list of top features and count occurrences
flat_top_features = [feature for sublist in top_features_list for feature in sublist]
feature_counts = Counter(flat_top_features)

# Find features that appear in the top 37 in all iterations
consistent_features = [feature for feature, count in feature_counts.items() if count == 5]

print(f"Number of consistent features across all iterations: {len(consistent_features)}")
print("Consistent features:", consistent_features)

# Plot the number of times each feature appears in the top 37
plt.figure(figsize=(12, 7))
plt.bar(feature_counts.keys(), feature_counts.values(), color='blue', alpha=0.7)
plt.xlabel('Neuron', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.title('Number of Times Each Feature Appears in Top 37', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.6)
plt.tight_layout()

plt.show()
