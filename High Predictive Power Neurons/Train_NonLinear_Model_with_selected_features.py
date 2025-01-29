# This script trains and evaluates non-linear models (CNN, Random Forest, SVM) 
# using selected features and high predictive power (HPP) features.
# It performs 5-fold cross-validation to compare the performance of models trained on these different sets of features.
# The script also includes visualization of the results to facilitate comparison.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import base_functions as bf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import ttest_ind

frps_data = True # frps is True, else spikes

frps_data_flag = '_frps' if frps_data else ''

CNN_TRAIN = False
RANDOM_FOREST_TRAIN = False
SVM_TRAIN = True

# Load the data for V1 area layers 2, 3, and 4 only ROP(reliable orientation preference) neurons
X, y = bf.import_data('L234', rop=True, area='V1', frps=frps_data)

#X_general, y_general = bf.import_data('L234', rop=None, area='V1', frps=frps_data)
X_general, y_general = bf.import_full_data(frps=frps_data)

# Load feature importances from perumtation importance in feature_importances.feather
important_permutation_features_path = f'feature_importances.feather'
important_permutation_features = pd.read_feather(important_permutation_features_path)
kept_permutation_features = important_permutation_features[-37:].index.values

print(f"Permutation features: {kept_permutation_features}")
print(f"Permutation features shape: {kept_permutation_features.shape}")
print(f"Permutation features type: {type(kept_permutation_features)}")

# Load important features
important_features_path = f'{bf.results_path}/feature_importances_lasso{frps_data_flag}.feather'
important_features = pd.read_feather(important_features_path).set_index('Neuron')['Importance']
kept_features = important_features[-25:].index

# Prepare data with selected features
X_kept = X[kept_features]

# Prepare data with HPP features
allhpps = {
    3: ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920']
}
X_hpps = X[allhpps[3]]

# Prepare data with JadBio selected features 

jadbio_features = [
    'V5721', 'V5898', 'V5274', 'V4380', 'V2227', 'V4694', 'V4475', 'V5600', 
    'V2275', 'V4698', 'V3411', 'V4981', 'V4451', 'V4903', 'V5020', 'V4724', 
    'V5825', 'V2600', 'V4933', 'V4845', 'V2332', 'V5639', 'V8856', 'V6059', 
    'V5705'
]

X_jadbio = X_general[jadbio_features]

# print HPP and selected neurons ids and their intersection ratio
print(f"Selected features: {X_kept.columns}")
print(f"HPP features: {X_hpps.columns}")
print(f"Intersection ratio: {len(set(X_hpps.columns) & set(X_kept.columns)) / len(set(X_hpps.columns)):.2f}")

# print intersection of HPP and selected neurons
print(f"Intersection of HPP and selected neurons: {set(X_hpps.columns) & set(X_kept.columns)}")

# print intersection of selected neurons with kept_permutation_features
print(f"Intersection of selected neurons with kept_permutation_features: {set(X_kept.columns) & set(kept_permutation_features)}")

# print ratio of intersection of selected neurons and jadbio features
print(f"Intersection ratio of selected neurons and jadbio features: {len(set(X_jadbio.columns) & set(X_kept.columns)) / len(set(X_jadbio.columns)):.2f}")

# Define CNN model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# CNN Training and Evaluation
if CNN_TRAIN:
# Perform 5-fold cross-validation
    accuracies_kept = []
    accuracies_hpps = []

    for train_index, test_index in kf.split(X):
        X_train_kept, X_test_kept = X_kept.iloc[train_index], X_kept.iloc[test_index]
        X_train_hpps, X_test_hpps = X_hpps.iloc[train_index], X_hpps.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshape data for CNN
        X_train_kept_cnn = np.expand_dims(X_train_kept.values, axis=2)
        X_test_kept_cnn = np.expand_dims(X_test_kept.values, axis=2)
        X_train_hpps_cnn = np.expand_dims(X_train_hpps.values, axis=2)
        X_test_hpps_cnn = np.expand_dims(X_test_hpps.values, axis=2)

        # Train model with selected features
        model_kept = create_cnn_model((X_train_kept_cnn.shape[1], 1))
        model_kept.fit(X_train_kept_cnn, y_train, epochs=5, batch_size=32, verbose=0)
        y_pred_kept = np.argmax(model_kept.predict(X_test_kept_cnn), axis=1)
        accuracies_kept.append(accuracy_score(y_test, y_pred_kept))

        # Train model with HPP features
        model_hpps = create_cnn_model((X_train_hpps_cnn.shape[1], 1))
        model_hpps.fit(X_train_hpps_cnn, y_train, epochs=5, batch_size=32, verbose=0)
        y_pred_hpps = np.argmax(model_hpps.predict(X_test_hpps_cnn), axis=1)
        accuracies_hpps.append(accuracy_score(y_test, y_pred_hpps))

    # Print results
    print(f"Mean accuracy with selected features (CNN): {np.mean(accuracies_kept):.4f}")
    print(f"Mean accuracy with HPP features (CNN): {np.mean(accuracies_hpps):.4f}")

    # Plot results
    plt.figure(figsize=(12, 7))
    plt.plot(accuracies_kept, label='Selected Features Accuracy', marker='o')
    plt.plot(accuracies_hpps, label='HPP Features Accuracy', marker='o')
    plt.xlabel('Fold', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('5-Fold Cross-Validation Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()


if RANDOM_FOREST_TRAIN:
    # Perform 5-fold cross-validation for Random Forest
    rf_accuracies_kept = []
    rf_accuracies_hpps = []

    for train_index, test_index in kf.split(X):
        X_train_kept, X_test_kept = X_kept.iloc[train_index], X_kept.iloc[test_index]
        X_train_hpps, X_test_hpps = X_hpps.iloc[train_index], X_hpps.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train Random Forest model with selected features
        rf_model_kept = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_kept.fit(X_train_kept, y_train)
        y_pred_kept = rf_model_kept.predict(X_test_kept)
        rf_accuracies_kept.append(accuracy_score(y_test, y_pred_kept))

        # Train Random Forest model with HPP features
        rf_model_hpps = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_hpps.fit(X_train_hpps, y_train)
        y_pred_hpps = rf_model_hpps.predict(X_test_hpps)
        rf_accuracies_hpps.append(accuracy_score(y_test, y_pred_hpps))

    # Print results for Random Forest
    print(f"Mean accuracy with selected features (Random Forest): {np.mean(rf_accuracies_kept):.4f}")
    print(f"Mean accuracy with HPP features (Random Forest): {np.mean(rf_accuracies_hpps):.4f}")

    # Plot results for Random Forest
    plt.figure(figsize=(12, 7))
    plt.plot(rf_accuracies_kept, label='Selected Features Accuracy (RF)', marker='o')
    plt.plot(rf_accuracies_hpps, label='HPP Features Accuracy (RF)', marker='o')
    plt.xlabel('Fold', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('5-Fold Cross-Validation Accuracy Comparison (Random Forest)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()

if SVM_TRAIN:
    # Run multiple iterations for SVM
    iterations = 5
    all_svm_accuracies_kept = []
    all_svm_accuracies_hpps = []
    all_svm_accuracies_permutation = []
    all_svm_accuracies_jadbio = []

    for _ in range(iterations):
        svm_accuracies_kept = []
        svm_accuracies_hpps = []
        svm_accuracies_permutation = []
        svm_accuracies_jadbio = []
        
        for train_index, test_index in kf.split(X):
            X_train_kept, X_test_kept = X_kept.iloc[train_index], X_kept.iloc[test_index]
            X_train_hpps, X_test_hpps = X_hpps.iloc[train_index], X_hpps.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_permutation = X[kept_permutation_features]
            X_permutation_train, X_permutation_test = X_permutation.iloc[train_index], X_permutation.iloc[test_index]

            # Train SVM model with selected features
            svm_model_kept = SVC(kernel='rbf', random_state=42)
            svm_model_kept.fit(X_train_kept, y_train)
            y_pred_kept = svm_model_kept.predict(X_test_kept)
            svm_accuracies_kept.append(accuracy_score(y_test, y_pred_kept))

            # Train SVM model with HPP features
            svm_model_hpps = SVC(kernel='rbf', random_state=42)
            svm_model_hpps.fit(X_train_hpps, y_train)
            y_pred_hpps = svm_model_hpps.predict(X_test_hpps)
            svm_accuracies_hpps.append(accuracy_score(y_test, y_pred_hpps))
            
            # Train SVM model with permutation features
            svm_model_permutation = SVC(kernel='rbf', random_state=42)
            svm_model_permutation.fit(X_permutation_train, y_train)
            y_pred_permutation = svm_model_permutation.predict(X_permutation_test)
            svm_accuracies_permutation.append(accuracy_score(y_test, y_pred_permutation))
            
            # Train with JadBio features 
            X_train_jadbio, X_test_jadbio = X_jadbio.iloc[train_index], X_jadbio.iloc[test_index]
            svm_model_jadbio = SVC(kernel='rbf', random_state=42)
            svm_model_jadbio.fit(X_train_jadbio, y_train)
            y_pred_jadbio = svm_model_jadbio.predict(X_test_jadbio)
            svm_accuracies_jadbio.append(accuracy_score(y_test, y_pred_jadbio))
            
            

        all_svm_accuracies_kept.extend(svm_accuracies_kept)
        all_svm_accuracies_hpps.extend(svm_accuracies_hpps)
        all_svm_accuracies_permutation.extend(svm_accuracies_permutation)

    # Calculate p-value
    t_stat, p_value = ttest_ind(all_svm_accuracies_kept, all_svm_accuracies_hpps)

    # Print results
    print(f"Mean accuracy with selected features (SVM): {np.mean(all_svm_accuracies_kept):.4f}")
    print(f"Mean accuracy with HPP features (SVM): {np.mean(all_svm_accuracies_hpps):.4f}")
    print(f"P-value: {p_value:.4f}")

    print(f"Mean accuracy with permutation features (SVM): {np.mean(all_svm_accuracies_permutation):.4f}")
    print(f"Mean accuracy with JadBio features (SVM): {np.mean(svm_accuracies_jadbio):.4f}")

    # Plot results for SVM
    plt.figure(figsize=(12, 7))
    plt.plot(all_svm_accuracies_kept, label='Selected Features Accuracy (SVM)', marker='o')
    plt.plot(all_svm_accuracies_hpps, label='HPP Features Accuracy (SVM)', marker='o')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('SVM Accuracy Comparison Over Multiple Iterations', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()

if SVM_TRAIN:
    # Perform 5-fold cross-validation for SVM on the whole dataset
    svm_accuracies_all = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train SVM model with all features
        svm_model_all = SVC(kernel='rbf', random_state=42)
        svm_model_all.fit(X_train, y_train)
        y_pred_all = svm_model_all.predict(X_test)
        svm_accuracies_all.append(accuracy_score(y_test, y_pred_all))

    # Print results for SVM with all features
    print(f"Mean accuracy with all features (SVM): {np.mean(svm_accuracies_all):.4f}")
    print(f"Highest cross-validated accuracy with all features (SVM): {np.max(svm_accuracies_all):.4f}")

    # Plot results for SVM with all features
    plt.figure(figsize=(12, 7))
    plt.plot(svm_accuracies_all, label='All Features Accuracy (SVM)', marker='o')
    plt.xlabel('Fold', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('5-Fold Cross-Validation Accuracy with All Features (SVM)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()

# Train an SVM model with features that have the highest Mutual Information
# Prepare data with all features
all_features = X.columns


# Calculate mutual information for all features

warnings.filterwarnings("ignore", category=UserWarning)
#ignore all warnings
warnings.simplefilter("ignore")

mi_all = mutual_info_classif(X, y, discrete_features=True)
mi_all_series = pd.Series(mi_all, index=all_features)

# Normalize all features
important_features_normalized = (important_features - important_features.min()) / (important_features.max() - important_features.min())
normalized_all = important_features_normalized.reindex(all_features).fillna(0)

# Train SVM model with features that have the highest Mutual Information
highest_mi_features = mi_all_series.nlargest(37).index
X_highest_mi = X[highest_mi_features]

svm_accuracies_highest_mi = []

for train_index, test_index in kf.split(X):
    X_train_highest_mi, X_test_highest_mi = X_highest_mi.iloc[train_index], X_highest_mi.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train SVM model with features that have the highest Mutual Information
    svm_model_highest_mi = SVC(kernel='rbf', random_state=42)
    svm_model_highest_mi.fit(X_train_highest_mi, y_train)
    y_pred_highest_mi = svm_model_highest_mi.predict(X_test_highest_mi)
    svm_accuracies_highest_mi.append(accuracy_score(y_test, y_pred_highest_mi))
    
# Print results for SVM with features that have the highest Mutual Information
print(f"Mean accuracy with features that have the highest Mutual Information (SVM): {np.mean(svm_accuracies_highest_mi):.4f}")
print(f"Highest cross-validated accuracy with features that have the highest Mutual Information (SVM): {np.max(svm_accuracies_highest_mi):.4f}")

