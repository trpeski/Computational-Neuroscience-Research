import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import KFold
import os
import sys
from scipy.stats import ttest_rel
import concurrent.futures
import matplotlib.pyplot as plt
from scipy import stats

allhpps = {
    3 : ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920'],
    4 : ['V166', 'V195', 'V223', 'V299', 'V306', 'V341', 'V351', 'V357', 'V369', 'V422', 'V448', 'V483', 'V486', 'V724', 'V960', 'V977', 'V1055', 'V1067', 'V1070', 'V1077', 'V1792', 'V1880', 'V1997', 'V2043', 'V2077', 'V2164', 'V2423', 'V3293', 'V3449', 'V3460', 'V3517', 'V3768', 'V4994', 'V5052', 'V5172', 'V5326', 'V5376', 'V5381', 'V5465', 'V5528', 'V5535', 'V5549', 'V5621', 'V5625', 'V5664', 'V6643', 'V6660', 'V6685', 'V6687', 'V6766', 'V6787', 'V6921', 'V7039', 'V7163'],
    5 : ['V284', 'V645', 'V686', 'V796', 'V810', 'V866', 'V895', 'V932', 'V989', 'V1150', 'V1906', 'V1916', 'V1945', 'V1990', 'V2135', 'V2311', 'V2337', 'V2362', 'V2388', 'V3263', 'V3276', 'V3465', 'V3527', 'V3554', 'V3557', 'V3563', 'V3572', 'V3580', 'V3633', 'V3662', 'V3728', 'V3733', 'V3776', 'V3905', 'V4039', 'V4691', 'V4739', 'V4844', 'V4854', 'V5071', 'V5107', 'V5296', 'V5333', 'V6906', 'V6948', 'V6949', 'V7000', 'V7003', 'V7093', 'V7210', 'V7266', 'V7294', 'V7389', 'V7422', 'V7464', 'V7492', 'V7607', 'V7839', 'V8080', 'V8108', 'V8157', 'V8159', 'V8215', 'V8230', 'V8314', 'V8323', 'V8367'],
    6 : ['V404', 'V466', 'V592', 'V648', 'V761', 'V1013', 'V1224', 'V1353', 'V1408', 'V1438', 'V1520', 'V1696', 'V1708', 'V1811', 'V1908', 'V1940', 'V1967', 'V1990', 'V2000', 'V2044', 'V2120', 'V2133', 'V2146', 'V2153', 'V2157', 'V2224', 'V2457', 'V2830', 'V2890', 'V3041', 'V3068', 'V3079', 'V3124', 'V3140', 'V4049', 'V5404', 'V5806', 'V6137', 'V6144', 'V6754', 'V6768', 'V6829', 'V6874', 'V7019'],
    7 : ['V303', 'V387', 'V504', 'V531', 'V748', 'V896', 'V899', 'V936', 'V1311', 'V2154', 'V2160', 'V2191', 'V2283', 'V2318', 'V2373', 'V2376', 'V2441', 'V2556', 'V2620', 'V2680', 'V2752', 'V2813', 'V2873', 'V3695', 'V3752', 'V3761', 'V3881', 'V4043', 'V4116', 'V4131', 'V4796', 'V5372', 'V5421', 'V5927', 'V5947', 'V5949', 'V5951', 'V6005', 'V6009', 'V6092', 'V6094', 'V6183', 'V6258', 'V6329', 'V6372', 'V6382', 'V6395', 'V6585', 'V6612', 'V7672', 'V7707', 'V7734']
}

# Append the code directory to the system path
sys.path.append(os.getcwd() + '/code')
import utils.data_explorer as de  # Custom module for data handling

# Step 1: Load the feature importances from the feather file
feature_importances_path = '/home/psilou/code/misc/tony/feature_importances.feather'
feature_importances = pd.read_feather(feature_importances_path)

# Sort the feature importances and get the top 37 neurons
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Keep top neurons excluding the last 20
top_neurons = feature_importances.index[:37].tolist()
#top_neurons = allhpps[3]  

# Step 2: Load data the same way as the original script
def import_data(layer, rop=None, area='V1'):
    neurons = de.get_generic_filtered_neurons('3', False, area, layer, rop)
    print(f"Number of neurons: {len(neurons)}")
    y = de.get_angles_per_frame('3')
    X = de.get_spiketrains('3', False).iloc[y['2pf']][de.st_list(neurons)]
    return X, y['class']

# Load the data
X, y = import_data('L234', rop=True, area='V1')
neuronids = X.columns

# Step 2 (continued): Keep only the 37 most important neurons
# Ensure the top neurons are present in the dataset
available_neurons = [neuron for neuron in top_neurons if neuron in X.columns]
print(f"Number of top neurons available in data: {len(available_neurons)}")

X = X[available_neurons]

# Step 3: Train a similar model using the same methodology

# Convert X to numpy array and reshape for Conv1D
X = X.values  # Shape: (num_samples, num_neurons)
X = X[..., np.newaxis]  # Shape becomes: (num_samples, num_neurons, 1)

# Convert labels to categorical
num_classes = 16
y = tf.keras.utils.to_categorical(y, num_classes)

# Define hyperparameters (same as the original script)
hp = {
    'conv_layers': 1,
    'filters': 16,
    'kernel_size': 6,
    'dense_units': 64,
    'dropout_rate': 0.3,
    'l2_reg': 0.002,
    'batch_size': 32,
    'epochs': 2,
    'learning_rate': 0.001
}

# Function to build and compile the model
def build_model(input_shape, num_classes, hp):
    model = models.Sequential()
    # Add Conv1D layers
    for i in range(hp['conv_layers']):
        if i == 0:
            model.add(layers.Conv1D(
                filters=hp['filters'],
                kernel_size=hp['kernel_size'],
                activation='relu',
                input_shape=input_shape,
                kernel_regularizer=regularizers.l2(hp['l2_reg'])
            ))
        else:
            model.add(layers.Conv1D(
                filters=hp['filters'],
                kernel_size=hp['kernel_size'],
                activation='relu',
                kernel_regularizer=regularizers.l2(hp['l2_reg'])
            ))
        model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    # Add Dense layers
    model.add(layers.Dense(
        units=hp['dense_units'],
        activation='relu',
        kernel_regularizer=regularizers.l2(hp['l2_reg'])
    ))
    model.add(layers.Dropout(rate=hp['dropout_rate']))
    model.add(layers.Dense(
        units=hp['dense_units'],
        activation='relu',
        kernel_regularizer=regularizers.l2(hp['l2_reg'])
    ))
    model.add(layers.Dense(num_classes, activation='softmax'))
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

# Train and evaluate the model using cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Build and train the model
    model = build_model(X_train.shape[1:], num_classes, hp)
    history = model.fit(
        X_train, y_train,
        epochs=hp['epochs'],
        batch_size=hp['batch_size'],
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracies.append(accuracy)
    print(f'Validation Accuracy: {accuracy*100:.2f}% - Validation Loss: {loss:.4f}')

# Step 4: Display the average accuracy
average_accuracy = np.mean(accuracies)
print(f'\nAverage Validation Accuracy across folds: {average_accuracy*100:.2f}%')

# Retrain on the whole dataset 
model = build_model(X.shape[1:], num_classes, hp)
history = model.fit(
    X, y,
    epochs=hp['epochs'],
    batch_size=hp['batch_size'],
    verbose=1
)



# Statistical test

# Compute average accuracy with top 37 neurons
average_accuracy_top = np.mean(accuracies)
print(f'\nAverage Validation Accuracy with Top 37 Neurons: {average_accuracy_top*100:.2f}%')

# Load the full dataset with all neurons
X_full, y_full = import_data('L234', rop=True, area='V1')

# Convert X_full to numpy array and reshape for Conv1D
X_full = X_full.values
X_full = X_full[..., np.newaxis]

# Convert labels to categorical
y_full = tf.keras.utils.to_categorical(y_full, num_classes)

# Get the list of all neurons
all_neurons = neuronids.tolist()


# Test statistical significance using random samples

# accuracies from random samples of 37 neurons (already sampled)
samples = [16.09, 15.69, 16.9, 15.32, 18.62, 17.87, 18.81, 15.21, 18.51, 15.35, 15.77, 16.5, 17.08, 16.81, 19.54, 16.1, 18.04, 18.19, 17.58, 17.27, 17.12, 17.04, 17.84, 16.19, 16.15, 18.72, 15.63, 18.01, 17.8, 18.02, 16.15, 16.9, 17.46, 16.52, 16.67, 17.18, 16.11, 17.05, 15.6, 15.77, 17.03, 17.62, 16.57, 17.14, 14.7, 17.53, 14.05, 16.43, 19.2, 18.34, 17.49, 16.53, 17.61, 15.59, 14.86, 15.04, 16.33, 17.43, 17.07, 16.75, 15.15, 18.19, 19.06, 17.0, 17.61, 15.51, 17.22, 17.21, 16.9, 16.94, 16.54, 15.19, 17.71, 15.89, 16.42, 15.28, 17.6, 15.91, 18.54, 18.87, 16.8, 17.93, 18.12, 16.86, 17.06, 15.53, 17.46, 17.12, 18.12, 15.65, 18.47, 15.63, 17.22, 16.08, 16.83, 16.37, 16.57, 18.2, 18.66, 16.12, 18.09, 18.75, 17.29, 18.1, 16.51, 17.48, 15.96, 17.18, 15.15, 15.57, 16.41, 15.71, 17.26, 15.11, 17.4, 16.83, 15.7, 15.05, 17.69, 16.57, 17.45, 17.44, 15.41, 17.85, 17.65, 18.07, 15.76, 17.29, 17.41, 18.01, 17.06, 17.67, 15.89, 15.98, 17.29, 17.71, 16.13, 17.51, 16.6, 15.07, 16.15, 17.45, 14.96, 17.15, 19.14, 18.53, 15.65, 15.46, 15.17, 16.11, 17.53, 19.14, 17.64, 16.51, 16.48, 16.86, 15.76, 16.7, 14.21, 13.92, 17.33, 17.01, 15.63, 15.71, 17.17, 17.18, 16.65, 18.84, 15.49, 15.75, 15.96, 17.55, 14.57, 16.38, 14.8, 16.64, 15.97, 13.81, 18.96, 15.75, 16.94, 18.32, 16.62, 16.92, 18.17, 16.26, 18.88, 16.12, 16.07, 16.4, 15.71, 16.86, 18.41, 15.37, 14.38, 18.32, 17.28, 17.67, 15.15, 16.89]



def test_feature_selection_significance(random_accuracies, selected_accuracy, alpha=0.05):

    # Convert inputs to numpy arrays
    random_accuracies = np.array(random_accuracies)
    
    # Calculate statistics of the null distribution
    mean_null = np.mean(random_accuracies)
    std_null = np.std(random_accuracies, ddof=1)  # ddof=1 for sample standard deviation
    
    # Calculate z-score for the selected accuracy
    z_score = (selected_accuracy - mean_null) / std_null
    
    # Calculate p-value (one-tailed test if we expect better than random,
    # otherwise use two-tailed test)
    # Using one-tailed test since we expect our selection to be better
    p_value = 1 - stats.norm.cdf(z_score)
    
    # Prepare results
    results = {
        'p_value': p_value,
        'z_score': z_score,
        'null_mean': mean_null,
        'null_std': std_null,
        'is_significant': p_value < alpha,
        'percentile': stats.percentileofscore(random_accuracies, selected_accuracy)
    }
    
    return results

def plot_significance_test(random_accuracies, selected_accuracy, results):

    plt.figure(figsize=(10, 6))

    # Plot histogram of null distribution
    plt.hist(
        random_accuracies,
        bins=100,
        density=True,
        alpha=0.7,
        label='Random Selections Distribution\n (200 random samples)'
    )

    # Plot normal distribution fit
    x = np.linspace(min(random_accuracies), max(random_accuracies), 100)
    plt.plot(
        x,
        stats.norm.pdf(x, results['null_mean'], results['null_std']),
        'r-',
        label='Normal Distribution Fit'
    )

    # Plot selected accuracy
    plt.axvline(
        x=selected_accuracy,
        color='g',
        linestyle='--',
        label=f'Selected Features (p={results["p_value"]:.4f})'
    )

    # Make plot nicer
    plt.title('Feature Selection Significance Test', fontweight='bold', fontsize=16)
    plt.xlabel('Accuracy', fontweight='bold', fontsize=14)
    plt.ylabel('Density', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Remove the right and upper axis lines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Adjust legend
    legend = plt.legend(fontsize=12, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Set tick labels to bold
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    return plt.gcf()

# Example usage
results = test_feature_selection_significance(
    random_accuracies=samples,
    selected_accuracy=average_accuracy_top*100
)

print(f"P-value: {results['p_value']:.4f}")
print(f"Z-score: {results['z_score']:.4f}")
print(f"Your selection is in the {results['percentile']:.1f}th percentile")
print(f"Statistically significant: {results['is_significant']}")

# Optional: Create visualization
fig = plot_significance_test(samples, average_accuracy_top * 100, results)

# Save the plot
fig.savefig('/home/psilou/code/misc/tony/significance_test.png')


