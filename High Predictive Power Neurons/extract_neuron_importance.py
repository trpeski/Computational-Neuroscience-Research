import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import KFold, train_test_split
import itertools
import os
import json
import pandas as pd
import sys
sys.path.append(os.getcwd()+'/code')
import utils.data_explorer as de


# Hpps identified by Chris, Marios
allhpps = {
    3 : ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920'],
    4 : ['V166', 'V195', 'V223', 'V299', 'V306', 'V341', 'V351', 'V357', 'V369', 'V422', 'V448', 'V483', 'V486', 'V724', 'V960', 'V977', 'V1055', 'V1067', 'V1070', 'V1077', 'V1792', 'V1880', 'V1997', 'V2043', 'V2077', 'V2164', 'V2423', 'V3293', 'V3449', 'V3460', 'V3517', 'V3768', 'V4994', 'V5052', 'V5172', 'V5326', 'V5376', 'V5381', 'V5465', 'V5528', 'V5535', 'V5549', 'V5621', 'V5625', 'V5664', 'V6643', 'V6660', 'V6685', 'V6687', 'V6766', 'V6787', 'V6921', 'V7039', 'V7163'],
    5 : ['V284', 'V645', 'V686', 'V796', 'V810', 'V866', 'V895', 'V932', 'V989', 'V1150', 'V1906', 'V1916', 'V1945', 'V1990', 'V2135', 'V2311', 'V2337', 'V2362', 'V2388', 'V3263', 'V3276', 'V3465', 'V3527', 'V3554', 'V3557', 'V3563', 'V3572', 'V3580', 'V3633', 'V3662', 'V3728', 'V3733', 'V3776', 'V3905', 'V4039', 'V4691', 'V4739', 'V4844', 'V4854', 'V5071', 'V5107', 'V5296', 'V5333', 'V6906', 'V6948', 'V6949', 'V7000', 'V7003', 'V7093', 'V7210', 'V7266', 'V7294', 'V7389', 'V7422', 'V7464', 'V7492', 'V7607', 'V7839', 'V8080', 'V8108', 'V8157', 'V8159', 'V8215', 'V8230', 'V8314', 'V8323', 'V8367'],
    6 : ['V404', 'V466', 'V592', 'V648', 'V761', 'V1013', 'V1224', 'V1353', 'V1408', 'V1438', 'V1520', 'V1696', 'V1708', 'V1811', 'V1908', 'V1940', 'V1967', 'V1990', 'V2000', 'V2044', 'V2120', 'V2133', 'V2146', 'V2153', 'V2157', 'V2224', 'V2457', 'V2830', 'V2890', 'V3041', 'V3068', 'V3079', 'V3124', 'V3140', 'V4049', 'V5404', 'V5806', 'V6137', 'V6144', 'V6754', 'V6768', 'V6829', 'V6874', 'V7019'],
    7 : ['V303', 'V387', 'V504', 'V531', 'V748', 'V896', 'V899', 'V936', 'V1311', 'V2154', 'V2160', 'V2191', 'V2283', 'V2318', 'V2373', 'V2376', 'V2441', 'V2556', 'V2620', 'V2680', 'V2752', 'V2813', 'V2873', 'V3695', 'V3752', 'V3761', 'V3881', 'V4043', 'V4116', 'V4131', 'V4796', 'V5372', 'V5421', 'V5927', 'V5947', 'V5949', 'V5951', 'V6005', 'V6009', 'V6092', 'V6094', 'V6183', 'V6258', 'V6329', 'V6372', 'V6382', 'V6395', 'V6585', 'V6612', 'V7672', 'V7707', 'V7734']
}

# Define hyperparameters and their ranges
hyperparameters = {
    'conv_layers': [1],
    'filters': [16],
    'kernel_size': [6],
    'dense_units': [64],
    'dropout_rate': [0.3],
    'l2_reg': [0.002],
    'batch_size': [128],
    'epochs': [15],
    'learning_rate': [0.001]
}

model_results_path = '/home/psilou/code/misc/tony/model_results'

# Create a directory to save results
if not os.path.exists(model_results_path):
    os.makedirs(model_results_path)

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
                filters=hp['filters'] * (2 ** i),
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

# Function to import neuron data 
def import_data(layer, rop=None, area='V1', hpp=None):
    neurons = de.get_generic_filtered_neurons('3', False, area, layer, rop)
    if hpp is not None:
        hpps = pd.Series(de.unst_list(allhpps[3])).astype(int)
        #hpp = [int(x[1:]) for x in hpps[3]]
        if hpp :
            neurons = set(hpps)&(set(neurons))
        else :
            neurons = set(neurons)-set(hpps)

    print(f"Number of neurons: {len(neurons)}")
    y = de.get_angles_per_frame('3')
    #X = de.get_spiketrains('3', False).iloc[y['2pf']][de.st_list(hpp)]
    X = de.get_spiketrains('3', False).iloc[y['2pf']][de.st_list(neurons)]
    return X, y['class']


#X, y = import_data('L234', rop=True, area='V1', hpp=True)
X, y = import_data('L234', rop=True, area='V1')

print(len(X), len(y))
neuronids = X.columns

# Convert X to numpy array and reshape for Conv1D
X = X.values  # Shape: (num_samples, num_neurons)
X = X[..., np.newaxis]  # Shape becomes: (num_samples, num_neurons, 1)


# Convert labels to categorical
num_classes = 16
y = tf.keras.utils.to_categorical(y, num_classes)

#k folds 
skf = KFold(n_splits=5, shuffle=True, random_state=42)

# Generate all combinations of hyperparameters
keys = list(hyperparameters.keys())
combinations = list(itertools.product(*(hyperparameters[key] for key in keys)))


# Train and evaluate the model for each combination of hyperparameters
results = []
model_id = 0
# Loop over all combinations of hyperparameters
for idx, combination in enumerate(combinations):
    
    model_id += 1
    accuracies = []

    # Iterate over all folds of the cross-validation
    for train_index, val_index in skf.split(X, y):
   
        hp = dict(zip(keys, combination))
        print(f"\nTraining model {idx+1}/{len(combinations)} with hyperparameters: {hp}")
        
        # Build and train the model
        model = build_model((X[0].shape), num_classes, hp)
        print(X.shape)
        print(X[train_index].shape)
        
   
        history = model.fit(
            X[train_index], y[train_index],
            epochs=hp['epochs'],
            batch_size=hp['batch_size'],
            validation_data=(X[val_index], y[val_index]),
            verbose=0
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X[val_index], y[val_index], verbose=0)
        print(f'model_{idx}_{model_id} - Validation Accuracy: {accuracy*100:.2f}% - Validation Loss: {loss:.4f}') 
        
        # Save model weights and metrics
        
        accuracies.append(accuracy)

    metrics = {
        'model_id': model_id,
        'hyperparameters': hp,
        'validation_loss': loss,
        'validation_accuracy': np.mean(accuracies)
    }
    results.append(metrics)

    model_filename = f"model_{model_id}.keras"
    model.save(os.path.join(model_results_path, model_filename))
        
        
        
# Get best model
best_model_index = np.argmax([result['validation_accuracy'] for result in results])
best_hyperparams = results[best_model_index]['hyperparameters']
best_accuracy = results[best_model_index]['validation_accuracy']
best_model_id = results[best_model_index]['model_id']

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(f'{model_results_path}', 'hyperparameter_tuning_results.csv'), index=False)

# Save the best hyperparameters to a JSON file
'''
with open(os.path.join(f'{model_results_path}', 'best_hyperparameters.json'), 'w') as f:
    json.dump({
        'best_model_index': best_model_index,
        'best_hyperparameters': best_hyperparams
    }, f)
'''
print(f"\nBest Validation Accuracy: {best_accuracy*100:.2f}%")
print(f"Best Hyperparameters: {best_hyperparams}")



import numpy as np
from concurrent.futures import ProcessPoolExecutor



from tensorflow.keras.models import load_model # type: ignore
model = load_model(os.path.join(model_results_path, f"model_{best_model_id}.keras"))

baseline_accuracy = model.evaluate(X, y, verbose=1)[1]
# Function to evaluate the importance of a group of features
def evaluate_permuted_neuron(neuron_index, n_repeats=15):
    
    scores = []
    
    for _ in range(n_repeats):
        X_val_permuted = X.copy()

        # Permute the neuron to break its structure and evaulate the impact on the model accuracy 
        np.random.shuffle(X_val_permuted[:, neuron_index, :])

        loss, accuracy = model.evaluate(X_val_permuted, y, verbose=0)

        # Normalize the importance by the baseline accuracy
        normalized_importance = (baseline_accuracy - accuracy) / baseline_accuracy
        #print(f"Neuron {neuronids[neuron_index]} - Accuracy: {accuracy*100:.2f}% - Normalized Importance: {normalized_importance}")
        scores.append(normalized_importance)
        


    predictive_power = np.mean(scores)
    return neuron_index, predictive_power



num_features = X.shape[1]
feature_indices = np.arange(num_features)
neurons = [[i] for i in range(num_features)]
neurons = [i for i in range(num_features)]


feature_importances = {}


feature_importances_path = '/home/psilou/code/misc/tony/feature_importances.feather'

reload = False
# Load groups instead of creating them
if reload and os.path.exists(feature_importances_path):
    feature_importances = np.load(feature_importances_path, allow_pickle=True).item()
else:
    # Loop over neurons sequentially
    for neuron, neuronid in zip(neurons, neuronids):
        neuron_index, importance = evaluate_permuted_neuron(neuron)
        feature_importances[neuronid] = importance
        print(f"neuron {neuronid} - Importance: {importance}")

        # Optionally save intermediate results
        #partial_importances = {neuron_key: importance}
        #np.save(f'feature_importances_{i}.npy', partial_importances)
        #print(f"Saved importances for group {i}")

    # Save the final importances to a file
    #np.save(feature_importances_path, feature_importances)
    feature_importances = pd.DataFrame(pd.Series(feature_importances), columns=['importance'])
    print(feature_importances)
    feature_importances.to_feather(feature_importances_path)
    print("Saved final feature importances to 'feature_importances.feather'")

# Sort neurons by importance
sorted_neurons = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

# Print the most important neurons
print("Most important neurons:")
for neuron, importance in sorted_neurons[:50]:
    print(f"Neuron {neuron} - Importance: {importance}")

