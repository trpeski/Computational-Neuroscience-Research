import torch
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os, sys
sys.path.append(os.getcwd()+'/../High Predictive Power Neurons')
import data_explorer as de 
import pandas as pd

def get_weights_from_L4_to_hpp(mouse):
    l4n = de.get_generic_filtered_neurons(mouse, False, 'V1', 'L4', None)
    hppn = de.unst_list(de.allhpps[int(mouse)])
    
    sttc = de.get_sttc(mouse, False)
    sttc['z'] = (sttc['STTC'] - sttc['CtrlGrpMean']) / sttc['CtrlGrpStd']
    sttc = sttc[sttc['z'] > 4]
    sttc = sttc[sttc['NeuronA'].isin(l4n) & sttc['NeuronB'].isin(hppn)]

    unique_neurons_4 = pd.unique(sttc['NeuronA'])
    neuron_id_map_4 = {old_id: new_id for new_id, old_id in enumerate(unique_neurons_4, start=1)}

    unique_neurons_23 = pd.unique(sttc['NeuronB'])
    neuron_id_map_23 = {old_id: new_id for new_id, old_id in enumerate(unique_neurons_23, start=1)}
    
    sttc['NeuronA'] = sttc['NeuronA'].map(neuron_id_map_4)
    sttc['NeuronB'] = sttc['NeuronB'].map(neuron_id_map_23)
    
    edge_list = list(sttc[['NeuronA', 'NeuronB', 'STTC']].itertuples(index=False, name=None))
    return edge_list, len(unique_neurons_4), len(unique_neurons_23)

# Define the file paths
file_paths = {
    'test_angles': r'C:\Users\antua\Downloads\monet\monet\test_angles.pt',
    'test_videos': r'C:\Users\antua\Downloads\monet\monet\test_videos.pt',
    'train_angles': r'C:\Users\antua\Downloads\monet\monet\train_angles.pt',
    'train_videos': r'C:\Users\antua\Downloads\monet\monet\train_videos.pt'
}

# Load the files
data = {key: torch.load(path).numpy() for key, path in file_paths.items()}

# Prepare the data
X_train = data['train_videos']
y_train = data['train_angles']
X_test = data['test_videos']
y_test = data['test_angles']

# Normalize the input data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the data to match the expected input shape for the model
X_train = np.transpose(X_train, (0, 2, 3, 1))
X_test = np.transpose(X_test, (0, 2, 3, 1))

# Collapse angles into 16 classes
y_train = (y_train // 22.5).astype(int)
y_test = (y_test // 22.5).astype(int)

# Get the STTC weights and neuron counts
mouse = '3'  # Example mouse ID
edge_list, num_neurons_4, num_neurons_23 = get_weights_from_L4_to_hpp(mouse)

# Define the CNN model
def create_model(num_neurons_4, num_neurons_23):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(126, 216, 6), padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(num_neurons_4, activation='relu'),  # Match the number of layer 4 neurons
        Dense(num_neurons_23, activation='relu'),  # Match the number of HPP neurons
        Dense(16, activation='softmax')  # Output layer for classification into 16 angles
    ])
    return model

# Initialize model with random weights
model_random = create_model(num_neurons_4, num_neurons_23)
model_random.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with random weights
model_random.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model with random weights
loss_random, accuracy_random = model_random.evaluate(X_test, y_test)
print(f'Test loss with random weights: {loss_random}')
print(f'Test accuracy with random weights: {accuracy_random}')

# Initialize model with STTC weights
model_sttc = create_model(num_neurons_4, num_neurons_23)

# Initialize the weights of the first dense layer with STTC values
initial_weights = model_sttc.layers[-3].get_weights()
for edge in edge_list:
    initial_weights[0][edge[0]-1, edge[1]-1] = edge[2]
model_sttc.layers[-3].set_weights(initial_weights)

model_sttc.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with STTC weights
model_sttc.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model with STTC weights
loss_sttc, accuracy_sttc = model_sttc.evaluate(X_test, y_test)
print(f'Test loss with STTC weights: {loss_sttc}')
print(f'Test accuracy with STTC weights: {accuracy_sttc}')

# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model_sttc.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()