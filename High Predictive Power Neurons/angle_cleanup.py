import numpy as np
import pandas as pd
import data_explorer as de

# Define the input Feather file path
input_file = r"E:\\Eleftheria\\workspace\\data\\frps\\mouse3_stimuli_frps.feather"
output_file = r"E:\\Eleftheria\\workspace\\data\\frps\\mouse3_stimuli_frps_with_angles.csv"

def get_class_of_angles (angle) :
    return int((angle+22.5/2)//22.5)

#angles = np.load('monet_m24617_tolias_angle_labels_actual.npy')
angles = de.get_directions('3')
angles = pd.Series(angles.flatten())
angles.to_csv('m3_angles.csv', index=False)
classes = angles.map(get_class_of_angles)
classes.to_csv('m3_angle_classes.csv', index=False)
classes = pd.DataFrame(classes, columns=['angle_class'])

neurons = de.get_generic_filtered_neurons('3', False, 'V1', 'L234', True)
neurons.sort()
print(f"Number of neurons: {len(neurons)}")
X = de.get_frps('3')[de.unst_list(neurons)]
X.columns = ['V' + col for col in X.columns]
print(X)
frps_and_angles = pd.concat([X,classes], axis=1)
print(frps_and_angles)
frps_and_angles.to_csv(output_file, index=False)