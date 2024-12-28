import numpy as np
import pandas as pd
import pickle

import os,sys
import data_explorer as de
import base_functions as bf
from sklearn.metrics.pairwise import cosine_similarity

spiketrains = bf.import_data('L234', rop=True, area='V1', frps=True)[0]
neurons = spiketrains.columns.values.tolist()
remaining_neurons = neurons.copy()
print(len(remaining_neurons))

features_reduced = 0
grouped = {}
while(len(neurons) > 0) :
    n = neurons[0]
    st = spiketrains[n].values.reshape(1, -1)
    similar_neurons = [n]
    
    neurons = neurons[1:]
    print(f'{len(neurons)} ({features_reduced})', end='\r')
    
    remaining_neurons = neurons.copy()
    for neuron in remaining_neurons:
        st_neuron = spiketrains[neuron].values.reshape(1, -1)
        similarity = cosine_similarity(st, st_neuron)[0, 0]
        if similarity > 0.8:
            similar_neurons.append(neuron)
            neurons.remove(neuron)
    
    if len(similar_neurons) > 1 :
        features_reduced += len(similar_neurons) - 1
        print('\t\tFound redundancy', features_reduced, end='\r')
        
    grouped[n] = set(similar_neurons)

groupname_to_index = {}
for groupindex in grouped :
    groupname = str.join(', ', grouped[groupindex])
    groupname_to_index[groupname] = groupindex

print(groupname_to_index)
print(len(groupname_to_index))
path = f'{bf.results_path}/redundancy_groups.pickle'
pickle.dump(groupname_to_index, open(path, 'wb'))
print()

print('Original Spiketrains')
print(spiketrains)
ogfeatures = spiketrains.columns.values.tolist()

grouped_spiketrains = spiketrains[list(groupname_to_index.values())]
grouped_spiketrains.columns = list(groupname_to_index.keys())

print()
print('Grouped Spiketrains')
print(grouped_spiketrains)

print()
print(f'Features reduced: {features_reduced} (from {len(ogfeatures)} to {len(grouped_spiketrains.columns.values.tolist())})')

path = f'{bf.results_path}/redundancy_free_spiketrains.feather'
grouped_spiketrains.to_feather(path)