import numpy as np
import pandas as pd
import pickle

import os,sys
import data_explorer as de
import base_functions as bf

spiketrains = bf.import_data('L234', rop=True, area='V1', frps=True)[0]
neurons = spiketrains.columns.values.tolist()
remaining_neurons = neurons.copy()
print(len(remaining_neurons))

features_reduced = 0
grouped = {}
while(len(neurons) > 0) :
    n = neurons[0]
    neurons = neurons[1:]#.remove(n)
    print(f'{len(neurons)} ({features_reduced})', end='\r')
    
    st = spiketrains[n].values.tolist()
    similar_neurons = [n]
    
    remaining_neurons = neurons.copy()
    for neuron in remaining_neurons:
        if st == spiketrains[neuron].values.tolist() :
            similar_neurons.append(neuron)
            neurons.remove(neuron)
    
    if len(similar_neurons) > 1 :
        features_reduced+=len(similar_neurons)-1
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