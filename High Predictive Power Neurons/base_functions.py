import numpy as np
import pandas as pd
import sys
import os

import os,sys
sys.path.append(os.getcwd())
import data_explorer as de

results_path = 'E:/Eleftheria/workspace/Computational-Neuroscience-Research/High Predictive Power Neurons'

# Function to import neuron data
def import_data(layer, rop=None, area='V1', hpp=None, frps=False) :
    neurons = de.get_generic_filtered_neurons('3', False, area, layer, rop)
    neurons.sort()
    print(f"Number of neurons: {len(neurons)}")

    if frps :
        y = de.get_directions('3').flatten()
        y = de.get_class_of_angles(y)
        X = de.get_frps('3')[de.unst_list(neurons)]
        X.columns = ['V' + col for col in X.columns]
    else :
        y = de.get_angles_per_frame('3')
        X = de.get_spiketrains('3', False).iloc[y['2pf']][de.st_list(neurons)]
        y = y['class']
    
    return X, y

def import_full_data(frps=False):
    if frps :
        y = de.get_directions('3').flatten()
        y = de.get_class_of_angles(y)
        X = de.get_frps('3')
        X.columns = ['V' + col for col in X.columns]
    else :
        y = de.get_angles_per_frame('3')
        X = de.get_spiketrains('3', False)
        y = y['class']
    
    return X, y