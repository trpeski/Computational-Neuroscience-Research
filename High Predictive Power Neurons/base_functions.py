import numpy as np
import pandas as pd
import sys
import os

import os,sys
sys.path.append(os.getcwd())
import data_explorer as de

results_path = 'E:/Eleftheria/workspace/Computational-Neuroscience-Research/High Predictive Power Neurons'

# Function to import neuron data
def import_data(layer, rop=None, area='V1', hpp=None, frps=False):
    neurons = de.get_generic_filtered_neurons('3', False, area, layer, rop)
    neurons.sort()
    print(f"Number of neurons: {len(neurons)}")

    if frps :
        y = de.get_directions('3')
        X = de.get_frps('3')[de.unst_list(neurons)]
    else :
        y = de.get_angles_per_frame('3')
        X = de.get_spiketrains('3', False).iloc[y['2pf']][de.st_list(neurons)]
        y = y['class']
    
    return X, y
