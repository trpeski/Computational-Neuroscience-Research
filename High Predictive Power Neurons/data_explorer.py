import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
import math
import random

from scipy.stats import ttest_ind
from scipy.stats import f_oneway

mice = {
    '3' : '24617',
    '4' : '24705',
    '5' : '25133',
    '6' : '25341',
    '7' : '25387'
}

s = {
    '3' : '3',
    '4' : '3',
    '5' : '6',
    '6' : '2',
    '7' : '3'
}
idx = {
    '3' : '10',
    '4' : '24', 
    '5' : '15',
    '6' : '20',
    '7' : '17'
}

norm_factors = {
    '3' : 0.62891,
    '4' : 0.62891,
    '5' : 0.58203,
    '6' : 0.62891,
    '7' : 0.71484
}

frames_to_keep = {
    False : {
        '3' : [420, 26541+1],
        '4' : [779, 26905+1],
        '5' : [419, 26954+1],
        '6' : [932, 27074+1],
        '7' : [848, 26974+1]
    }, 

    True : {
        '3' : [None, None],
        '4' : [None, None],
        '5' : [None, None],
        '6' : [None, None],
        '7' : [None, None]
    }
}

base_path = f'E:/Eleftheria/workspace/data'

def sample_from_list (list, n, replace) :
    return np.random.choice(list, size=n, replace=replace)

def round_down(x, a):
    return math.floor(x / a) * a

def round_up (x,a) :
    return math.ceil(x / a) * a

def round_nearest(x, a):
    return round(x / a) * a

def get_activity_text (spont) :
    if spont : return 'spont'
    return 'stim'


def st_id (id) :
    if str(id)[0] == 'V' :
        return id
    return f'V{id}'

def unst_id (id) :
    if str(id)[0] == 'V' :
        return str(id)[1:]
    return str(id)

def st_list (list) :
    return [st_id(l) for l in list]

def unst_list (list) :
    return [unst_id(l) for l in list]


def get_undirected_pairs (keys) :
    undirected_pairs = [(keys[i],keys[j]) for i in range(len(keys)-1) for j in range(i+1, len(keys))]
    return undirected_pairs


def get_orientation (x) :
    phi   = np.min([abs(x), 360 - abs(x)])
    omega = phi if phi <= 90 else 180-phi
    return omega

def get_180ange_my_style (x) :
    gamma = x if x<=180 else x-180
    return gamma

def orientation_difference(w1, w2):
    x = w2 - w1
    return get_orientation(x)

def make_aggr_bins (data_dict, bins=[], min_bin=None, max_bin=None, bin_count=None, bin_size=None) :
    if len(bins)!=0: return bins

    if min_bin is None :
        min_bin = min([min(data) for data in data_dict.values()])
    
    if max_bin is None :
        max_bin = max([max(data) for data in data_dict.values()])

    # If no bin size is provided
    if bin_size is None :
        # If no bin count is provided
        if bin_count is None :
            # Set bin_count to a default
            bin_count = 25
            
        # Calculate bin size
        bin_size=(max_bin-min_bin)/bin_count

    bins = np.arange(min_bin, max_bin+2*bin_size, bin_size)
    return bins

def get_bounds (mouse) :
    return pd.read_csv(f'{base_path}/within_boundaries/mouse'+mice[mouse]+'_withinBoundaries_15um-no_gap.csv')

def get_neurons_by_area_layer_path (mouse, area, layer) :
    return f'{base_path}/neurons/by_area_layer/mouse'+mice[mouse]+'_neurons_'+area+'_'+layer+'.csv'

def get_neurons_by_area_layer (mouse, area, layer) :
    return pd.read_csv(get_neurons_by_area_layer_path(mouse, area, layer))

def get_exclusion_by_area_layer_path (mouse, spont, area, layer) :
    #at = '' if area == 'V1' else '' 
    att = '' if spont else '_stimuli'
    return f'{base_path}/neurons/exclusion_by_area_layer/mouse'+mice[mouse]+f'{att}_neurons_exclusion{area}_{layer}.csv'

def get_filtered_neurons_by_area_layer (mouse, spont, area, layer) :
    neurons = [
        pd.read_csv(get_exclusion_by_area_layer_path(mouse, spont, area, f'L{l}')) 
        for l in layer[1:]
    ]
    neurons = pd.concat(neurons, ignore_index=True)['neuronID'].values.tolist()
    #path = get_exclusion_by_area_layer_path(mouse, spont, area, layer)
    return neurons

def get_generic_filtered_neurons (mouse, spont, area, layer, rop) :
    neurons = get_filtered_neurons_by_area_layer(mouse, spont, area, layer)
    if rop is not None :
        ropns = get_rop(mouse)
        if rop : neurons = list(set(neurons).intersection(set(ropns)))
        else : neurons = list(set(neurons)-(set(ropns)))

    return neurons

def make_bins (data, bins=[], min_bin=None, max_bin=None, bin_count=None, bin_size=None) :
    print(min_bin, max_bin, bin_size)

    if len(bins)!=0: return bins
    
    if min_bin is None :
        min_bin = min(data)
    
    if max_bin is None :
        max_bin = max(data)

    # If no bin size is provided
    if bin_size is None :
        # If no bin count is provided
        if bin_count is None :
            # Set bin_count to a default
            bin_count = 25
            
        # Calculate bin size
        bin_size=(max_bin-min_bin)/bin_count

    bins = np.arange(min_bin, max_bin+bin_size, bin_size)
    return bins
 

def make_histogram (data, bins=[], min_bin=None, max_bin=None, bin_count=None, bin_size=None, norm=True) :
    bins = make_bins(data, bins, min_bin, max_bin, bin_count, bin_size)
    hist, edges = np.histogram(data, bins)

    if not norm : return hist, bins
    else : 
        return hist/sum(hist), bins

def make_aggr_hist (data, bins) :
    hists = {}
    for case in data :
        hists[case], _ = make_histogram(data[case], bins)
    
    hists = pd.DataFrame(hists)
    return hists.mean(axis=1), hists.sem(axis=1)

def get_coords (mouse) :
    return pd.read_csv(f'{base_path}/coords/mouse{mice[mouse]}_coords.csv')

def get_euclidean_distance (mouse) :
    path = f'{base_path}/euclidean_distance/M{mouse}_euclidean_distance_data_all_neurons.feather'
    dists = pd.read_feather(path)
    return dists


def get_all_pair_euclidean_distance_path (mouse) :
    path = f'{base_path}/euclidean_distance/M{mouse}_euclidean_distance_data_all_pairs.feather'
    return path

def get_all_pair_euclidean_distance (mouse) :
    path = get_all_pair_euclidean_distance_path(mouse)
    dists = pd.read_feather(path)
    return dists

#TODO
def get_sttc (mouse, spont) :
    if spont :
        path = f'/home/savaglio/Firing_Prediction/Mouse_{mouse}/mouse{mice[str(mouse)]}_IoannisThreshold_3nz_1.5dc_full_60min_500-shifts_0-dt_pairs.feather'
    else :
        path = f'/home/savaglio/Firing_Prediction/Mouse_{mouse}/Mouse{mouse}_Stimuli_Eventograms_All_Neurons_3nz_1.5dc_500-shifts_0-dt_pairs.feather'
    
    df = pd.read_feather(path)
    return df

def get_specific_layer_sttc (mouse, spont, layer1, layer2, area='V1') :
    sttc = get_sttc(mouse, spont)
    n1 = get_neurons_of_layer(mouse, spont, layer1, area)
    n2 = get_neurons_of_layer(mouse, spont, layer2, area)
    sttc = sttc[sttc['NeuronA'].isin(n1) & sttc['NeuronB'].isin(n2)]

    return sttc

def get_spiketrains (mouse, spont) :
    if spont :
        #data_path = f'/home/savaglio/Firing_Prediction/Mouse_{mouse}/mouse{mice[str(mouse)]}_IoannisThreshold_3nz_1.5dc_full_60min.feather'
        data_path = f'{base_path}/spiketrains/mouse{mice[str(mouse)]}_IoannisThreshold_3nz_1.5dc_full_60min.feather'
    else :
        #data_path = f'/home/savaglio/Firing_Prediction/Mouse_{mouse}/Mouse{mouse}_Stimuli_Eventograms_All_Neurons_3nz_1.5dc.feather'
        data_path = f'{base_path}/spiketrains/Mouse{mouse}_Stimuli_Eventograms_All_Neurons_3nz_1.5dc.feather'
        
    return pd.read_feather(data_path)

def get_neurons_of_layer_spimple (mouse, spont, layer, area='V1') :
    at = '' if spont else '_stimuli'
    #path = f'/home/psilou/code/panel11/miscdata/mouse{mice[str(mouse)]}{at}_neurons_exclusion{area}_{layer}.csv'
    path = get_exclusion_by_area_layer_path(mouse, spont, area, layer)
    ns =  pd.read_csv(path)['neuronID'].values.tolist()
    return ns

def get_neurons_of_layer (mouse, spont, layer, area='V1') :
    layers = listify_layer(layer)
    neurons = []
    for l in layers : 
        neurons+= get_neurons_of_layer_spimple(mouse, spont, l, area)

    return neurons

def get_neurons (mouse, spont, area='V1') : 
    at = '' if spont else '_stimuli'
    neurons_ID_L2 = get_neurons_of_layer_spimple(mouse, spont, 'L2', area)
    neurons_ID_L3 = get_neurons_of_layer_spimple(mouse, spont, 'L3', area)
    neurons_ID_L4 = get_neurons_of_layer_spimple(mouse, spont, 'L4', area)

    return neurons_ID_L2, neurons_ID_L3, neurons_ID_L4

def get_neurons_dict (mouse, spont, area='V1') : 
    n2,n3,n4 = get_neurons(mouse, spont, area)
    return {'L2' : n2, 'L3' : n3, 'L4' : n4, 'L23': n2+n3}

def get_neurons_dict_spontstim (mouse, area='V1') :
    return {True : get_neurons_dict(mouse, True, area), False : get_neurons_dict(mouse, False, area)}


def get_selection_neurons (mouse, layer, rop, area='V1') :
    neurons = get_neurons_dict(mouse, False, area)
    
    if layer == 'L234' :
        neurons = neurons['L23']+neurons['L4']
    else :
        neurons = neurons[layer]
        
    if rop is not None :
        ropns = get_rop(mouse)
        if rop :
            neurons = [n for n in neurons if n in ropns]
        else :
            neurons = [n for n in neurons if n not in ropns]
            
    return neurons

def get_rop (mouse) :
    #ropneurons = pickle.load(open(f'/home/psilou/code/panel11/miscdata/m{mouse}_reliably_tuned_neurons.pickle', 'rb'))
    ropneurons = pickle.load(open(f'{base_path}/neurons/reliably_tuned/m{mouse}_reliably_tuned_neurons.pickle', 'rb'))
    return ropneurons
    
def get_subset_text (subset) :
    if subset is not None :
        if subset == 'min_rop_nonrop' :
            subsett = '_min_rop_nonrop_subsets'
        else :
            print('This subset method doesnt exist or you have not named it yet (dataexplorer)')
            exit(1)
    else :
        subsett = ''
    return subsett

def get_cofiring_path (mouse, group_layer, spont, gspont, grop, norm=False, null=False, strict_null=False, subset=None, garea='V1', refarea='V1', perc=False) :
    at = 'spont' if spont else 'stimuli'
    
    if gspont == 'both' :
        gat = 'both'
    elif gspont :
        gat = 'spont' 
    else :
        gat = 'stimuli'

    if grop is None :
        grt = ''
    else :
        grt = '_rop' if grop else '_nonrop' 
    
    if norm : nt = '_norm'
    else : nt = ''   

    if perc : pt = '_perc'
    else : pt = ''

    if strict_null : snt = 'strict'
    else : snt = ''

    if type(null) is int :
        nult = f'_{snt}null{null}'
    elif null == True :
        nult = f'_{snt}null'
    else :
        nult = '' 
    
    subsett = get_subset_text(subset)

    gat = get_areat(garea)
    rat = get_areat(refarea)

    path = f'{base_path}/cofiring/m{mouse}_{at}_cofiring{pt}{nt}_of{grt}{group_layer}{gat}_groups_of_L23{rat}_neurons{nult}{subsett}.csv'
    print(path)
    return path 

def get_cofiring (mouse, group_layer, spont, gspont, grop, norm=False, null=False, strict_null=False, subset=None, garea='V1', refarea='V1', perc=False) :
    path = get_cofiring_path(mouse, group_layer, spont, gspont, grop, norm=norm, null=null, strict_null=strict_null, subset=subset, garea=garea, refarea=refarea, perc=perc)
    print(path)

    try :
        data = pd.read_csv(path)
    except :
        data = pd.read_feather(path)
    
    return data

def get_areat (area) :
    if area == 'V1' :
        return ''
    return f'_{area}'

def get_sttc_groups_path_simple (mouse, layer1, layer2, spont, null=False, strict_null=False, rop=None, subset=None, area1='V1', area2='V1') :
    area1t = get_areat(area1)
    area2t = get_areat(area2)

    if spont == 'both' :
        path = f'{base_path}/sttc_groups_spont_stim_intersection/'
        path += f'{mouse}_{layer1}{layer2}{area1t}{area2t}_groups_across_activity.csv'
    
    else :
        if spont :
            path = f'{base_path}/sttc_groups_new/spont/'
        else :
            path = f'{base_path}/sttc_groups_new/stim/'

        if strict_null : snt = f'strict'
        else : snt = '' 
        
        if type(null) is int : 
            nt=f'_{snt}null{null}'
        elif null : nt=f'_{snt}null'
        else : nt = '_observed'
        
        if rop is not None :
            if rop :
                rt = '_rop'
            else :
                rt = '_nonrop'
        else :
            rt = ''

        subsett = get_subset_text(subset)

        path += f'Mouse_{mouse}_1.5_dc_0_dt_{layer1}_{layer2}{area1t}{area2t}_correlated{rt}_sttc_groups{nt}{subsett}.feather'

    return path 

def get_sttc_groups_simple (mouse, layer1, layer2, spont, null=False, strict_null=False, rop=None, subset=None, area1='V1', area2='V1') :
    path = get_sttc_groups_path_simple(mouse, layer1, layer2, spont, null, strict_null, rop, subset, area1, area2)
    
    if path[-4]=='.' :
        return pd.read_csv(path)
    else :
        return pd.read_feather(path)

def listify_layer (layer) :
    if len(layer)>2 : 
        layers = []
        for l in layer[1:] :
            layers.append(f'L{l}')
    else :
        layers = [layer]

    return layers

def listify_layercase (layer1, layer2) :
    if layer2 == 'L23' :
        destls = ['L2','L3', 'L2','L3']
        
        if layer1 == 'L234':
            sourcels = ['L2','L3','L4', 'L4']
        elif layer1 == 'L23' :
            sourcels = ['L2','L3']
        else :
            sourcels = [layer1, layer1]
    else :
        if layer1 == 'L234' :
            sourcels = ['L2','L3','L4']
            destls = [layer2, layer2, layer2]
        elif layer1 == 'L23' :
            sourcels = ['L2','L3']
            destls = [layer2, layer2]
        else :
            sourcels = [layer1]
            destls = [layer2]
    
    return sourcels, destls
    
def get_sttc_groups (mouse, layer1, layer2, spont, null=False, strict_null=False, rop=None, subset=None, area1='V1', area2='V1') :
    sourcels, destls = listify_layercase(layer1, layer2)

    groups = []
    for sl, dl in zip(sourcels, destls) :
        groups.append(get_sttc_groups_simple(mouse, sl, dl, spont, null, strict_null, rop, subset, area1, area2))

    groups = pd.concat(groups, ignore_index=True)

    if layer1=='L234' :
        nneurons = []
        ngroups = []
        ngroupsizes = [] 
        for n in groups['Obs Ref Neuron'].unique().tolist() :
            ng = groups[groups['Obs Ref Neuron']==n]
            if len(ng.index) < 2 : continue
            nneurons.append(n)
            ngroups.append(ng['STTC Group'].iloc[0]+'_'+ng['STTC Group'].iloc[1])
            ngroupsizes.append(ng['Group Size'].iloc[0]+ng['Group Size'].iloc[1])

        groups = pd.DataFrame({
            'Obs Ref Neuron' : nneurons, 
            'STTC Group' : ngroups, 
            'Group Size' : ngroupsizes
        })

    groups['quant'] = pd.qcut(groups['Group Size'], 4, labels=['small', 'small_middle', 'large_middle', 'large'])
    return groups

        
    
def get_rop_dependent_group_simple_path (mouse, spont, layer1, layer2, rop, area1='V1', area2='V1') :
    if spont == 'both' : at = 'both'
    elif spont : at = 'spont'
    else : at = 'stim'

    area1t = get_areat(area1)
    area2t = get_areat(area2)
    
    if rop : rt = 'rop'
    else : rt = 'nonrop'
    
    path = f'{base_path}/sttc_groups_rop_dependent/{at}/Mouse_{mouse}_{layer1}_{layer2}{area1t}{area2t}_{rt}_sttc_groups.feather'
    return path 

def get_rop_dependent_group_simple (mouse, spont, layer1, layer2, rop):
    return pd.read_feather(get_rop_dependent_group_simple_path (mouse, spont, layer1, layer2, rop))

def get_rop_dependent_group (mouse, spont, layer1, layer2, rop):
    sourcels, destls = listify_layercase(layer1, layer2)
    
    groups = []
    for sl, dl in zip(sourcels, destls) :
        groups.append(get_rop_dependent_group_simple(mouse, spont, sl, dl, rop))
        
    groups = pd.concat(groups, ignore_index=True)
    return groups

  
def get_rop_dependent_group_subset_simple_path (mouse, spont, layer1, layer2) :
    if spont=='both' : at = 'both'
    elif spont : at = 'spont'
    else : at = 'stim'
    
    path = f'{base_path}/sttc_groups_rop_dependent_subsets/{at}/Mouse_{mouse}_{layer1}_{layer2}_rop_dependent_subsets_sttc_groups.feather'
    return path 


def get_rop_dependent_group_subset_simple (mouse, spont, layer1, layer2):
    return pd.read_feather(get_rop_dependent_group_subset_simple_path (mouse, spont, layer1, layer2))

def get_rop_dependent_group_subset (mouse, spont, layer1, layer2):
    sourcels, destls = listify_layercase(layer1, layer2)
    
    groups = []
    for sl, dl in zip(sourcels, destls) :
        groups.append(get_rop_dependent_group_subset_simple(mouse, spont, sl, dl))
        
    groups = pd.concat(groups, ignore_index=True)
    return groups



def get_rop_dependent_group_subset_cofiring_simple_path (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period=None, perc=False, norm=False, area1='V1', area2='V1') :
    if spont : at = 'spont'
    else : at = 'stim'
    
    if gspont=='both' : gat = 'both'
    elif gspont : gat = 'spont'
    else : gat = 'stim'
    
    if rop : rt = 'rop'
    else : rt = 'nonrop'
    
    area1t = get_areat(area1)
    area2t = get_areat(area2)

    if similar_tuning_period is None : stp = ''
    else :
        assert(spont==False)
        if similar_tuning_period : stp = '_similar_tuning_period'
        else : stp = '_orthogonal_tuning_period'
    
    if perc : pt = '_perc'
    else : pt = ''
    
    if norm :
        nt = '_norm'
    else :
        nt = ''
    
    
    path = f'{base_path}/cofiring/rop_dependent_subsets/Mouse{mouse}_{at}{stp}_cofiring{pt}{nt}_of_{gat}_{layer1}_{layer2}{area1t}{area2t}_{rt}_subsets_{subset}.feather'
    return path 


def get_rop_dependent_group_subset_cofiring_simple (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period=None, perc=False, norm=False, area1='V1', area2='V1'):
    return pd.read_feather(get_rop_dependent_group_subset_cofiring_simple_path (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period, perc, norm, area1, area2))

def get_rop_dependent_group_subset_cofiring (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period=None, perc=False, norm=False, area1='V1', area2='V1') :
    sourcels, destls = listify_layercase(layer1, layer2)
    
    groups = []
    for sl, dl in zip(sourcels, destls) :
        groups.append(get_rop_dependent_group_subset_cofiring_simple(mouse, spont, gspont, sl, dl, rop, subset, similar_tuning_period, perc, norm, area1, area2))
        
    groups = pd.concat(groups, axis=1)
    return groups

def get_rop_dependent_cofiring_response_simple_path (mouse, spont, gspont, layer1, layer2, rop, subset=None, std=False, similar_tuning_period=None, perc=False, area1='V1', area2='V1') :
    if spont : at = 'spont'
    else : at = 'stim'

    if gspont=='both' : gat = 'both'
    elif gspont : gat = 'spont'
    else : gat = 'stim'
    
    rt = 'rop' if rop else 'nonrop'
    si = f'_{subset}' if subset is not None else ''

    area1t = get_areat(area1)
    area2t = get_areat(area2)
    
    if similar_tuning_period is None : stp = ''
    else :
        assert(spont==False)
        if similar_tuning_period : stp = '_similar_tuning_period'
        else : stp = '_orthogonal_tuning_period'
     
    if perc: 
        pt = '_perc'
    else :
        pt = ''
    
    path = f'{base_path}/firing_response/m{mouse}_{at}{stp}_response_of_{layer2}{area2t}_neurons_to_{gat}_{layer1}{area1t}_{rt}_group_cofiring{pt}{si}'
    if std :
        path += '_stds'
    path +='.feather'
    
    return path

def get_rop_dependent_cofiring_response_simple (mouse, spont, gspont, layer1, layer2, rop, subset=None, std=False,  similar_tuning_period=None, perc=False, area1='V1', area2='V1') :
    return pd.read_feather(get_rop_dependent_cofiring_response_simple_path (mouse, spont, gspont, layer1, layer2, rop, subset, std, similar_tuning_period, perc, area1, area2))

def get_rop_dependent_cofiring_response (mouse, spont, gspont, layer1, layer2, rop, subset=None, std=False, similar_tuning_period=None, perc=False, area1='V1', area2='V1') :
    sourcels, destls = listify_layercase(layer1, layer2, rop)
    
    groups = []
    for sl, dl in zip(sourcels, destls) :
        groups.append(get_rop_dependent_cofiring_response_simple(mouse, spont, gspont, sl, dl, rop, subset, std, similar_tuning_period, perc, area1, area2))
        
    groups = pd.concat(groups, ignore_index=True)
    return groups


def get_rop_dependent_cofiring_events_counts_path (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period=None) :
    if spont : at = 'spont'
    else : at = 'stim'
    
    if gspont=='both' : gat = 'both' 
    elif gspont : gat = 'spont' 
    else : gat = 'stim'
    
    if rop : rt = 'rop'
    else : rt = 'nonrop'
    
    if similar_tuning_period is None : stp = ''
    else :
        assert(spont==False)
        if similar_tuning_period : stp = '_similar_tuning_period'
        else : stp = '_orthogonal_tuning_period'
    
    path = f'{base_path}/cofiring/rop_dependent_subsets/Mouse{mouse}_counts_of_{at}{stp}_cofiring_of_{gat}_{layer1}_{layer2}_{rt}_subsets_{subset}.feather'
    return path 

def get_rop_dependent_cofiring_events_counts_simple (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period=None) :
    return pd.read_feather(get_rop_dependent_cofiring_events_counts_path(mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period))

def get_rop_dependent_cofiring_events_counts (mouse, spont, gspont, layer1, layer2, rop, subset, similar_tuning_period=None) :
    sourcels, destls = listify_layercase(layer1, layer2)
    
    groups = []
    for sl, dl in zip(sourcels, destls) :
        groups.append(get_rop_dependent_cofiring_events_counts_simple(mouse, spont, gspont, sl, dl, rop, subset, similar_tuning_period))
        
    groups = pd.concat(groups, ignore_index=True)
    return groups

def get_direction_tuning (mouse) :
    path = f'{base_path}/direction_tuning/mouse{mice[str(mouse)]}_s{s[str(mouse)]}_idx{idx[str(mouse)]}_strongest_amplitude_peak_all_cells.csv'
    return pd.read_csv(path)


def get_frames_of_periods_of_similar_tuning_path (mouse, n_similar=1) :
    path = f'{base_path}/spiketrains/periods_with_similar_angle_to_preference/mouse{mouse}_frames_of_similar_angle_period_keeping_{n_similar}_neighboring_angles.feather'
    return path

def get_frames_of_periods_of_orthogonal_tuning_path (mouse, n_similar=1) :
    path = f'{base_path}/spiketrains/periods_with_orthogonally-similar_angle_preference/mouse{mouse}_frames_of_orthogonal_angle_period_keeping_{n_similar}_neighboring_angles.feather'
    return path

def get_frames_of_periods_of_similar_tuning (mouse, n_similar=1) :
    return pd.read_feather(get_frames_of_periods_of_similar_tuning_path(mouse, n_similar))

def get_frames_of_periods_of_orthogonal_tuning (mouse, n_similar=1) :
    return pd.read_feather(get_frames_of_periods_of_orthogonal_tuning_path(mouse, n_similar))


def get_periods_of_similar_tuning_path (mouse, n_similar=1) :
    path = f'{base_path}/spiketrains/periods_with_similar_angle_to_preference/mouse{mouse}_similar_angle_period_keeping_{n_similar}_neighboring_angles.feather'
    return path

def get_periods_of_orthogonal_tuning_path (mouse, n_similar=1) :
    path = f'{base_path}/spiketrains/periods_with_orthogonally-similar_angle_preference/mouse{mouse}_orthogonal_angle_period_keeping_{n_similar}_neighboring_angles.feather'
    return path

def get_periods_of_similar_tuning (mouse, n_similar=1) :
    return pd.read_feather(get_periods_of_similar_tuning_path(mouse, n_similar))

def get_periods_of_orthogonal_tuning (mouse, n_similar=1) :
    return pd.read_feather(get_periods_of_orthogonal_tuning_path(mouse, n_similar))





def perform_permutation_test_equal_mean(dist1, dist2, permutation_test_size):
    # Mixed dists
    mixed_dists = np.concatenate([dist1, dist2])
    # Permutations test for mean
    sample_stat = abs(np.mean(dist1) - np.mean(dist2))

    stats = np.zeros(permutation_test_size)

    # Dist shapes
    D = dist1.shape[0]
    M = mixed_dists.shape[0]

    for k in range(permutation_test_size):
        random_shuffle_indices = np.random.choice(np.arange(M), M, replace=False)
        groupA = mixed_dists[random_shuffle_indices[0:D]]
        groupB = mixed_dists[random_shuffle_indices[D:]]
        stats[k] = abs(np.mean(groupA) - np.mean(groupB))
    p_value = np.mean(stats > sample_stat)

    return p_value

# Perform permutation test of equal means, welch's t-test, and ANOVA and return the statistics and p values
def perform_multiple_statistical_tests_on_two_dists(dist1, dist2, permutations=10000):
    # Welch's t-test
    welch_t_stat, welch_p_value = ttest_ind(dist1, dist2, equal_var=False, alternative='two-sided')

    # ANOVA
    anova_f_stat, anova_p_value = f_oneway(dist1, dist2)

    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    # Permutation test
    #perm_p_value_old = perform_permutation_test_on_two_dists_equal_means(dist1, dist2, permutations=permutations)
    #perm_p_value = perform_permutation_test_equal_mean(dist1, dist2,permutations)
    perm_p_value = perform_permutation_test_equal_mean(dist1, dist2,permutations)
    #print(perm_p_value_old, perm_p_value)

    # Return the statistics
    return welch_t_stat, welch_p_value, anova_f_stat, anova_p_value, perm_p_value

def get_cofiring_overlap_path (mouse, layer1, layer2, spont) :
    data_path = f'{base_path}/cofiring/cofiring_overlap/m{mouse}_{layer1}_{layer2}_cofiring_prob_data_of_ref_neurons_{get_activity_text(spont)}.feather'
    return data_path

def get_cofiring_overlap (mouse, layer1, layer2, spont) :
    data_path = get_cofiring_overlap_path(mouse, layer1, layer2, spont)
    return pd.read_feather(data_path)

def get_sttc_group_overlap_path (mouse, layer1, layer2, spont, null) :
    nt = '_null' if null else ''
    path = f'{base_path}/sttc_group_overlap/m{mouse}_spont{spont}_group{layer1}_ref{layer2}{nt}.feather'
    return path

def get_sttc_group_overlap (mouse, layer1, layer2, spont, null) :
    return pd.read_feather(get_sttc_group_overlap_path(mouse, layer1, layer2, spont, null))


def get_sttc_group_full_overlap_path (mouse, layer1, layer2, spont, null) :
    nt = '_null' if null else ''
    path = f'{base_path}/sttc_group_overlap/m{mouse}_spont{spont}_group{layer1}_ref{layer2}{nt}_full_overlap_data.pickle'
    return path

def get_spont_stim_sttc_group_overlap_path (mouse, layer1, layer2) :
    path = f'{base_path}/sttc_group_overlap/m{mouse}_spont_stim_group{layer1}_ref{layer2}_overlap.feather'
    return path

def get_sttc_group_full_overlap (mouse, layer1, layer2, spont, null) :
    return pickle.load(open(get_sttc_group_full_overlap_path(mouse, layer1, layer2, spont, null), 'rb'))

def get_firing_response_path (mouse, spont, gspont, perc=False, norm=False, grop=None, null=False, stds=False, strict_null=False, subset=None, area1='V1', area2='V1') :
    at = 'spont' if spont else 'stimuli'
    if gspont == 'both' :
        gat = 'both'
    elif gspont :
        gat = 'spont' 
    else :
        gat = 'stimuli'
        
    if perc : 
        pt = '_perc'
    else :
        pt = ''
        
    if norm : 
        nt = '_norm'
    else :
        nt = ''
        
    if stds :
        st = '_stds'
    else :
        st = ''

    if strict_null :
        snt = 'strict'
    else :
        snt = ''

    area1t = get_areat(area1)
    area2t = get_areat(area2)

    if grop is None :
        grt = ''
    elif grop :
        grt = '_rop'
    else :
        grt = '_nonrop'

    if type(null) is int :
        nult = f'_{snt}null{null}'
    elif null == True:
        nult = f'_{snt}null'
    else :
        nult = ''

    subsett = get_subset_text(subset)

    path = f'{base_path}/firing_response/m{mouse}_{at}_response_of_L23{area2t}_neurons_to_{gat}{grt}_L4{area1t}_groups{st}{pt}{nt}{nult}{subsett}.feather'
    return path

def get_firing_response (mouse, spont, gspont, perc=False, norm=False, grop=None, null=False, stds=False, strict_null=False, subset=None, area1='V1', area2='V1') :
    path = get_firing_response_path(mouse, spont, gspont, perc, norm, grop, null, stds, strict_null, subset, area1, area2)
    return pd.read_feather(path)

def get_absolute_orientation_difference_path (mouse) :
    return f'{base_path}/absolute_orientation_difference/M{mouse}_absolute_orientation_difference_all_neurons_no_th.feather'

def get_absolute_orientation_difference (mouse) :
    #return pickle.load(open(get_absolute_orientation_difference_path(mouse),'rb'))
    return pd.read_feather(get_absolute_orientation_difference_path(mouse))

def get_pearson_correlations_of_cofiring_with_population_filename (mouse, spont, gspont, glayer, player) :
    at = get_activity_text(spont)
    gat = get_activity_text(gspont)
    
    path = f"mouse{mouse}_pearson_corr_between_{gat}_{glayer}_group_cofiring_and_{at}_{player}_population_firing"
    return path

def get_pearson_correlations_of_cofiring_with_population_path (mouse, spont, gspont, glayer, player) :
    path = f"{base_path}/cofiring/pearson_correlation/{get_pearson_correlations_of_cofiring_with_population_filename(mouse, spont, gspont, glayer, player)}.feather"
    return path

def get_pearson_correlations_of_cofiring_with_population (mouse, spont, gspont, glayer, player) :
    path = get_pearson_correlations_of_cofiring_with_population_path (mouse, spont, gspont, glayer, player)
    return pd.read_feather(path)


def get_pearson_correlations_of_frps_with_population_path (mouse, spont, player) :
    at = get_activity_text(spont)
    
    path = f"{base_path}/cofiring/pearson_correlation/mouse{mouse}_pearson_corr_between_frps_cofiring_and_{at}_{player}_population_firing.feather"
    return path

def get_pearson_correlations_of_frps_with_population (mouse, spont, player) :
    path = get_pearson_correlations_of_frps_with_population_path (mouse, spont, player)
    return pd.read_feather(path)

#sta can be 'vector', 'sem', 'std' or 'lag_values' or 'signal_frames_counts'
def get_pop_eta_path (mouse, spont, sta='vector', area='V1') :
    at = get_activity_text(spont)
    areat = get_areat(area)
    return f'{base_path}/eta/population_firing/spike_onset/mouse{mouse}_{at}_population_eta_{sta}{areat}.feather'

def get_pop_eta (mouse, spont, sta='vector', area='V1') :
    return pd.read_feather(get_pop_eta_path (mouse, spont, sta, area))

def get_aod_by_frame_path (mouse) :
    path = f'{base_path}/absolute_orientation_difference/m{mouse}_aod_of_each_neuron_each_frame.feather'
    return path

def get_aod_by_frame (mouse) :
    return pd.read_feather(get_aod_by_frame_path(mouse))

def get_aod_by_stimulus_freashness_path () :
    path = f'{base_path}/absolut/aod_by_stimulus_freshness.pickle'
    return path

def get_aod_by_stimulus_freashness () :
    return pickle.load(open(get_aod_by_stimulus_freashness_path(), 'rb'))


def get_rare_frames (mouse, spont, layer) :
    
    if layer == 'L4' :
        if not spont :
            path = f'{base_path}/rare_frames/M{mouse}_gc_L4-L23_rareframes_zthresh-3.feather'
            rare_frames = pickle.load(open(path, 'rb'))
        else :
            path = f'/home/brozi/L4_and_L23_rare_frames_separately_spontaneous.pickle'
            rare_frames = pickle.load(open(path, 'rb'))[f'M{mouse}']['L4']

    elif layer == 'L23': 
        if not spont :
            path = f'/home/brozi/Stimuli/Rare_Frames/stimuli_L23_group_rare_frames_zthresh=2.pickle'
            rare_frames = pickle.load(open(path, 'rb'))[f'M{mouse}']
        else : 
            path = f'/home/brozi/L4_and_L23_rare_frames_separately_spontaneous.pickle'
            rare_frames = pickle.load(open(path, 'rb'))[f'M{mouse}']['L23']

    elif layer == 'L4_and_L23':
        l4frames = get_rare_frames(mouse, spont, 'L4')
        l23frames = get_rare_frames(mouse, spont, 'L23')
        neurons = list(set(l4frames.keys()) & set(l23frames.keys()))
        rare_frames = {n: list(set(l4frames[n]) & set(l23frames[n])) for n in neurons}
    
    return rare_frames

def get_class_of_angles (angle) :
    return int((angle+22.5/2)//22.5)

def get_angles_per_frame (mouse) :
    spiketrains = pd.read_feather(f'{base_path}/spiketrains/m{mouse}_(NOFILTER)_EVENTOGRAMS_with_angles.feather')[['2pf','ang']]
    spiketrains['class'] = ((spiketrains['ang']+22.5/2)//22.5).astype(int)
    return spiketrains


def get_directions_path (mouse) :
    #return f'data/mouse{mouse}/spiketrains/m{mouse}_MonetDirectionPerSegment.npy'
    return f'{base_path}/angles/monet_m{mice[mouse]}_tolias_angle_labels_actual.npy'

def get_directions (mouse): 
    return np.load(get_directions_path(str(mouse)))

def get_doc_path (mouse, spont, srcarea, srclayer, destarea, destlayer) :
    path = f'{base_path}/doc/mouse{mouse}_{get_activity_text(spont)}_{srcarea},{srclayer}-{destarea},{destlayer}.feather'
    return path

def get_doc (mouse, spont, srcarea, srclayer, destarea, destlayer) :
    if srclayer == destlayer == 'L23' :
        path = get_doc_path(mouse, spont, srcarea, 'L2', destarea, 'L2')
        d2 = pd.read_feather(path)
        path = get_doc_path(mouse, spont, srcarea, 'L3', destarea, 'L3')
        d3 = pd.read_feather(path)
        return pd.concat([d2, d3], ignore_index=True)
    
    elif 'L23' == destlayer :
        path = get_doc_path(mouse, spont, srcarea, srclayer, destarea, 'L2')
        d2 = pd.read_feather(path)
        path = get_doc_path(mouse, spont, srcarea, srclayer, destarea, 'L3')
        d3 = pd.read_feather(path)
        return pd.concat([d2, d3], ignore_index=True)

    else :
        path = get_doc_path(mouse, spont, srcarea, srclayer, destarea, destlayer)
        return pd.read_feather(path)

def get_firing_rate_path (mouse, spont, threshold=None) :
    tt=f'gt_{threshold}' if threshold is not None else '' 
    path = f'{base_path}/firing_rate/mouse'+mice[mouse]+f'{get_activity_text(spont)}_firing_rates{tt}.feather'
    return path

def get_firing_rate (mouse, spont, threshold=None) :
    path = get_firing_rate_path(mouse, spont, threshold)
    return pd.read_feather(path)

def get_clustering_coefficient_path (mouse, spont, sourcelayer, destlayer) :
    st = '' if spont else '_stimuli'
    path = f'{base_path}/clustering_coefficient/mouse{mice[mouse]}_cc_V1,{sourcelayer}-V1,{destlayer}{st}.csv' 
    #f'data/clustering_coefficient/mouse{mice[mouse]}_{get_activity_text(spont)}_clustering_coefficient.feather'
    return path

def get_clustering_coefficient (mouse, spont, sourcelayer, destlayer) :
    if sourcelayer == destlayer == 'L23' :
        path = get_clustering_coefficient_path(mouse, spont, 'L2', 'L2')
        d2 = pd.read_csv(path)
        path = get_clustering_coefficient_path(mouse, spont, 'L3', 'L3')
        d3 = pd.read_csv(path)
        return pd.concat([d2, d3], ignore_index=True)
    elif 'L23' == destlayer :
        path = get_clustering_coefficient_path(mouse, spont, sourcelayer, 'L2')
        d2 = pd.read_csv(path)
        path = get_clustering_coefficient_path(mouse, spont, sourcelayer, 'L3')
        d3 = pd.read_csv(path)
        return pd.concat([d2, d3], ignore_index=True)
    else :
        return pd.read_csv(get_clustering_coefficient_path(mouse, spont, sourcelayer, destlayer))
    
def get_group_activity_quantile_normalized_path (mouse, layer, rop, x) :
    if rop is None :
        ropt = ''
    else:
        if rop : ropt='_rop'
        else : ropt='_nonrop'
        
    path = f'{base_path}/stimulus_prediction/cardinal_neuron_activity/m{mouse}_x{x}_{layer}{ropt}.feather'
    return path
    
def get_group_activity_quantile_normalized (mouse, layer, rop, x) :
    path = get_group_activity_quantile_normalized_path(mouse, layer, rop, x)
    return pd.read_feather(path)

def get_ROPs_that_participate_in_active_group_per_frame_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_ROPs_that_participate_in_active_group_per_frame.pickle'
    return path

def get_ROPs_that_participate_in_active_group_per_frame (mouse) :
    return pickle.load(open(get_ROPs_that_participate_in_active_group_per_frame_path(mouse), 'rb'))

def get_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_majority_orientation_preference_of_ROPs_that_participate_in_active_group_per_frame.feather'
    return path

def get_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame (mouse) :
    path = get_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path(mouse)
    return pd.read_feather(path)


def get_predicted_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_predicted_majority_orientation_preference_of_ROPs_that_participate_in_active_group_per_frame.feather'
    return path

def get_predicted_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame (mouse) :
    path = get_predicted_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path(mouse)
    return pd.read_feather(path)


def get_segment_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_segment_majority_orientation_preference_of_ROPs_that_participate_in_active_group_per_frame.feather'
    return path

def get_segment_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame (mouse) :
    path = get_segment_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path(mouse)
    return pd.read_feather(path)

def get_predicted_segment_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_predicted_segment_majority_orientation_preference_of_ROPs_that_participate_in_active_group_per_frame.feather'
    return path

def get_predicted_segment_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame (mouse) :
    path = get_predicted_segment_majority_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path(mouse)
    return pd.read_feather(path)

def get_predicted_segment_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_predicted_segment_orientation_preference_of_ROPs_that_participate_in_active_group_per_frame.feather'
    return path

def get_predicted_segment_orientation_preference_ROPs_that_participate_in_active_group_per_frame (mouse) :
    path = get_predicted_segment_orientation_preference_ROPs_that_participate_in_active_group_per_frame_path(mouse)
    return pickle.load(open(path, 'rb'))



def get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_active_frames_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_aod_for_ROPs_that_participate_in_active_group_per_frame.pickle'
    return path

def get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_active_frames (mouse) :
    path = get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_active_frames_path(mouse)
    return pickle.load(open(path, 'rb'))

def get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_rest_frames_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_aod_for_ROPs_that_participate_in_active_group_per_frame_for_rest_frames.pickle'
    return path

def get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_rest_frames (mouse) :
    path = get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_rest_frames_path(mouse)
    return pickle.load(open(path, 'rb'))

def get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments.pickle'
    return path

def get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments (mouse) :
    path = get_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments_path(mouse)
    return pickle.load(open(path, 'rb'))


def get_predicted_majority_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_predicted_majority_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments.pickle'
    return path

def get_predicted_majority_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments (mouse) :
    path = get_predicted_majority_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments_path(mouse)
    return pickle.load(open(path, 'rb'))


def get_predicted_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments_path (mouse) :
    path = f'{base_path}/rare_frames/rops_that_spike_and_participate_in_active_groups/M{mouse}_predicted_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments.pickle'
    return path

def get_predicted_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments (mouse) :
    path = get_predicted_aod_for_ROPs_that_participate_in_active_group_per_frame_for_segments_path(mouse)
    return pickle.load(open(path, 'rb'))

def get_spatial_spread_path (spont) :
    at = get_activity_text(spont)
    path = f'{base_path}/euclidean_distance/spatial_spread_{at}.pickle'
    return path

def get_spatial_spread (spont) :
    path = get_spatial_spread_path(spont)
    return pickle.load(open(path, 'rb'))

def flair_decoder_ibm (flair) :
    regular, layer, spont = flair.split('_')
    
    snn_type = None

    if regular == 'regular' :
        regular = True
    elif 'snn' in regular :
        snn_type = regular
        regular = False

    else :
        regular = None

    if spont == 'spont' :
        spont = True
    elif spont == 'stimuli' :
        spont = False
    else :
        spont = None
    
    if layer not in ['L23','L4', 'C1', 'C2', 'H1', 'H2', 'H3'] :
        layer = None

    #print('Error with flair')

    return regular, spont, layer, snn_type

def flair_decoder_pap (flair) :
    flaircomponents = flair.split('_')
    spont, flaircomponents = flaircomponents[0], flaircomponents[1:]

    layerinfos = []
    if len(flaircomponents) == 0 :
        pass

    elif len(flaircomponents) == 1 :
        layerinfo = flaircomponents[0]
        layerinfos.append(layerinfo)

    elif len(flaircomponents) == 2 :
        srclayerinfo = flaircomponents[0]
        destlayerinfo = flaircomponents[1]
        layerinfos.append(srclayerinfo)
        layerinfos.append(destlayerinfo)

    layers = []
    rops = []
    for layerinfo in layerinfos :
        if ',' in layerinfo :
            layer, rop = layerinfo.split(',')
            rop = True if rop == 'rop' else False
        else :
            layer = layerinfo
            rop = None
        
        layers.append(layer)
        rops.append(rop)
    
    if len(layers) == 1 :
        layers.append(layers[0])
        rops.append(rops[0]) 

    if spont == 'spont' :
        spont = True
    elif spont == 'stimuli' :
        spont = False
    else :
        spont = None
    
    return spont, layers, rops

def flair_encoder (regular, spont, layer, snn_type) :
    if regular is not None and regular :
        regular = 'regular'
    else : 
        if snn_type is not None :
            regular = snn_type
    
    if spont is None :
        spont = ''
    else :
        if spont :
            spont = 'spont'
        else :
            spont = 'stimuli'

    return f'{regular}_{spont}_{layer}_{snn_type}'

def get_spiketrains_with_flair_path (mouse, flair) :
    regular, spont, layer, snn_type = flair_decoder(flair)
    path = f'{base_path}/general_spiketrains/{flair}.feather'
    print(path)
    return path

def load_spiketrains_with_flair (mouse, flair) :
    regular, spont, layer, snn_type = flair_decoder(flair)

    if regular :
        return get_spiketrains(mouse, spont)
    
    else :
        path = get_spiketrains_with_flair_path(mouse, flair)
        return pd.read_feather(path)



def get_firing_rates_with_flair_path (flair) :
    return f'{base_path}/general_metrics/firing_rate/{flair}.feather'

def get_firing_rates_with_flair_path_pap (mouse, flair) :
    return f'{base_path}/general_metrics/firing_rate/m{mouse}_{flair}.feather'

def load_firing_rates_with_flair (mouse, flair) :
    regular, spont, layer, snn_type = flair_decoder_ibm(flair)

    if regular :
        return get_firing_rate(mouse, spont)

    else :
        path = get_firing_rates_with_flair_path(flair)
        return pd.read_feather(path)    

def load_firing_rates_with_flair_pap (mouse, flair) :
    #mouse, spont, layers, rop = flair_decoder_pap(flair)
    path = get_firing_rates_with_flair_path_pap(mouse, flair)
    return pd.read_feather(path)



def get_filtered_neurons_with_flair_path (flair) :
    return f'{base_path}/general_metrics/filtered_neurons/{flair}.pickle'

def get_filtered_neurons_with_flair (flair) :
    return pickle.load(open(get_filtered_neurons_with_flair_path(flair), 'rb'))

def get_preference_of_neurons_with_flair_path (flair, cardinal=False) :
    cardinalt = '_cardinal' if cardinal else ''
    return f'{base_path}/general_metrics/preference/{flair}{cardinalt}.pickle'

def get_preference_of_neurons_with_flair (flair, cardinal=False) :
    return pickle.load(open(get_preference_of_neurons_with_flair_path(flair, cardinal), 'rb'))

def get_firing_rates_by_direction_with_flair_path (flair, cardinal=False, hz=True) :
    cardinalt = '_cardinal' if cardinal else ''
    
    if hz : hzt = '_hz' 
    else : hzt = ''

    return f'{base_path}/general_metrics/firing_rate/by_direction/{flair}{cardinalt}{hzt}.feather'

def get_firing_rates_by_direction_with_flair (flair, cardinal, hz=True) :
    return pd.read_feather(get_firing_rates_by_direction_with_flair_path(flair, cardinal, hz))

def load_filtered_neurons_with_flair (mouse, flair) :
    regular, spont, layer, snn_type = flair_decoder(flair)

    if regular :
        return get_neurons_of_layer(mouse, spont, layer)

    else :
        path = get_filtered_neurons_with_flair_path(flair)
        return pickle.load(open(path, 'rb')).astype(int).values.tolist()

def get_sttc_with_flair_only_path () :
    return f'{base_path}/general_sttc/STTC/Results'

def get_sttc_with_flair_filename_base (flair) :
    return f'{flair}'

def get_sttc_with_flair_filename (flair, shifts) :
    return f'{get_sttc_with_flair_filename_base(flair)}_{shifts}-shifts_0-dt_pairs.feather'

def get_sttc_with_flair_path (flair, shifts) :
    return get_sttc_with_flair_only_path() + '/' + get_sttc_with_flair_filename(flair, shifts)

def load_sttc_with_flair (mouse, flair, shifts) :
    regular, spont, layer, snn_type = flair_decoder(flair)

    if regular :
        return get_sttc(mouse, spont)

    else :
        path = get_sttc_with_flair_path(flair, shifts)
        return pd.read_feather(path)    

def get_intralayer_ndoc_with_flair_path (flair, correlated, shifts) :
    if not correlated : ct = '_anticorrelated'
    else : ct = ''

    st = f'_shifts={shifts}' if shifts!=500 else ''
    return f'{base_path}/general_metrics/docs/{flair}{ct}{st}.feather'

def load_intralayer_ndoc_with_flair (mouse, flair, correlated, shifts) :
    regular, spont, layer, snn_type = flair_decoder(flair)

    if regular :
        if not correlated : print('======= Anticorrelated DOC does not exist! =======')
        return get_doc(mouse, spont, 'V1', layer, 'V1', layer)

    else :
        path = get_intralayer_ndoc_with_flair_path(flair, correlated, shifts)
        return pd.read_feather(path) 

def get_ndoc_with_flair_path (mouse, flair) :
    return f'{base_path}/general_metrics/docs/m{mouse}_{flair}.feather'

def get_ndoc_with_flair (mouse, flair) :
    return pd.read_feather(get_ndoc_with_flair_path(mouse, flair))

def get_cc_with_flair_path (mouse, flair) :
    return f'{base_path}/general_metrics/cc/m{mouse}_{flair}.feather'

def get_cc_with_flair (mouse, flair) :
    return pd.read_feather(get_cc_with_flair_path(mouse, flair))

def get_stateless_frames_flair (layer, epoch=None, less_neurons=False) :
    ln = '-less-neurons' if less_neurons else ''

    if epoch is None or epoch == 2 :
        return f'stateless-cnn-snn{ln}_{layer}_stimuli'
    
    

def get_stateless_segments_flair (layer, epoch=None) :
    if epoch is None or epoch == 2 :
        return f'stateless-cnn-snn-double-dataset-segments_{layer}_stimuli'

    elif epoch==0 :
        return f'stateless-cnn-snn-double-dataset-segments-untrained_{layer}_stimuli'


def get_reduced_dimensionality_with_flair_path(flair, method, standardize, wise, filtered_neurons=True, onehot=False) :
    '''
    Parameters
    ----------
    wise : str, Direction in which the dimensionality is reduced. 
        Can be 'framewise' or 'neuronwise'
    '''
    
    st = '_standardized' if standardize else ''
    fnt = '_filtered_neurons' if filtered_neurons else ''
    oht = '_onehot' if onehot else ''
    return f'/home/psilou/data/reduced_dimensionality/spiketrains/reduce_{wise}/{flair}_{method}{st}{fnt}{oht}.feather'


def get_reduced_dimensionality_with_flair(flair, method, standardize, wise, filtered_neurons, onehot) :
    '''
    Parameters
    ----------
    wise : str, Direction in which the dimensionality is reduced. 
        Can be 'framewise' or 'neuronwise'
    '''
    return pd.read_feather(get_reduced_dimensionality_with_flair_path(flair, method, standardize, wise, filtered_neurons, onehot))




def etas_with_flair_path(flair, segment) :
    if segment :
        stxt = '_per_segment'
    else :
        stxt = '_per_frame'

    return f'{base_path}/general_metrics/eta/{flair}_{stxt}.pickle'

def etas_with_flair (flair, segment) :
    etas = pickle.load(open(etas_with_flair_path(flair, segment), 'rb'))
    return etas['means'], etas['sems'], etas['stds']

def peak_dip_analysis_with_flair_path (flair, segment) :
    if segment :
        stxt = '_per_segment'
    else :
        stxt = '_per_frame'

    return f'{base_path}/general_metrics/peak_dip_analysis/{flair}{stxt}.pickle'

def peak_dip_analysis_with_flair (flair, segment) :
    peak_dip_analysis = pickle.load(open(peak_dip_analysis_with_flair_path(flair, segment), 'rb'))
    return peak_dip_analysis['zscores'], peak_dip_analysis['peaks'], peak_dip_analysis['dips']

def tuples_with_fixed_neuron_cnt_path (target_ncnt, input_image_size) :
    return f'{base_path}/general_metrics/model_parameters/tuples_with_neuron_count={target_ncnt}_given_input_size={input_image_size}.pkl'
    
def load_tuples_with_fixed_neuron_cnt (target_ncnt, input_image_size) :
    path = tuples_with_fixed_neuron_cnt_path(target_ncnt, input_image_size)
    tuples = pickle.load(open(path, 'rb'))
    return tuples

def get_wavelike_scores_for_all_params_path (target_ncnt, input_image_size, i) :
    path = f'{base_path}/general_metrics/wavelike_score/wavelike_scores_{target_ncnt}_{input_image_size}_iteration={i}.pkl'
    return path

def get_wavelike_scores_for_all_params (target_ncnt, input_image_size, i) :
    path = get_wavelike_scores_for_all_params_path (target_ncnt, input_image_size, i)
    return pickle.load(open(path, 'rb'))

def get_frps (mouse) :
    return pd.read_feather(f'{base_path}/frps/mouse{mouse}_stimuli_frps.feather')