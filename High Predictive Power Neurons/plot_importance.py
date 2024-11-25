import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.getcwd()+'/code')
import utils.data_explorer as de

# OG Hpps 
allhpps = {
    3 : ['V1620', 'V1713', 'V1937', 'V2205', 'V2275', 'V2600', 'V2647', 'V3411', 'V3451', 'V3594', 'V4316', 'V4380', 'V4451', 'V4475', 'V4631', 'V4724', 'V4845', 'V4904', 'V4928', 'V4933', 'V4981', 'V5020', 'V5035', 'V5274', 'V5583', 'V5600', 'V5639', 'V5699', 'V5825', 'V5835', 'V5898', 'V5968', 'V6268', 'V8270', 'V8622', 'V8783', 'V8920'],
    4 : ['V166', 'V195', 'V223', 'V299', 'V306', 'V341', 'V351', 'V357', 'V369', 'V422', 'V448', 'V483', 'V486', 'V724', 'V960', 'V977', 'V1055', 'V1067', 'V1070', 'V1077', 'V1792', 'V1880', 'V1997', 'V2043', 'V2077', 'V2164', 'V2423', 'V3293', 'V3449', 'V3460', 'V3517', 'V3768', 'V4994', 'V5052', 'V5172', 'V5326', 'V5376', 'V5381', 'V5465', 'V5528', 'V5535', 'V5549', 'V5621', 'V5625', 'V5664', 'V6643', 'V6660', 'V6685', 'V6687', 'V6766', 'V6787', 'V6921', 'V7039', 'V7163'],
    5 : ['V284', 'V645', 'V686', 'V796', 'V810', 'V866', 'V895', 'V932', 'V989', 'V1150', 'V1906', 'V1916', 'V1945', 'V1990', 'V2135', 'V2311', 'V2337', 'V2362', 'V2388', 'V3263', 'V3276', 'V3465', 'V3527', 'V3554', 'V3557', 'V3563', 'V3572', 'V3580', 'V3633', 'V3662', 'V3728', 'V3733', 'V3776', 'V3905', 'V4039', 'V4691', 'V4739', 'V4844', 'V4854', 'V5071', 'V5107', 'V5296', 'V5333', 'V6906', 'V6948', 'V6949', 'V7000', 'V7003', 'V7093', 'V7210', 'V7266', 'V7294', 'V7389', 'V7422', 'V7464', 'V7492', 'V7607', 'V7839', 'V8080', 'V8108', 'V8157', 'V8159', 'V8215', 'V8230', 'V8314', 'V8323', 'V8367'],
    6 : ['V404', 'V466', 'V592', 'V648', 'V761', 'V1013', 'V1224', 'V1353', 'V1408', 'V1438', 'V1520', 'V1696', 'V1708', 'V1811', 'V1908', 'V1940', 'V1967', 'V1990', 'V2000', 'V2044', 'V2120', 'V2133', 'V2146', 'V2153', 'V2157', 'V2224', 'V2457', 'V2830', 'V2890', 'V3041', 'V3068', 'V3079', 'V3124', 'V3140', 'V4049', 'V5404', 'V5806', 'V6137', 'V6144', 'V6754', 'V6768', 'V6829', 'V6874', 'V7019'],
    7 : ['V303', 'V387', 'V504', 'V531', 'V748', 'V896', 'V899', 'V936', 'V1311', 'V2154', 'V2160', 'V2191', 'V2283', 'V2318', 'V2373', 'V2376', 'V2441', 'V2556', 'V2620', 'V2680', 'V2752', 'V2813', 'V2873', 'V3695', 'V3752', 'V3761', 'V3881', 'V4043', 'V4116', 'V4131', 'V4796', 'V5372', 'V5421', 'V5927', 'V5947', 'V5949', 'V5951', 'V6005', 'V6009', 'V6092', 'V6094', 'V6183', 'V6258', 'V6329', 'V6372', 'V6382', 'V6395', 'V6585', 'V6612', 'V7672', 'V7707', 'V7734']
}


# Load and process data
neurons = pd.read_feather('/home/psilou/code/misc/tony/feature_importances.feather')


# flip sign becauase original calculation was fliped 
#neurons['importance'] = -neurons['importance']
# take absolute 
#neurons['importance'] = neurons['importance'].abs()
# normalize 
neurons['importance'] = (neurons['importance'] - neurons['importance'].min()) / (neurons['importance'].max() - neurons['importance'].min())

neurons = neurons.sort_values(by=['importance'])

importance_mean = neurons['importance'].mean()
importance_std = neurons['importance'].std()
print(f'Importance mean: {importance_mean:.2f} std: {importance_std:.2f}')

firing_rates = de.get_firing_rate('3', False, None)
firing_rates = firing_rates[firing_rates['neuron_id'].astype(str).isin(de.unst_list(neurons.index))]
#firing_rates = firing_rates.set_index('neuron_id').loc[de.unst_list(neurons.index)].reset_index()

# Normalize the importance values
#neurons['importance'] = neurons['importance'] - neurons['importance'].min()  # Shift to start at 0
#neurons['importance'] /= neurons['importance'].max()
#neurons['importance'] = neurons['importance'] * 2 - 1  # Scale to [0, 1]


neurons_sorted = neurons
neurons_sorted = neurons_sorted.reset_index()
neurons_sorted = neurons_sorted.rename(columns={'index': 'neuron_id'})

oghppsorted = neurons_sorted[neurons_sorted['neuron_id'].astype(str).isin(allhpps[3])]

# Plot normalized importance
plt.figure(figsize=(10, 6))
plt.plot(neurons_sorted.index, neurons_sorted['importance'], marker='o', linestyle='-', color='b', label='Normalized Importance')
print(oghppsorted)

#plot mean and std of importance 
plt.axhline(importance_mean, color='r', linestyle='--', label='Mean Importance', linewidth=2)
plt.axhline(importance_mean + importance_std, color='g', linestyle='--', label='Mean + Std Importance')
plt.axhline(importance_mean - importance_std, color='g', linestyle='--', label='Mean - Std Importance')

# plot a line where the 37 most important neurons are
plt.axvline(len(neurons_sorted)-37, color='black', linestyle='--', label='37 Most Important Neurons')

important_neurons = neurons_sorted.iloc[-37:]['neuron_id'].tolist()
textstr = '\n'.join(important_neurons)


plt.plot(
    oghppsorted.index,
    oghppsorted['importance'], 
    marker='o', 
    lw=0, 
    color='red', 
    label='Original HPPs',
    markersize=8,
    markeredgewidth=2
)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Neuron (sorted by importance)', fontsize=14, fontweight='bold')
plt.ylabel('Normalized Importance', fontsize=14, fontweight='bold')
plt.title('Normalized Importance Per Neuron', fontsize=16, fontweight='bold')

# print the position of all hpps [3] in the sorted list
hpps = de.unst_list(allhpps[3])
hpps = neurons_sorted[neurons_sorted.index.astype(str).isin(hpps)]
#print(hpps)

# Create a second y-axis to plot the firing rates
'''
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(firing_rates.index, firing_rates['firing_rate'], marker='x', lw=0, color='r', label='Firing Rate')
ax2.set_ylabel('Firing Rate')
ax2.legend(loc='upper right')
'''
# Add text box with the names of the 37 most important neurons
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
#         verticalalignment='bottom', bbox=props, wrap=True)

# Set font properties for the plot
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Neuron (sorted by importance)', fontsize=14, fontweight='bold')
plt.ylabel('Normalized Importance', fontsize=14, fontweight='bold')
plt.title('Normalized Importance Per Neuron', fontsize=16, fontweight='bold')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(fontsize=18, title_fontsize='14', frameon=True, fancybox=True, shadow=True, borderpad=1, loc='best', prop={'weight': 'bold'})
plt.tight_layout()
plotpath = f'/home/psilou/code/misc/tony/plot_importances.png'
plt.savefig(plotpath)
print(plotpath)

oghpps = firing_rates[firing_rates['neuron_id'].astype(str).isin(de.unst_list(allhpps[3]))]['firing_rate']
#print(oghpps.mean(), oghpps.std())
print(f'OG HPPs: {oghpps.mean():.2f} {oghpps.std():.2f}')

ogrest = de.get_generic_filtered_neurons('3', False, 'V1', 'L234', True)
ogrest = list(set(pd.Series(ogrest).astype(str).values.tolist()) - set(de.unst_list(allhpps[3])))
ogrest = firing_rates[firing_rates['neuron_id'].astype(str).isin(de.unst_list(ogrest))]['firing_rate']

print(f'OG Rest: {ogrest.mean():.2f} {ogrest.std():.2f}')


ourhpps = firing_rates[firing_rates['neuron_id'].astype(str).isin(de.unst_list((neurons_sorted.iloc[:37])['neuron_id']))]['firing_rate']
our_hpp_mean = ourhpps.mean()
print(f'Our HPPs: {our_hpp_mean:.2f} {ourhpps.std():.2f}')
#print(ourhpps.mean(), ourhpps.std())

restneurons = neurons.iloc[37:].index
ourrest = firing_rates[firing_rates['neuron_id'].astype(str).isin(de.unst_list(restneurons))]['firing_rate']
print(f'Our Rest: {ourrest.mean():.2f} {ourrest.std():.2f}')


print('common hpps:', len(set(allhpps[3]) & set(neurons_sorted.iloc[:37]['neuron_id'])))
