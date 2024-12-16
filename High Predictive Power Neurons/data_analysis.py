import pandas as pd
results_path = 'E:/Eleftheria/workspace/Computational-Neuroscience-Research/High Predictive Power Neurons'
maxiter=100
feature_weights_matrix = pd.read_feather(f'{results_path}/feature_weight_matrix_iters={maxiter}.feather')
feature_keep_matrix = (feature_weights_matrix != 0).astype(int)
print("Features kept:")
print(feature_keep_matrix)
print('Perc of features kept per class:')
print(100*feature_keep_matrix.mean(axis=1))
print('Perc of classes that kept the feature:')
print(100*feature_keep_matrix.mean(axis=0))
print()