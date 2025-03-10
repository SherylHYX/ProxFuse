import os

import numpy as np
import pandas as pd
import networkx as nx

data = pd.read_csv('../data/male_behav_data_20250310.csv')
data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)

# define some constants, unlearned weights with same 0.1 added values
edge_weights_dict = {'party_party': 1.0, # in the party of the same individual (denoted FM) at the same time
                     'party_prox5': 1.0, # prox5 and in the party of FM, respectively, at the same time
                     'party_prox2': 1.0, # prox2 and in the party of FM, respectively, at the same time
                     'party_focal': 1.0, # in the party of FM
                     'prox5_prox5': 5.7,  # prox5 to FM at the same time
                     'prox2_prox5': 7.0,  # prox2 and prox5 to FM, respectively, at the same time
                     'prox5_focal': 8.6, # prox5 to FM
                     'prox2_prox2': 10.6, #prox2 to FM at the same time
                     'prox2_focal': 10.6, # prox2 to FM
                     'grooming': 10.7, # grooming with FM (adding to prox2, as this must already be prox2 to FM)
                     'consec_add': 0.0, # added value to an edge which previously occurred, to be constructed as a separte graph with the above types
                    }
consec_addition_parameter = edge_weights_dict['consec_add']
# Convert all values to int
edge_weights_dict = {key: int(value) for key, value in edge_weights_dict.items()} # int first, then divide by 10
edge_name_list = [name for name in edge_weights_dict.keys() if len(name)==11] + ['grooming']
edge_weights_values = edge_weights_dict.values()
column_name_list = ['code', 'prox2', 'prox5', 'party']

# define fake names
fake_names = {'estrousfemale', 'adultfemale', 'many', 'adolf', 'adultfemale', \
    'juvm', 'unknown party association', 'unknown', 'juvf', 'adolm'}
# to prepare name strings
name_str = list(np.load('../data/chimps_names.npy'))
name_str_focal = list(np.load('../data/chimps_names_focal.npy'))
name_str_non_focal = list(np.load('../data/chimps_names_non_focal.npy'))
name_str_non_focal_male = list(np.load('../data/chimps_names_non_focal_male.npy'))
name_str_non_focal_female = list(np.load('../data/chimps_names_non_focal_female.npy'))
num_chimps = len(name_str)
num_chimps_focal = len(name_str_focal)
num_chimps_non_focal = len(name_str_non_focal)
num_chimps_non_focal_male = len(name_str_non_focal_male)
num_chimps_non_focal_female = len(name_str_non_focal_female)
print('There are a total of {} chimps, {} of which are focal observations while the remaining {} are non-focal observations. \
    Among non-focal observations, {} are male and {} are female.'.format(num_chimps, num_chimps_focal, num_chimps_non_focal, \
        num_chimps_non_focal_male, num_chimps_non_focal_female))

# combine the single type graphs
for year in range(1998, 2023):
    for month in range(1, 13):
        full_array = 0
        for i, edge_key in enumerate(edge_name_list):
            graph_weight = edge_weights_dict[edge_key]
            curr_adj = np.load(os.path.join(os.path.dirname(os.path.realpath(
                __file__)), '../data/graphs_without_hardcoded_parameters/'+edge_key+'_array_'+str(year)+'_'+str(month)+'.npy'))
            # consec addition to be added
            curr_consec_adj = np.load(os.path.join(os.path.dirname(os.path.realpath(
                __file__)), '../data/graphs_without_hardcoded_parameters/'+edge_key+'_consec_array_'+str(year)+'_'+str(month)+'.npy'))
            full_array += graph_weight * (curr_adj + consec_addition_parameter * curr_consec_adj)
        np.save('../data/learned_graphs/full_array_'+str(year)+'_'+str(month), full_array)