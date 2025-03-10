import os

import numpy as np
import pandas as pd

IsRegenerate = False # parameter to regenerate data
data = pd.read_csv('../data/male_behav_data_20250310.csv')
data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)

# define some constants
edge_weights_dict = {'party_party': 1, # in the party of the same individual (denoted FM) at the same time
                     'party_prox5': 1, # prox5 and in the party of FM, respectively, at the same time
                     'party_prox2': 1, # prox2 and in the party of FM, respectively, at the same time
                     'party_focal': 1, # in the party of FM
                     'prox5_prox5': 1,  # prox5 to FM at the same time
                     'prox2_prox5': 1,  # prox2 and prox5 to FM, respectively, at the same time
                     'prox5_focal': 1, # prox5 to FM
                     'prox2_prox2': 1, #prox2 to FM at the same time
                     'prox2_focal': 1, # prox2 to FM
                     'grooming': 1, # grooming with FM (code individual), in additional to prox2
                     'consec_add': 1, # added value to an edge which previously occurred, to be constructed as a separte graph with the above types
                    } # unit edge weights for now except consecutive additions for the same type
# Convert all values to int
edge_weights_dict = {key: int(value) for key, value in edge_weights_dict.items()} # int first, then divide by 10
edge_name_list = [name for name in edge_weights_dict.keys() if len(name) == 11] # grooming considered separately
edge_weights_values = edge_weights_dict.values()
column_name_list = ['code', 'prox2', 'prox5', 'party']

# define fake names
fake_names = {'estrousfemale', 'adultfemale', 'many', 'adolf', 'adultfemale', \
    'juvm', 'unknown party association', 'unknown', 'juvf', 'adolm'}
# to prepare name strings
if os.path.exists('../data/chimps_names_focal.npy') and not IsRegenerate:
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
else:
    name_str_focal = list(set(data['code']))
    name_str_non_focal = []
    for column_name in column_name_list:
        df = pd.DataFrame([])
        df[column_name] = (data[column_name][data[column_name].notnull()]).str.split(',')
        df = df.explode(column_name)

        # Remove leading and trailing whitespaces from names
        df[column_name] = df[column_name].str.strip()

        # Get the set of all unique names
        unique_names = set(df[column_name].unique())

        # perform removal
        while("" in unique_names):
            unique_names.remove("")
            
        name_str_non_focal.extend(list(unique_names))    
        # Print the set of unique names
        print(unique_names)
    name_str_non_focal = list((set(name_str_non_focal).difference(name_str_focal)).difference(fake_names))
    name_str_non_focal_male = [name for name in name_str_non_focal if len(name)==2]
    name_str_non_focal_female = [name for name in name_str_non_focal if len(name)>2]
    name_str_non_focal = name_str_non_focal_male + name_str_non_focal_female
    name_str = name_str_focal + name_str_non_focal
    np.save('../data/chimps_names.npy', name_str)
    np.save('../data/chimps_names_focal.npy', name_str_focal)
    np.save('../data/chimps_names_non_focal.npy', name_str_non_focal)
    np.save('../data/chimps_names_non_focal_male.npy', name_str_non_focal_male)
    np.save('../data/chimps_names_non_focal_female.npy', name_str_non_focal_female)
    
    # Create a DataFrame with indices and names
    df = pd.DataFrame(name_str, columns=['Name'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Index'}, inplace=True)

    # Save the DataFrame to an Excel file
    excel_file_path = '../data/names_and_indices.xlsx'  # Specify your desired path
    df.to_excel(excel_file_path, index=False, engine='openpyxl')

    print(f"DataFrame saved to {excel_file_path}.")
    num_chimps = len(name_str)
    num_chimps_focal = len(name_str_focal)
    num_chimps_non_focal = len(name_str_non_focal)
    num_chimps_non_focal_male = len(name_str_non_focal_male)
    num_chimps_non_focal_female = len(name_str_non_focal_female)
    print('There are a total of {} chimps, {} of which are focal observations while the remaining {} are non-focal observations. \
        Among non-focal observations, {} are male and {} are female.'.format(num_chimps, num_chimps_focal, num_chimps_non_focal, \
            num_chimps_non_focal_male, num_chimps_non_focal_female))

if not os.path.exists('../data/graphs_without_hardcoded_parameters/grooming_array_2006_8.npy') and IsRegenerate:
    # prepare monthly graphs
    grooming_weight = edge_weights_dict['grooming']
    for year in range(1998, 2023):
        for month in range(1, 13):
            mask = (data['date'].dt.month==month) & (data['date'].dt.year==year)
            content_of_interest = data.loc[mask].copy()
            full_array_dict = {}
            full_array_consec_dict = {} # for consecutive occurrences
            curr_grooming_array = np.zeros((num_chimps, num_chimps)) # current array for a certain date, initialized
            full_grooming_array = np.zeros((num_chimps, num_chimps)) # full array
            curr_grooming_consec_array = np.zeros((num_chimps, num_chimps)) # current consecutive array for a certain date, initialized
            full_grooming_consec_array = np.zeros((num_chimps, num_chimps)) # full consecutive occurrence array

            for i in range(len(edge_name_list)):
                full_array_dict[edge_name_list[i]] = np.zeros((num_chimps, num_chimps)) # current array for this date, reinitialized
                full_array_consec_dict[edge_name_list[i]] = np.zeros((num_chimps, num_chimps)) # current array for this date, reinitialized
            curr_date = 0 # the current date
            for _, row in content_of_interest.iterrows():
                # read date and check whether we need to clear history
                new_date = row['date']
                if curr_date != new_date: # need reinitialization of the history
                    curr_array_dict = {}
                    curr_array_consec_dict = {} # for consecutive occurrences
                    for i in range(len(edge_name_list)):
                        curr_array_dict[edge_name_list[i]] = np.zeros((num_chimps, num_chimps)) # current array for this date, reinitialized
                        curr_array_consec_dict[edge_name_list[i]] = np.zeros((num_chimps, num_chimps)) # current array for this date, reinitialized
                    curr_date = new_date
                
                # read data
                focal_list = [row['code']]
                code_ind = name_str.index(row['code'])
                try:
                    prox2_list = row['prox2'].split(',')
                except AttributeError:
                    prox2_list = []
                try:
                    prox5_list = row['prox5'].split(',')
                except AttributeError:
                    prox5_list = []
                try:
                    party_list = row['party'].split(',')
                except AttributeError:
                    party_list = []
                # for grooming
                try:
                    groomer_ind = name_str.index(row['groomer'])
                    groomee_ind = name_str.index(row['groomee'])
                    if groomer_ind != code_ind:
                        grooming = groomer_ind
                    else:
                        grooming = groomee_ind 
                except ValueError:
                    grooming = None
                
                # process array data
                # grooming with FM (code individual), in additional to prox2
                if grooming is not None:
                    name1_ind = code_ind
                    name2_ind = grooming
                    if curr_grooming_array[name1_ind, name2_ind] == 0:
                        curr_grooming_array[name1_ind, name2_ind] = grooming_weight
                        curr_grooming_array[name2_ind, name1_ind] = grooming_weight
                        full_grooming_array[name1_ind, name2_ind] += grooming_weight
                        full_grooming_array[name2_ind, name1_ind] += grooming_weight
                    else: # consecutive occurrences to be added
                        curr_grooming_consec_array[name1_ind, name2_ind] += grooming_weight
                        curr_grooming_consec_array[name2_ind, name1_ind] += grooming_weight
                        full_grooming_consec_array[name1_ind, name2_ind] += grooming_weight
                        full_grooming_consec_array[name2_ind, name1_ind] += grooming_weight
                
                # process array data
                # exclude prox2 and prox5 lists from party list using set operation
                party_list = list(set(party_list).difference(prox2_list).difference(prox5_list))

                # prepare dict to name lists
                name_list_dict = {
                    'prox2': prox2_list,
                    'prox5': prox5_list,
                    'party': party_list,
                    'focal': focal_list
                }
                
                # prepare name lists and add them based on a certain edge key
                for edge_key in edge_name_list:
                    if len(edge_key) == 11:
                        name_list1 = name_list_dict[edge_key[:5]]
                        name_list2 = name_list_dict[edge_key[-5:]]

                    # add edges
                    for name1 in name_list1:
                        for name2 in name_list2:
                            if name1 != "" and name2 != "" and name1 not in fake_names and name2 not in fake_names:
                                name1_ind = name_str.index(name1)
                                name2_ind = name_str.index(name2)
                                if not edge_key == 'prox2_focal' or name1_ind != grooming: # exclude the grooming partner from prox2_focal data
                                    if curr_array_dict[edge_key][name1_ind, name2_ind] == 0:
                                        curr_array_dict[edge_key][name1_ind, name2_ind] = edge_weights_dict[edge_key]
                                        curr_array_dict[edge_key][name2_ind, name1_ind] = edge_weights_dict[edge_key]
                                        full_array_dict[edge_key][name1_ind, name2_ind] += edge_weights_dict[edge_key]
                                        full_array_dict[edge_key][name2_ind, name1_ind] += edge_weights_dict[edge_key]
                                    else: # consecutive occurrences to be added
                                        curr_array_consec_dict[edge_key][name1_ind, name2_ind] += edge_weights_dict['consec_add']
                                        curr_array_consec_dict[edge_key][name2_ind, name1_ind] += edge_weights_dict['consec_add']
                                        full_array_consec_dict[edge_key][name1_ind, name2_ind] += edge_weights_dict['consec_add']
                                        full_array_consec_dict[edge_key][name2_ind, name1_ind] += edge_weights_dict['consec_add']

            for edge_key in edge_name_list:
                np.save('../data/graphs_without_hardcoded_parameters/'+edge_key+'_array_'+str(year)+'_'+str(month), full_array_dict[edge_key])
                np.save('../data/graphs_without_hardcoded_parameters/'+edge_key+'_consec_array_'+str(year)+'_'+str(month), full_array_consec_dict[edge_key])
            # grooming
            np.save('../data/graphs_without_hardcoded_parameters/grooming_array_'+str(year)+'_'+str(month), full_grooming_array)
            np.save('../data/graphs_without_hardcoded_parameters/grooming_consec_array_'+str(year)+'_'+str(month), full_grooming_consec_array)

# prepare a dictionary for the data
start_year = 1998
end_year = 2022
months_per_year = 12
T = end_year - start_year + 1
n = num_chimps
H = len(edge_name_list)

raw_networks = np.zeros((H, T, n, n))  # Raw adjacency matrices
add_networks = np.zeros((H, T, n, n))  # Increment adjacency matrices
focal_list = []                        # Focal sets at each time step
existing_node_list = []                # Existing nodes at each time step

for year in range(start_year, end_year+1):
    # Construct the adjacency matrices
    for h, edge_key in enumerate(edge_name_list):
        curr_adj = 0
        curr_consec_adj = 0
        for month in range(1, months_per_year+1):
            curr_adj += np.load('../data/graphs_without_hardcoded_parameters/'+edge_key+'_array_'+str(year)+'_'+str(month)+'.npy')
            # consec addition to be added
            curr_consec_adj += np.load('../data/graphs_without_hardcoded_parameters/'+edge_key+'_consec_array_'+str(year)+'_'+str(month)+'.npy')
        raw_networks[h, year-start_year] = curr_adj
        add_networks[h, year-start_year] = curr_consec_adj
        # Combine raw networks for existing node determination
        combined_adj = sum([raw_networks[h, year-start_year] for h in range(H)])
        # locate the existing nodes at time t in combined_adj
        node_degrees = np.sum(combined_adj, axis=1)
        existing_nodes = np.where(node_degrees > 0)[0]
        existing_node_list.append(existing_nodes)

chimps_data_dict = {
    'raw_networks': raw_networks,
    'add_networks': add_networks,
    'focal_list': focal_list,
    'name_str': name_str,
    'name_str_focal': name_str_focal,
    'name_str_non_focal': name_str_non_focal,
    'num_chimps': num_chimps,
    "existing_nodes": existing_node_list
}

np.save('../data/chimps_data_dict.npy', chimps_data_dict)