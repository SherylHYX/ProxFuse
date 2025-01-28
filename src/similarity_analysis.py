import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

start_year = 1998
end_year = 2022
months_per_year = 12
year_of_split = 2018
array_save_name_list = ['full']
frequency_list = ['yearly']
graph_type_list = ['learned'] # ['learned', 'unlearned', 'binary']

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


########################################################################################

def compute_similarity_measures(community_identities_all):
    num_time_steps, num_nodes = community_identities_all.shape
    longest_shared_duration_matrix = np.zeros((num_nodes, num_nodes))
    shared_identity_count_matrix = np.zeros((num_nodes, num_nodes))
    longest_shared_duration_matrix[:] = np.nan # so that diagonals will be blank
    shared_identity_count_matrix[:] = np.nan

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid duplicate computations; matrices are symmetric
            longest_duration = 0
            current_duration = 0
            shared_count = 0

            for time_step in range(num_time_steps):
                community_i = community_identities_all[time_step, i]
                community_j = community_identities_all[time_step, j]

                # Count shared identities, ignoring -1
                if community_i == community_j and community_i != -1:
                    shared_count += 1
                    current_duration += 1
                else:
                    if current_duration > 0:
                        longest_duration = max(longest_duration, current_duration)
                        current_duration = 0  # Reset for next continuous segment

            # Check for the last sequence before exiting loop
            longest_duration = max(longest_duration, current_duration)

            # Update matrices
            longest_shared_duration_matrix[i, j] = longest_duration
            longest_shared_duration_matrix[j, i] = longest_duration
            shared_identity_count_matrix[i, j] = shared_count
            shared_identity_count_matrix[j, i] = shared_count


    return longest_shared_duration_matrix, shared_identity_count_matrix

def common_community_similarity_analysis(array_save_name, frequency, graph_type='learned'):
    analysis_name = frequency + '_' + array_save_name + '_' + graph_type
    if graph_type == 'unlearned':
        file_path = '../unanalysis_raw_data/' + analysis_name + '/community_identities.npy'
    else:
        file_path = '../analysis_raw_data/' + analysis_name + '/community_identities.npy'
    if graph_type == 'binary':
        file_path = file_path.replace('community_identities', 'community_identities_binary')
    community_identities_all = np.load(file_path)
    valid_mask = ~np.any(np.isnan(community_identities_all), axis=1)
    # then extract those communities
    valid_community_identities_all = community_identities_all[valid_mask]

    longest_shared_duration_matrix, shared_identity_count_matrix = compute_similarity_measures(valid_community_identities_all)
    
    # Save the similarity matrices
    np.save('../analysis_raw_data/'+analysis_name+'/duration_similarity', longest_shared_duration_matrix)
    np.save('../analysis_raw_data/'+analysis_name+'/count_similarity', shared_identity_count_matrix)

    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), constrained_layout=True)

    # Longest Shared Duration Heatmap
    sns.heatmap(longest_shared_duration_matrix, ax=axs[0], cmap='viridis', cbar_kws={'label': ''})
    axs[0].set_xticks([])  # Remove xticks
    axs[0].set_yticks([])  # Remove yticks

    # Shared Identity Count Heatmap
    sns.heatmap(shared_identity_count_matrix, ax=axs[1], cmap='viridis', cbar_kws={'label': ''})
    axs[1].set_xticks([])  # Remove xticks
    axs[1].set_yticks([])  # Remove yticks

    # Save the combined plot
    plt.savefig(f"../analysis_communities/similarity/combined_similarity_{analysis_name}.pdf", format='pdf')
    plt.show()

###########################################################################################

def compute_quantile(values, probabilities, q=0.95):
    """
    Compute the q-quantile for discrete integers with associated probabilities.
    
    Parameters:
    - values: An array of discrete integers.
    - probabilities: An array of probabilities associated with each integer in `values`.
    - q: The quantile to compute, e.g., 0.95 for the 95th percentile.
    
    Returns:
    - The q-quantile value.
    """
    # Ensure values and probabilities are numpy arrays for efficient computation
    values = np.array(values)
    probabilities = np.array(probabilities)
    
    # Sort values and probabilities by the values
    idx = np.argsort(values)
    sorted_values = values[idx]
    sorted_probabilities = probabilities[idx]
    
    # Compute the cumulative sum of probabilities
    cumulative_probabilities = np.cumsum(sorted_probabilities)
    
    # Find the first value where the cumulative probability is greater than or equal to q
    quantile_index = np.where(cumulative_probabilities >= q)[0][0]
    quantile_value = sorted_values[quantile_index]
    
    return quantile_value

def compute_p_value(probabilities, observed_value):
    """
    Computes the p-value for an observed value given a Numpy array of probabilities.
    
    :param probabilities: Numpy array containing the probabilities for each discrete value from 0 to T.
    :param observed_value: The observed value for which to compute the p-value.
    :return: The p-value for the observed value.
    """
    # Ensure probabilities sum to 1
    if not np.isclose(np.sum(probabilities), 1):
        raise ValueError("The probabilities must sum to 1 instead of {}.".format(np.sum(probabilities)))
    
    # Ensure the observed value is within the valid range
    if observed_value < 0 or observed_value >= len(probabilities):
        raise ValueError("The observed value is out of the valid range.")
    
    # Compute the p-value
    p_value = np.sum(probabilities[int(observed_value):])
    
    return p_value

def compute_similariity_distribution_mean_std(valid_community_identities_ij_all):
    num_time_steps = valid_community_identities_ij_all.shape[0] # the valid time step number
    p_array = np.zeros(num_time_steps) # p_array[t] means the probability at time t for i, j to share community identity
    for t in range(num_time_steps):
        communities_t = valid_community_identities_ij_all[t]
        num_communities = int(communities_t.max() + 1)
        num_nodes = np.sum(communities_t != -1)
        for k in range(num_communities):
            num_nodes_community_k = np.sum(communities_t == k)
            p_array[t] += num_nodes_community_k * (num_nodes_community_k) # numerator only
        p_array[t] /= num_nodes * (num_nodes - 1) # divided by denominator


    # initilization
    M = np.zeros((num_time_steps, num_time_steps + 1))
    C = np.zeros((num_time_steps, num_time_steps + 1))
    # M[t, L] measures the probability of a node pair to have at most L shared consecutive paths, 
    # C[t, L] measures the probability of a node pair to have at most L shared paths (no need to be consecutive, total counts), 
    # where L is at least 0 and at most num_time_steps

    # first, boundary conditions
    M[0, 0] = 1 - p_array[0]
    M[0, 1] = p_array[0]
    C[0, 0] = 1 - p_array[0]
    C[0, 1] = p_array[0]
    for t in range(1, num_time_steps):
        M[t, 0] = np.prod(1 - p_array[:(t+1)])
        M[t, t+1] = np.prod(p_array[:(t+1)])
        C[t, 0] = np.prod(1 - p_array[:(t+1)])
        C[t, t+1] = np.prod(p_array[:(t+1)])
        M[t, t] += (1 - p_array[0]) * np.prod(p_array[1:(t+1)]) + (1 - p_array[t]) * np.prod(p_array[:t])


    # then apply recurrence
    for t in range(2, num_time_steps):
        for L in range(1, t):
            M[t, L] = M[t-L-1, :(L+1)].sum() * (1 - p_array[t-L]) * np.prod(p_array[(t-L+1):(t+1)])
            M[t, L] += (1 - p_array[t]) * M[t-1, L]
            if L >= 2:
                for s in range(t-L+1, t):
                    M[t, L] += (1 - p_array[s]) * np.prod(p_array[(s+1):(t+1)]) * M[s-1, L]
    for t in range(1, num_time_steps):
        for L in range(1, t+1):
            C[t, L] = p_array[t] * C[t-1, L-1] + (1-p_array[t]) * C[t-1, L]
    

    # now compute the mean and std values
    longest_shared_path_distribution = M[num_time_steps-1]
    num_shared_counts_distribution = C[num_time_steps-1]
    return longest_shared_path_distribution, num_shared_counts_distribution

def similarity_p_value_analysis(array_save_name, frequency, graph_type='learned'):
    analysis_name = frequency + '_' + array_save_name + '_' + graph_type
    if graph_type == 'unlearned':
        file_path = '../unanalysis_raw_data/' + analysis_name + '/community_identities.npy'
    else:
        file_path = '../analysis_raw_data/' + analysis_name + '/community_identities.npy'
    if graph_type == 'binary':
        file_path = file_path.replace('community_identities', 'community_identities_binary')
    community_identities_all = np.load(file_path)
    num_nodes = community_identities_all.shape[1]
    longest_p_value_matrix = np.ones((num_nodes, num_nodes))
    shared_p_value_matrix = np.ones((num_nodes, num_nodes))

    # load the original similarity matrices
    longest_shared_duration_matrix = np.load('../analysis_raw_data/'+analysis_name+'/duration_similarity.npy')
    shared_identity_count_matrix = np.load('../analysis_raw_data/'+analysis_name+'/count_similarity.npy')

    # normalization based on the random community structure results
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # first locate time steps when i and j co-exist (not community -1, and not NaN)
            valid_mask_ij = (community_identities_all[:, i] != -1) * (community_identities_all[:, j] != -1) * ~np.any(np.isnan(community_identities_all), axis=1)
            # then extract those communities
            valid_community_identities_ij_all = community_identities_all[valid_mask_ij]
            if np.sum(valid_mask_ij) > 0: # they do co-exist at least for one time step
                longest_shared_path_distribution, num_shared_counts_distribution = compute_similariity_distribution_mean_std(valid_community_identities_ij_all)
                longest_p_value_matrix[i, j] = compute_p_value(longest_shared_path_distribution, longest_shared_duration_matrix[i, j])
                shared_p_value_matrix[i, j] = compute_p_value(num_shared_counts_distribution, shared_identity_count_matrix[i, j])
            else:
                longest_p_value_matrix[i, j] = np.nan
                shared_p_value_matrix[i, j] = np.nan
            # symmetrization
            longest_p_value_matrix[j, i] = longest_p_value_matrix[i, j]
            shared_p_value_matrix[j, i] = shared_p_value_matrix[i, j]


    # Save the p-value matrices
    np.save('../analysis_raw_data/'+analysis_name+'/p_value_duration_similarity', longest_p_value_matrix)
    np.save('../analysis_raw_data/'+analysis_name+'/p_value_count_similarity', shared_p_value_matrix)
    
    # Plotting

    # normalized similarity plots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), constrained_layout=True)

    # Longest Shared Duration Heatmap
    sns.heatmap(longest_shared_duration_matrix, ax=axs[0], cmap='viridis', cbar_kws={'label': ''})
    axs[0].set_xticks([])  # Remove xticks
    axs[0].set_yticks([])  # Remove yticks

    # Shared Identity Count Heatmap
    sns.heatmap(shared_identity_count_matrix, ax=axs[1], cmap='viridis', cbar_kws={'label': ''})
    axs[1].set_xticks([])  # Remove xticks
    axs[1].set_yticks([])  # Remove yticks

    # Save the combined plot
    plt.savefig(f"../analysis_communities/similarity/combined_normalized_similarity_{analysis_name}.pdf", format='pdf')
    plt.show()

    # p-value plots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), constrained_layout=True)

    # Longest Shared Duration Heatmap
    sns.heatmap(longest_p_value_matrix, ax=axs[0], cmap='viridis', cbar_kws={'label': ''})
    axs[0].set_xticks([])  # Remove xticks
    axs[0].set_yticks([])  # Remove yticks

    # Shared Identity Count Heatmap
    sns.heatmap(shared_p_value_matrix, ax=axs[1], cmap='viridis', cbar_kws={'label': ''})
    axs[1].set_xticks([])  # Remove xticks
    axs[1].set_yticks([])  # Remove yticks

    
    # Save the combined plot
    plt.savefig(f"../analysis_communities/similarity/combined_p_value_similarity_{analysis_name}.pdf", format='pdf')
    plt.show()

###########################################################################################

def filter_subsets(cliques):
    """
    Filters out cliques that are proper subsets of larger cliques.

    Args:
    cliques (list of lists): The cliques to filter.

    Returns:
    list of lists: The filtered cliques.
    """
    # Sort cliques by length (descending) to ensure larger cliques are considered first
    sorted_cliques = sorted(cliques, key=len, reverse=True)
    filtered_cliques = []
    for i in range(len(sorted_cliques)):
        for j in range(i + 1, len(sorted_cliques)):
            if set(sorted_cliques[i]).issuperset(sorted_cliques[j]):
                break
        else:
            # If not broken, it means this clique is not a superset of any other, so we keep it
            filtered_cliques.append(sorted_cliques[i])
    return filtered_cliques

def common_community_similarity_outstanding_analysis(array_save_name, frequency, graph_type='learned', significance_level=0.05):
    # for individual pairs with normalized value more than 2, for a hypothesis testing
    analysis_name = frequency + '_' + array_save_name + '_' + graph_type

    # load p values
    longest_p_value_matrix = np.load('../analysis_raw_data/'+analysis_name+'/p_value_duration_similarity.npy')
    shared_p_value_matrix = np.load('../analysis_raw_data/'+analysis_name+'/p_value_count_similarity.npy')

    filename = f"../analysis_communities/similarity/threshold_similarity_graph_cliques_{analysis_name}.txt"
    print(f"{analysis_name} thresholded similarity analysis")
    with open(filename, 'w') as f:
        f.write(f"{analysis_name} thresholded similarity analysis \n\n")
    
    # Find symmetric pairs with values greater than 2
    for p_value_matrix, similarity_name in zip([longest_p_value_matrix, shared_p_value_matrix], ['longest', 'count']):
        # Extract the upper triangular part of the matrix, including the diagonal
        upper_tri = np.triu(p_value_matrix)

        # Count non-NaN values in the upper triangular matrix
        non_nan_count = np.sum(~np.isnan(upper_tri))

        # corrected significance value based on Bonferroni correction
        corrected_significance_level = significance_level/non_nan_count

        rows, cols = np.where(p_value_matrix < corrected_significance_level)
        edges = {(min(i,j), max(i,j)) for i, j in zip(rows, cols) if i != j}

        # Construct a graph
        G = nx.Graph()
        G.add_edges_from(edges)

        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_size=250)
        nx.draw_networkx_edges(G, pos, width=1)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

        plt.axis("off")
        plt.savefig(f"../analysis_communities/similarity/thresholded_graph_{similarity_name}_similarity_{analysis_name}.pdf", format='pdf')
        plt.show()

        # Finding cliques
        cliques = [clique for clique in nx.find_cliques(G)]

        # Filtering out cliques that are subsets of larger cliques
        filtered_cliques = filter_subsets(cliques)
        print(f"For {similarity_name} similarity:")
        with open(filename, 'a') as f:
            f.write(f"For {similarity_name} similarity:\n")
        
        print("Number of unique cliques after removing subsets:", len(filtered_cliques))
        print("\nSome of the unique cliques (not subsets of larger cliques):")
        with open(filename, 'a') as f:
            f.write("Number of unique cliques after removing subsets:"+str(len(filtered_cliques)))
            f.write("\nSome of the unique cliques (not subsets of larger cliques):\n")
        for clique in filtered_cliques:
            print(clique)
            with open(filename, 'a') as f:
                f.write(str(clique))
                f.write('\n')

    print('*'*30)
    with open(filename, 'a') as f:
        f.write('*'*30+'\n')

###################################################################################

if __name__ == "__main__":
    for frequency in frequency_list:
        for array_save_name in array_save_name_list:
            for graph_type in graph_type_list:
                common_community_similarity_analysis(array_save_name, frequency, graph_type)

    for frequency in frequency_list:
        for array_save_name in array_save_name_list:
            for graph_type in graph_type_list:
                similarity_p_value_analysis(array_save_name, frequency, graph_type)
    
    for frequency in frequency_list:
        for array_save_name in array_save_name_list:
            for graph_type in graph_type_list:
                common_community_similarity_outstanding_analysis(array_save_name, frequency, graph_type)