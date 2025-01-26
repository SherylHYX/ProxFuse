import random
import numpy as np
import networkx as nx

def generate_simple_synthetic_model(n, T, ph_list, padd, w_list, w_add, seed=None):
    """
    Generate synthetic multilayer network model with nested levels of proximities.

    Args:
        n (int): Total number of nodes.
        T (int): Number of time steps.
        ph_list (list): List of edge probabilities for each hierarchy. We also use them for edge generation, deletion, and change probabilities.
        padd (float): Float of increment edge probabilities for each hierarchy.
        w_list (list): List of weights for each hierarchy.
        w_add (float): Weight for increment networks.
        seed    (int): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary containing raw networks and increment networks.
    """
    H = len(ph_list)  # Number of hierarchies
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    raw_networks = np.zeros((H, T, n, n))  # Raw adjacency matrices
    add_networks = np.zeros((H, T, n, n))  # Increment adjacency matrices
    exixsting_node_list = []

    for t in range(T):
        if t == 0:
            # Generate networks for each hierarchy
            for h in range(H):
                # Generate raw network using G(n, p)
                G_raw = nx.erdos_renyi_graph(n, ph_list[h])
                # assign random edge weights to edges in an undirected fashion, taking integer values from 1 to H
                for (u, v) in G_raw.edges():
                    G_raw[u][v]['weight'] = random.randint(1, H)
                raw_adj = nx.adjacency_matrix(G_raw).toarray()
                raw_networks[h, t] = raw_adj

                # Generate increment network
                G_add = nx.erdos_renyi_graph(n, padd)
                # assign random edge weights to edges in an undirected fashion, taking integer values from 1 to H
                for (u, v) in G_add.edges():
                    G_add[u][v]['weight'] = random.randint(1, H)
                raw_adj_binary = np.where(raw_adj > 0, 1, 0)
                add_adj = np.multiply(raw_adj_binary, nx.adjacency_matrix(G_add).toarray()) # Element-wise multiplication
                add_networks[h, t] = add_adj
            combined_adj = sum([sum(w_list[:(h+1)]) * (raw_networks[h, t] + w_add * add_networks[h, t]) for h in range(H)])
            fixed_combined_adj = combined_adj.copy() # for later construction
            G_fixed = nx.from_numpy_array(fixed_combined_adj)
            total_edges = list(G_fixed.edges(data=True))
            # compute the number of edges in each hierarchy if they add up to the total number of edges, based on p_h_list
            num_edges = len(total_edges)
            ph_sum = sum(ph_list)
            normalized_ph_list = [ph / ph_sum for ph in ph_list]
        else: # t > 0
            # Shuffle edges and divide into groups
            random.shuffle(total_edges)
            num_edges_h = np.random.multinomial(num_edges, normalized_ph_list)
            edge_list_h = [total_edges[sum(num_edges_h[:i]):sum(num_edges_h[:i+1])] for i in range(H)]

            # Create subgraphs with the same node set as G_fixed
            subgraph_list = []
            subgraph_adj_list = []
            subgraph_adj_sum = 0
            for edges in edge_list_h:
                subgraph = nx.Graph()
                subgraph.add_nodes_from(G_fixed.nodes)  # Add all nodes from the original graph
                subgraph.add_edges_from([(u, v, d) for u, v, d in edges])  # Add edges with weights
                subgraph_list.append(subgraph)
                subgraph_adj = nx.adjacency_matrix(subgraph).toarray()
                subgraph_adj_list.append(subgraph_adj)
                subgraph_adj_sum += subgraph_adj.sum()

            # compute edge weights for each subgraph
            for h in range(H):
                sum_adj_h = subgraph_adj_list[h]/sum(w_list[:(h+1)]) # this is raw_networks[h, t] + w_add * add_networks[h, t]
                
                # now select where edges present in add_networks[h, t]
                G_add = nx.erdos_renyi_graph(n, padd)
                # if w_add = 0, no influence to combinedadj, so we only need to restrict that the edges present in raw_networks[h, t]  
                # assign random edge weights to edges in an undirected fashion, taking integer values from 1 to H
                for (u, v) in G_add.edges():
                    G_add[u][v]['weight'] = random.randint(1, H)
                sum_adj_binary = np.where(sum_adj_h > 0, 1, 0)
                add_adj = np.multiply(sum_adj_binary, nx.adjacency_matrix(G_add).toarray()) # Element-wise multiplication
                if w_add > 0: # need to have nonnegative raw_networks[h, t] and add_networks[h, t]
                    add_adj = np.maximum(0, np.minimum(add_adj, sum_adj_h/w_add-0.1)) # make sure the sum_adj_h/w_add is larger than add_adj and add_adj is nonnegative
                add_networks[h, t] = add_adj
                raw_networks[h, t] = sum_adj_h - w_add * add_adj
              
                
        # Combine raw and increment networks for existing node determination
        combined_adj = sum([sum(w_list[:(h+1)]) * (raw_networks[h, t] + w_add * add_networks[h, t]) for h in range(H)])
        # locate the existing nodes at time t in combined_adj
        node_degrees = np.sum(combined_adj, axis=1)
        existing_nodes = np.where(node_degrees > 0)[0]
        exixsting_node_list.append(existing_nodes)
                                        

    return {
        "raw_networks": raw_networks,
        "add_networks": add_networks,
        "w_list": w_list,
        "w_add": w_add,
        "n": n,
        "existing_nodes": exixsting_node_list
    }