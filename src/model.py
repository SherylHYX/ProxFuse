import torch

class ProxFuse(torch.nn.Module):
    def __init__(self, device, raw_networks, add_networks, coexisting_nodes, \
                 initial_values=torch.ones(5) * 0.1, initial_addition=torch.ones(1) * 0.1):
        super(ProxFuse, self).__init__()
        self.device = device
        self.raw_networks = raw_networks
        self.add_networks = add_networks
        self.coexisting_nodes = coexisting_nodes
        self.H = len(initial_values)
        initial_trans_increments = torch.log(torch.exp(initial_values)-1)
        self.trans_graph_weight_increments = torch.nn.Parameter(initial_trans_increments)
        # Learnable weights for each type of the graphs, restricting the first graph to have weight 1
        self.trans_addition = torch.nn.Parameter(torch.log(initial_addition/(1-initial_addition))) # logit transformation

    def forward(self, start_t):
        sim_mat_list = []
        normalized_weighed_degree_list = []
        # find co-existing nodes in the two time points
        coexisting_nodes = self.coexisting_nodes[start_t]
        for t in range(start_t, start_t+2):
            adj = 0
            addition_parameter = 1/(1+torch.exp(-self.trans_addition))
            for h in range(self.H+1): # considering also the start
                if h == 0: # restrict the graph weight to be 1 (normalization)
                    graph_weight = 1
                else:
                    graph_weight = 1 + (torch.log(1+torch.exp(self.trans_graph_weight_increments[:h]))).sum()
                curr_adj = self.raw_networks[h, t]
                curr_add_adj = self.add_networks[h, t]
                adj += graph_weight * (curr_adj + addition_parameter * curr_add_adj)
            adj = adj[coexisting_nodes][:, coexisting_nodes] # only consider the co-existing nodes
            # construct a similarity matrix based on dot product of the rows of adj
            # Compute row-wise norms
            row_norms = torch.norm(adj, p=2, dim=1, keepdim=True)  # L2 norm of each row

            # Normalize each row of A to have unit 2-norm
            normalized_adj = adj / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero
            sim_mat = torch.mm(normalized_adj, normalized_adj.t()) # cosine similarity
            sim_mat_list.append(sim_mat)
            # sim_mat_list.append(normalized_adj)

            weighted_degree = torch.sum(adj, dim=1, keepdim=True)
            normalized_weighed_degree = weighted_degree / (torch.sum(weighted_degree) + 1e-8)
            normalized_weighed_degree_list.append(normalized_weighed_degree)
        return sim_mat_list, normalized_weighed_degree_list