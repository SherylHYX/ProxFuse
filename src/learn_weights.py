import os
import argparse
import random

import numpy as np
import torch
from model import ProxFuse
from simple_synthetic_generation import generate_simple_synthetic_model

### Parse Arguments
parser = argparse.ArgumentParser(description='GNN Experiments')
parser.add_argument(
    '--max_epochs', type=int, default=5000, help='maximum number of epochs')
parser.add_argument(
    '--early_stop_epochs', type=int, default=3000, help='early stop epochs')
parser.add_argument(
    '--train_steps', type=int, default=8, help='number of training steps')
parser.add_argument(
    '--val_steps', type=int, default=3, help='number of validation steps')
parser.add_argument(
    '--test_steps', type=int, default=3, help='number of test steps')
parser.add_argument(
    '--n', type=int, default=100, help='number of individuals')
parser.add_argument(
    '--ph_list', nargs="+", type=float, default=[0.1, 0.1, 0.1, 0.1, 0.1], help='List of edge probabilities for each hierarchy')
parser.add_argument(
    '--padd', type=float, default=0.1, help='increment edge probabilities for each hierarchy')
parser.add_argument(                
    '--w_list', nargs="+", type=float, default=[1, 0, 1, 0, 1], help='List of weights for each hierarchy')
parser.add_argument(                
    '--w_add', type=float, default=0, help='weight for increment networks')
parser.add_argument(
    '--chimps', action='store_true', default=False, help='experiments on chimps instead of synthetic data')
parser.add_argument(
    '--initial_values', nargs="+", type=float, default=[0.1, 0.1, 0.1, 0.1], help='the set of initial weights')
parser.add_argument(
    '--initial_addition', type=float, default=0.1, help='the initial addition weight (learnable)')
parser.add_argument(
    '--sim_coeff', type=float, default=1.0, help='the coefficient for similarity loss')
parser.add_argument(
    '--deg_coeff', type=float, default=1.0, help='the coefficient for degree loss')
parser.add_argument(
    '--reg_coeff', type=float, default=0.001, help='the coefficient for regularization loss')
parser.add_argument(
    '--lr', type=float, default=0.1, help='learning rate')
parser.add_argument(
    '--uniform_initial_value', type=float, default=-1, help='initial value for all weights, -1 means not using this option')
parser.add_argument(
    '--initial_alternative_ind', type=int, default=-1, help='alternative initial value index, -1 means not using this option, can take up to H for initial_addition')
parser.add_argument(
    '--initial_alternative_value', type=float, default=1.0, help='alternative initial value, only used when initial_alternative_ind is not -1')

parser.add_argument('--seed', type=int, default=10, help='random seed')
args = parser.parse_args()
if args.chimps:
    args.H = 10
else:
    args.H = len(args.ph_list)
if args.uniform_initial_value != -1:
    initial_values_all = [args.uniform_initial_value] * args.H
    initial_values_all[args.initial_alternative_ind] = args.initial_alternative_value
    args.initial_values = initial_values_all[:-1]
    args.initial_addition = min(initial_values_all[-1], 1) # do not exceed one
args.T = args.train_steps + args.val_steps + args.test_steps + 1
initial_values_str = ''.join(str(e) for e in args.initial_values)
if not args.chimps:
    ph_list_str = ''.join(str(e) for e in args.ph_list)
    w_list_str = ''.join(str(e) for e in args.w_list)
    data_info = (f'simple_synthetic/n{args.n}_T{args.T}_H{args.H}_ph{ph_list_str}_padd{args.padd}_w{w_list_str}_wAdd{args.w_add}_seed{args.seed}')
else:
    data_info = 'chimps'
for val_ind in range(len(args.initial_values)):
    if args.initial_values[val_ind] == 0:
        args.initial_values[val_ind] = 0.0001 # do not have actual zero
if args.initial_addition == 0:
    args.initial_addition = 0.0001 # do not have actual zero
elif args.initial_addition == 1:
    args.initial_addition = 0.9999 # do not have actual one
assert args.initial_addition <= 1, "Initial addition weight should be upper-bounded by 1!"

if not args.chimps:
    assert args.padd <= 1, "Increment edge probabilities for each hierarchy should be upper-bounded by 1!"
    assert args.w_add <= 1, "Weight for increment networks should be upper-bounded by 1!"
    assert len(args.ph_list) == len(args.w_list), "The number of edge probabilities should be the same as the number of actual weights!"
    assert len(args.w_list) == len(args.initial_values) + 1, "The number of learnable weights should be the number of actual weights minus one (except the first one)!"

    try: # load data_dict from synthetic_data folder which is saved as dictionary
        data_dict = np.load(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), f'../data/{data_info}.npy'), allow_pickle=True).item()
        print('Synthetic data loaded!')
    except FileNotFoundError: # generate data_dict and save it
        data_dict = generate_simple_synthetic_model(args.n, args.T, args.ph_list, args.padd, args.w_list, args.w_add, args.seed) # generate data
        np.save(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), f'../data/{data_info}.npy'), data_dict)
        print('Synthetic data generated and saved!')
else: # chimps
    assert len(args.initial_values) == 9, "The number of initial values should be 9 for chimps data!"
    data_dict = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), f'../data/chimps_data_dict.npy'), allow_pickle=True).item()
    print('Chimps data loaded!')


# set numpy and torch seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# model args
model_args_info = (f'trainSteps{args.train_steps}_val{args.val_steps}_test{args.test_steps}'
                f'_maxEpochs{args.max_epochs}_early{args.early_stop_epochs}_lr{args.lr}_seed{args.seed}'
                    f'initVal{initial_values_str}_initAdd{args.initial_addition}'
                    f'_simCoeff{args.sim_coeff}_degCoeff{args.deg_coeff}_regCoeff{args.reg_coeff}')
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_models',
    data_info, model_args_info+'_model.pth')
parameter_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_parameters',
    data_info, model_args_info+'_parameters.npy')
epoch_cost_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_epoch_cost',
    data_info, model_args_info+'_epoch_cost.npy')
# create folders if needed
result_paths = []
result_paths.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_models', data_info))
result_paths.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_parameters', data_info))
result_paths.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_epoch_cost', data_info))
for path in result_paths:
    os.makedirs(path, exist_ok=True)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_networks = torch.FloatTensor(data_dict["raw_networks"]).to(args.device)
added_networks = torch.FloatTensor(data_dict["add_networks"]).to(args.device)
existing_nodes = data_dict["existing_nodes"]
coexisting_nodes = []
for t in range(args.T-1):
    coexisting = list(set(existing_nodes[t]).intersection(existing_nodes[t+1]))
    coexisting_nodes.append(coexisting)

model = ProxFuse(args.device, raw_networks, added_networks, coexisting_nodes, \
                    torch.tensor(args.initial_values).to(args.device), torch.tensor(args.initial_addition).to(args.device)).to(args.device)  

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

best_val_cost = 10000
non_improved_epochs = 0
best_epoch_num = 1


for epoch in range(args.max_epochs):
    # training
    model.train()
    sim_mse = 0
    degree_mse = 0
    for start_step in range(args.train_steps):
        sim_mat_list, normalized_weighted_degree_list = model(start_step)
        sim_mse += torch.mean((sim_mat_list[1] - sim_mat_list[0])**2)
        degree_mse += torch.mean((normalized_weighted_degree_list[1] - normalized_weighted_degree_list[0])**2)
    cost = args.sim_coeff * sim_mse + args.deg_coeff * degree_mse
    cost = cost/(args.train_steps) # normalization
    # Compute the L2 norm of each transformed weight tensor and count the number of parameters
    add_parameter = 1/(1+torch.exp(-model.trans_addition))
    # duplicate add_parameter to have the same size as the current_parameters
    # duplicated_add_parameter = add_parameter.repeat(len(args.initial_values))
    current_parameters = (torch.log(1+torch.exp(model.trans_graph_weight_increments)))
    transformed_weights = torch.cat((current_parameters, add_parameter.view(1)))
    regularization_term = torch.norm(transformed_weights, p=2) ** 2 # sum(torch.sum(torch.abs(w)) for w in transformed_weights)
    num_params = len(transformed_weights)
    regularization_loss = regularization_term / num_params
    train_loss = cost + args.reg_coeff * regularization_loss
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Training in epoch {}: Loss is {:.4f}, regularization loss is {:.4f}, Similarity MSE is {:.4f} and Degree MSE is {:.4f}".format(epoch+1, train_loss, regularization_loss, sim_mse, degree_mse))
    print("The current training set of increment parameters are: {}, and addition parameter: {:.3f}.".format(current_parameters.detach().cpu(), add_parameter.detach().cpu()))
    # validaiton
    model.eval()
    sim_mse = 0
    degree_mse = 0
    for start_step in range(args.train_steps, args.val_steps+args.train_steps):
        sim_mat_list, normalized_weighted_degree_list = model(start_step)
        sim_mse += torch.mean((sim_mat_list[1] - sim_mat_list[0])**2)
        degree_mse += torch.mean((normalized_weighted_degree_list[1] - normalized_weighted_degree_list[0])**2)
    cost = args.sim_coeff * sim_mse + args.deg_coeff * degree_mse
    cost = (cost/(args.val_steps)).detach().cpu()
    print("Validation in epoch {}: Loss is {:.4f}, Similarity MSE is {:.4f} and Degree MSE is {:.4f}".format(epoch+1, cost, sim_mse, degree_mse))

    if cost < best_val_cost:
        best_val_cost = cost
        non_improved_epochs = 0
        best_epoch_num = epoch+1
        add_parameter = (1/(1+torch.exp(-model.trans_addition))).detach().cpu()
        current_parameters = (torch.log(1+torch.exp(model.trans_graph_weight_increments))).detach().cpu()
        print('The current best set of increment parameters are: {}, and '.format(current_parameters), end='')
        print('addition parameter: {:.3f}.'.format(add_parameter))
        torch.save(model.state_dict(), model_path)
    else:
        non_improved_epochs += 1
        if non_improved_epochs >= args.early_stop_epochs:
            print("Early stopped!")
            break

model = ProxFuse(args.device, raw_networks, added_networks, coexisting_nodes, \
                    torch.tensor(args.initial_values).to(args.device), torch.tensor(args.initial_addition).to(args.device)).to(args.device)  
model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
model.eval()
sim_mse = 0
degree_mse = 0
for start_step in range(args.val_steps+args.train_steps, args.T-1):
    sim_mat_list, normalized_weighted_degree_list = model(start_step)
    sim_mse += torch.mean((sim_mat_list[1] - sim_mat_list[0])**2)
    degree_mse += torch.mean((normalized_weighted_degree_list[1] - normalized_weighted_degree_list[0])**2)
cost = args.sim_coeff * sim_mse + args.deg_coeff * degree_mse
cost = (cost/args.test_steps).detach().cpu()
print("Test for best epoch {}: Loss is {:.4f}, Similarity MSE is {:.4f} and Degree MSE is {:.4f}".format(best_epoch_num, cost, sim_mse, degree_mse))

add_parameter = (1/(1+torch.exp(-model.trans_addition))).detach().cpu()
increments = (torch.log(1+torch.exp(model.trans_graph_weight_increments))).detach().cpu()
print('After {} epochs, the best set of increment parameters are: {}, and '.format(epoch+1, increments), end='')
print('addition parameter: {:.3f}.'.format(add_parameter))

all_parameters = list(increments)
all_parameters.append(add_parameter)
np.save(parameter_path, all_parameters)
epoch_cost = [best_epoch_num, cost]
np.save(epoch_cost_path, epoch_cost)
    


