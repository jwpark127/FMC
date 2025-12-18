import numpy as np
from collections import Counter
import torch
from itertools import permutations
from scipy.stats import multivariate_normal

def sorted_counter(label, max_):
    not_in_list = []
    label = to_numpy(label)
    for i in range(max_):
        if i not in label:
            not_in_list.append(i)
    sorted_dict = sorted(Counter(label).items(), key = lambda x: (x[0], x[1]))
    for excluded in not_in_list:
        sorted_dict.insert(excluded, [excluded, 0])
    return np.array(sorted_dict, dtype = np.int64)

def calc_distance(data, center, label, n_components):
    L1_distance= 0
    L2_distance= 0
    if type(label) != np.ndarray:
        label = to_numpy(label)
    if type(center) != np.ndarray:
        center = to_numpy(center)
    if type(data) != np.ndarray:
        data = to_numpy(data)
    if type(n_components) == torch.Tensor:
        n_components = n_components.item()
    for kth in range(n_components):
        L1_distance += np.sqrt(((data[label == kth] - center[kth])**2).sum(axis = 1)).sum()
        L2_distance += ((data[label == kth] - center[kth])**2).sum()
    return L1_distance, L2_distance

def re_calc_distance(data, label, n_components):
    L1_distance= 0
    L2_distance= 0
    if type(label) != np.ndarray:
        label = to_numpy(label)
    if type(data) != np.ndarray:
        data = to_numpy(data)
    if type(n_components) == torch.Tensor:
        n_components = n_components.item()

    new_centers = []
    for kth in range(n_components):
        new_centers.append(data[label == kth].mean(axis = 0))

    for kth in range(n_components):
        L1_distance += np.sqrt(((data[label == kth] - new_centers[kth])**2).sum(axis = 1)).sum()
        L2_distance += ((data[label == kth] - new_centers[kth])**2).sum()
    return L1_distance, L2_distance

def calculate_balance(model, S):
    S = S.to('cpu')
    unique_S = S.unique()
    assignments = [sorted_counter(model.predict()[S==s], model.K)[:,1] for s in unique_S]
    permutes = permutations(unique_S, 2)
    ratio_matrix = np.array([assignments[per[0]] / assignments[per[1]] for per in permutes])
    return ratio_matrix.min()

def calculate_Fscore(resp, S):
    unique_S = S.unique()
    resp_list = np.array([resp[S == s].mean(axis = 0) for s in unique_S])
    permutes = permutations(unique_S, 2)
    ratio_matrix = np.abs(np.array([resp_list[per[0]] - resp_list[per[1]] for per in permutes]))
    return ratio_matrix.max()

def get_max_index(tensor):
    if type(tensor) == np.ndarray:
        tensor = torch.tensor(tensor)
    max_index = torch.argmax(tensor)
    max_index_2d = np.unravel_index(max_index.item(), tensor.shape)
    return max_index_2d

def get_min_index(tensor):
    if type(tensor) == np.ndarray:
        tensor = torch.tensor(tensor)
    max_index = torch.argmin(tensor)
    max_index_2d = np.unravel_index(max_index.item(), tensor.shape)
    return max_index_2d

def to_numpy(tensor: torch.tensor) -> np.array:
    if type(tensor) != np.ndarray:
        return tensor.cpu().detach().numpy()
    else:
        return np.array(tensor)
    
def to_tensor(numpy_array: np.array) -> torch.tensor:
    return torch.tensor(numpy_array)

def resp_(test, k, mu, prec, weights):
    resp_ = np.zeros((test.shape[0], k))
    for kth in range(k):
        resp_[:,kth] = weights[kth] * multivariate_normal(mean = mu[kth], cov = 1/prec).pdf(test)
    return resp_