import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import spmm, get_laplacian, degree
import math
import numpy as np


def calculate_importance(P_train, P_val, S_train, S_val, Y_train, Y_val):
    """
    Calculate the importance of training samples in a machine learning model.

    Parameters:
    P_train (Tensor): Aggregated features for training data, represented as (\tilde{A}_{train}X).
                      Size is (n_train, d), where n_train is the number of training samples
                      and d is the feature dimension.
    P_val (Tensor): Aggregated features for validation data, represented as (\tilde{A}_{valid}X).
                    Size is (n_valid, d), where n_valid is the number of validation samples
                    and d is the feature dimension.
    S_train (Tensor): Softmax output from the model for training data.
                      Size is (n_train, c), where c is the number of classes.
    S_val (Tensor): Softmax output from the model for validation data.
                    Size is (n_valid, c), where c is the number of classes.
    Y_train (Tensor): Label matrix for training data.
                      Size is (n_train, c), where c is the number of classes.
    Y_val (Tensor): Label matrix for validation data.
                    Size is (n_valid, c), where c is the number of classes.

    Returns:
    Tensor: A vector of importance scores for each training sample.
    """
    # Compute the dot product of aggregated feature matrices for training and validation data.
    # This results in a matrix of size (n_train, n_valid) representing feature similarities.
    u = torch.matmul(P_train, P_val.T)

    # Compute the dot product of the discrepancy matrices between softmax outputs and actual labels
    # for training and validation data. This results in a matrix of size (n_train, n_valid)
    # representing prediction discrepancies.
    v = torch.matmul(S_train - Y_train, (S_val - Y_val).T)

    # Sum the element-wise product of the two matrices along the second dimension (n_valid).
    # This results in a vector of size (n_train) representing the importance score of each training sample.
    imp = torch.sum(u * v, dim=1)

    return imp


def normalize_adjacency(edge_index, num_nodes):
    edge_index, edge_weight = gcn_norm(edge_index, None, num_nodes, False,
                                       True, 'source_to_target')
    sp_adj = torch.sparse_coo_tensor(edge_index, edge_weight, 
                                     (num_nodes, num_nodes))
    return sp_adj.t()  # return sp_adj -> Old Version


def SSGC_Embedding(x, edge_index, k: int, alpha: float):
    sp_adj = normalize_adjacency(edge_index, x.size(0))
    ret_x = torch.zeros_like(x)
    tmp_x = x
    for _ in range(k):
        tmp_x = spmm(sp_adj, tmp_x)
        ret_x += (1-alpha) * tmp_x + alpha * x
    return ret_x / k


def APPNP_Embedding(x, edge_index, k: int, alpha: float):
    sp_adj = normalize_adjacency(edge_index, x.size(0))
    x_0 = x.clone()
    for _ in range(k):
        x = (1-alpha) * spmm(sp_adj, x) + alpha * x_0
    return x


def get_in_neighbor_dist(edge_index, num_nodes):
    src, dst = edge_index
    
    adj_list = []
    for i in range(num_nodes):
        adj_i = torch.zeros(num_nodes, dtype=torch.float32).to(edge_index.device)
        src_idx = src[dst==i]
        for j in src_idx:
            adj_i[j] = adj_i[j] + 1
        adj_list.append(adj_i)
        
    adj_list = torch.stack(adj_list, dim=0)
    adj_list = F.normalize(adj_list, p=1.0, dim=1)
    return adj_list


@torch.no_grad()
def sample_synthesis_pair(class_num_tensor, idx_info, aux_pool_idx, aux_pool_label):
    max_num = class_num_tensor.max().long().item()
    
    # Sampling center nodes
    center_idx = []
    for l, item in enumerate(idx_info):
        class_size = item.size(0)
        tmp_idx = torch.randint(class_size, (max_num-class_size,))
        center_idx.append(item[tmp_idx])
    center_idx = torch.cat(center_idx, dim=0)
    
    # Sampling auxiliary nodes
    prob = torch.log(class_num_tensor)/ class_num_tensor
    aux_pool_prob = prob[aux_pool_label]
    tmp_idx = torch.multinomial(aux_pool_prob, center_idx.size(0), replacement=True) 
    aux_idx = aux_pool_idx[tmp_idx]
    
    # Sort source nodes
    center_idx, sorted_idx = torch.sort(center_idx)
    aux_idx = aux_idx[sorted_idx]
    
    return center_idx, aux_idx


@torch.no_grad()
def sample_neighbors(edge_index, num_nodes, in_neighbor_dist, center_idx, aux_idx, lambdas, train_node_mask=None):
    src, dst = edge_index
    
    mixed_neig_prob = lambdas * in_neighbor_dist[center_idx] + (1 - lambdas) * in_neighbor_dist[aux_idx]
    
    # sample degree
    in_degree = degree(dst, num_nodes=num_nodes)
    max_degree = in_degree.max().int().item() + 1
    degree_dist = torch.zeros((max_degree,)).to(dst.device)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(in_degree).bool()
    degree_dist.scatter_add_(0, in_degree[train_node_mask].long(), 
                             torch.ones_like(in_degree[train_node_mask]))
    degree_dist = degree_dist.repeat((center_idx.size(0),1))
    sampled_degree = torch.multinomial(degree_dist, 1).squeeze(dim=1)
    sampled_degree = torch.min(sampled_degree, in_degree[center_idx]) 
    
    # Sample neighbor
    sampled_neighbors =  torch.multinomial(mixed_neig_prob + 1e-12, max_degree) 
    tmp_idx = torch.arange(max_degree).to(dst.device)
    new_neighbor = sampled_neighbors[(tmp_idx.unsqueeze(dim=0) - sampled_degree.unsqueeze(dim=1))<0]
    new_nodes = torch.arange(sampled_degree.size(0)).to(dst.device) + num_nodes
    new_nodes = torch.repeat_interleave(new_nodes, sampled_degree.long())
    
    new_src = torch.cat([src, new_neighbor], dim=0)
    new_dst = torch.cat([dst, new_nodes], dim=0)
    
    return torch.stack([new_src, new_dst], dim=0)


def mix_feature(old_x, center_idx, aux_idx, lambdas):
    new_x = old_x.clone()
    
    # feature mixup
    aug_x = lambdas * new_x[center_idx] + (1-lambdas) * new_x[aux_idx]
    
    return torch.cat([new_x, aug_x], dim=0)


def subsample_meta_mask(meta_idx_info, subsampled_size, num_nodes):
    ret_mask = torch.zeros(num_nodes).bool().to(meta_idx_info[0].device)
    subsampled_size = max(subsampled_size, 1)
    
    for item in meta_idx_info:
        shuffled_item = item[torch.randperm(item.size(0))]
        ret_mask[shuffled_item[:subsampled_size]] = True
    
    return ret_mask


def ul_scale_schedule(cur_epoch, maxi_epoch: int = 500):
    if cur_epoch > maxi_epoch:
        return 0.0
    else:
        return np.cos((cur_epoch / maxi_epoch) * np.pi * 0.5)
