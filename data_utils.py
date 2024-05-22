import torch
import numpy as np
from torch_scatter import scatter_add
from sklearn.model_selection import train_test_split


def get_dataset(name, path, split_type='full'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == 'photo' or name == 'computers':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(path, name, transform = T.NormalizeFeatures())
        data.x = F.batch_norm(data.x, torch.mean(data.x, dim=0), torch.var(data.x, dim=0))
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label)).to(label.device)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


## Construct LT ##
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1: # We does not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask)).to(label.device)
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.cpu().numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def split_for_Amazon_natural_LT(labels, num_class: int, ratio:float):
    training_ratio = 10. / 100.
    validation_ratio = 10. / (100. - 10.)
    num_nodes = labels.size(0)
    labels_cpu = labels.cpu()
    labels_np = labels.cpu().numpy()
    
    data_train_mask = torch.zeros(num_nodes).bool()
    valid_test_nids = []
    index = torch.arange(num_nodes)

    class_index_list = []
    class_num_list = []
    for i in range(num_class):
        class_index_list.append(index[labels_cpu==i])
        class_num_list.append((labels_cpu==i).sum().item())
        
    class_num_tensor = torch.tensor(class_num_list)
    sorted_n_data, sorted_indices = torch.sort(class_num_tensor, descending=True)
    
    mu = np.power(1./ratio, 1./(num_class - 1.))
    mu_list = []
    for i in range(num_class):
        mu_list.append(np.power(mu, i))
    mu_array = np.array(mu_list)
    mu_array = mu_array / np.sum(mu_array)
    
    torch.manual_seed(0)
    for i in range(num_class):        
        total_num_per_class = sorted_n_data[i].item()
        num_train_per_class = int(np.ceil(num_nodes * training_ratio * mu_array[i]))
        
        org_index = sorted_indices[i].item()
        index_perm = torch.randperm(total_num_per_class)
        tmp_train_index = class_index_list[org_index][index_perm][:num_train_per_class]
        tmp_val_test_index = class_index_list[org_index][index_perm][num_train_per_class:]
        
        data_train_mask[tmp_train_index] = True
        valid_test_nids.append(tmp_val_test_index)
    valid_test_nids = torch.cat(valid_test_nids, dim=0).numpy()
    
    valid_idx, test_idx = train_test_split(valid_test_nids, stratify=labels_np[valid_test_nids],
                                           train_size=validation_ratio, random_state=2, 
                                           shuffle=True)
    
    data_train_mask = data_train_mask.to(labels.device)
    data_valid_mask = torch.zeros(num_nodes, dtype=torch.bool).to(labels.device)
    data_test_mask  = torch.zeros(num_nodes, dtype=torch.bool).to(labels.device)

    data_valid_mask[torch.tensor(valid_idx).long()] = True
    data_test_mask[torch.tensor(test_idx).long()] = True
    
    return data_train_mask, data_valid_mask, data_test_mask
