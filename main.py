import os
import random
import torch
import torch.nn.functional as F
from data_utils import *
from nets import *
from imp_calc import *
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import statistics
import numpy as np
import warnings
import argparse
import pandas as pd
from sklearn_extra.cluster import KMedoids
import time

start_time = time.time()

warnings.filterwarnings("ignore")

## Arg Parser ##
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Dataset Name')
parser.add_argument('--imb-ratio', type=int, default=50,
                    help='Imbalance Ratio')
parser.add_argument('--running-time', type=int, default=3,
                    help='Running times')

# Architecture
parser.add_argument('--net', type=str, default='GCN',
                    help='Architecture name')
parser.add_argument('--n-layer', type=int, default=2,
                    help='the number of layers')
parser.add_argument('--feat-dim', type=int, default=256,
                    help='Feature dimension')
# Method Specific
parser.add_argument('--meta-size', type=int, default=5,
                    help='Number of samples per class in the meta set')

parser.add_argument('--context-emb', type=str, choices=['SSGC', 'APPNP'], 
                    help='Context embedding type')
parser.add_argument('--depth', type=int, 
                    help='Aggregation depth')
parser.add_argument('--alpha', type=float, 
                    help='Aggregation alpha')
parser.add_argument('--meta-drop', type=float, default=0.0,
                    help='downsampling rate for meta set')

parser.add_argument('--syn-scale', type=float, default=1.0, 
                    help='The scaling ratio for synthetic loss')
parser.add_argument('--ul-warmup', type=int, default=0, 
                    help='Warmup period for unlabeled set')
parser.add_argument('--ul-decay-epoch', type=int, default=500, 
                    help='The decay epoch for unlabeled loss')
parser.add_argument('--ul-scale', type=float, default=1.0, 
                    help='The scaling ratio for unlabeled loss')

parser.add_argument('--km-depth', type=int, 
                    help='Aggregation depth for computing KMedoids')
parser.add_argument('--km-alpha', type=float, 
                    help='Aggregation alpha for computing KMedoids')
parser.add_argument('--km-method', type=str, 
                    help='Implementation of the KMedoids algorithm')
parser.add_argument('--km-metric', type=str, choices=['euclidean', 'cosine', 'manhattan'], 
                    help='distance metrics for computing KMedoids')
parser.add_argument('--km-init', type=str, default='heuristic',
                    choices=['heuristic', 'k-medoids++', 'build'], 
                    help='initialization methods for computing KMedoids')

parser.add_argument('--device', type=int, default=7)
args = parser.parse_args()
print(args)

device = torch.device('cuda:%d'%(args.device) if torch.cuda.is_available() else 'cpu')

## Load dataset
path = os.path.join('data', args.dataset)
dataset = get_dataset(args.dataset, path, split_type='geom-gcn')
data = dataset[0]
n_cls = data.y.max().item() + 1
data = data.to(device)

@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x, data.edge_index[:, train_edge_mask], None)
    probs = F.softmax(logits, dim=1)
    accs, baccs, f1s, aurocs, pc_f1s = [], [], [], [], []

    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(y_true, probs[mask].cpu().numpy(), 
                            average='macro', multi_class='ovr')
        pc_f1 = f1_score(y_true, y_pred, average=None)

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
        aurocs.append(auc)
        pc_f1s.append(pc_f1)
        
    return accs, baccs, f1s, aurocs, pc_f1s


repeatition = args.running_time
seed = 100
avg_val_acc_f1, avg_test_acc, avg_test_bacc, avg_test_f1, avg_test_auc, avg_test_per_class_f1s = [], [], [], [], [], []
for r in range(repeatition):
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        data_train_mask, data_val_mask, data_test_mask = data.train_mask[:,r%10].clone().bool(), data.val_mask[:,r%10].clone().bool(), data.test_mask[:,r%10].clone().bool()
    else:
        data_train_mask, data_val_mask, data_test_mask = split_for_Amazon_natural_LT(data.y.clone(), n_cls, args.imb_ratio)
    
    ## Fix seed ##
    torch.cuda.empty_cache()
    seed += 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data

    ## Construct a long-tailed graph ##
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = make_longtailed_data_remove(data.edge_index, data.y, n_data, 
                                                                                                                  n_cls, args.imb_ratio, 
                                                                                                                  data_train_mask.clone())
    else:
        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = make_longtailed_data_remove(data.edge_index, data.y, n_data, 
                                                                                                                  n_cls, 1., 
                                                                                                                  data_train_mask.clone())
    class_num_tensor = torch.Tensor(class_num_list).to(device).float()
    
    ## Sample a meta set for our method using KMedoid
    data_meta_mask = torch.zeros_like(data.y).bool()

    if args.context_emb == 'SSGC':
        P_KM = SSGC_Embedding(data.x, data.edge_index[:, train_edge_mask], 
                              k=args.km_depth, alpha=args.km_alpha)
    elif args.context_emb == 'APPNP':
        P_KM = APPNP_Embedding(data.x, data.edge_index[:, train_edge_mask], 
                              k=args.km_depth, alpha=args.km_alpha)
    else:
        raise NotImplementedError("Undefined Embedding!")
    P_KM = P_KM.cpu().numpy()
    
    new_idx_info = []
    meta_idx_info = []
    for i in range(n_cls):
        class_idx_tensor = idx_info[i]
        tmp_num_nodes = class_idx_tensor.size(0)
        class_idx_numpy = class_idx_tensor.cpu().numpy()

        train_emb = P_KM[class_idx_numpy]
        tmp_kmedoids = KMedoids(n_clusters=args.meta_size, method=args.km_method, 
                                metric=args.km_metric, init=args.km_init).fit(train_emb)
        selected_local_idx = torch.tensor(tmp_kmedoids.medoid_indices_).to(device)

        selected_mask = torch.zeros(tmp_num_nodes).bool().to(device)
        selected_mask[selected_local_idx] = True
        tmp_meta = class_idx_tensor[selected_mask]
        non_selected_mask = torch.logical_not(selected_mask)

        new_idx_info.append(class_idx_tensor[non_selected_mask])
        meta_idx_info.append(tmp_meta)

        # recompute the mask
        data_meta_mask[tmp_meta] = True
        data_train_mask[tmp_meta] = False

    new_class_num_list = []
    for i in range(n_cls):
        new_class_num_list.append(new_idx_info[i].size(0))
    new_class_num_tensor = torch.tensor(new_class_num_list).to(device)
    
    # Select a GNN model
    if args.net == 'GCN':
        model = create_gcn(nfeat=dataset.num_features, nhid=args.feat_dim,
                           nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    elif args.net == 'GAT':
        model = create_gat(nfeat=dataset.num_features, nhid=args.feat_dim,
                           nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    elif args.net == "SAGE":
        model = create_sage(nfeat=dataset.num_features, nhid=args.feat_dim,
                            nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    else:
        raise NotImplementedError("Not Implemented Architecture!")
    model = model.to(device)
    
    # Build the optimizer and the scheduler
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0),], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor = 0.5,
                                                           patience = 100,
                                                           verbose=False)
    
    #Prepare for Importance Calculation #
    if args.context_emb == 'SSGC':
        context_embedding = SSGC_Embedding
    elif args.context_emb == 'APPNP':
        context_embedding = APPNP_Embedding
    else:
        raise NotImplementedError("Undefined Architecture!")
    
    data_unlabeled_mask = torch.logical_or(data_val_mask, data_test_mask)
    num_of_unlabeled = torch.sum(data_unlabeled_mask).item()
    num_of_train = torch.sum(data_train_mask).item()
    filtering_mask_train = torch.ones(num_of_train).bool().to(device)
    filtering_mask_syn = None
    filtering_mask_unlabeled = None
    pesudo_y = None
    
    in_neighbor_dist = get_in_neighbor_dist(data.edge_index[:, train_edge_mask], data.x.size(0))
    aux_pool_idx = torch.cat(idx_info)
    aux_pool_label = data.y[aux_pool_idx]
    beta = torch.distributions.beta.Beta(2, 2)
    
    Y = F.one_hot(data.y, num_classes=n_cls)
    
    # Training part
    best_val_acc_f1 = 0
    for epoch in range(1, 2001):
        if epoch > 1:
            with torch.no_grad():
                model.train()
                
                # Synthesize nodes
                center_idx, aux_idx = sample_synthesis_pair(class_num_tensor, idx_info, aux_pool_idx, aux_pool_label)
                lambdas = beta.sample((center_idx.size(0),1)).to(device)
                new_edge_index= sample_neighbors(data.edge_index[:, train_edge_mask], data.x.size(0), 
                                                 in_neighbor_dist, center_idx, aux_idx, 
                                                 lambdas, train_node_mask)
                new_x = mix_feature(data.x, center_idx, aux_idx, lambdas)
                new_y = torch.cat([data.y, data.y[center_idx]])
                Y_mixed = lambdas * F.one_hot(data.y[center_idx], num_classes=n_cls) + (1-lambdas) * F.one_hot(data.y[aux_idx], num_classes=n_cls)
                new_Y = torch.cat([Y, Y_mixed], dim=0)
                
                # Recalculate the mask
                not_syn_mask = torch.zeros_like(center_idx).bool()
                tmp_data_train_mask = torch.cat([data_train_mask, not_syn_mask])
                tmp_data_unlabeled_mask = torch.cat([data_unlabeled_mask, not_syn_mask])
                
                tmp_data_meta_mask = subsample_meta_mask(meta_idx_info, int(args.meta_size * (1-args.meta_drop)), data.x.size(0))
                tmp_data_meta_mask = tmp_data_meta_mask.to(device)
                tmp_data_meta_mask = torch.cat([tmp_data_meta_mask, not_syn_mask])
                
                tmp_data_syn_mask = torch.cat([torch.zeros_like(data_train_mask).bool(),
                                               torch.ones_like(center_idx).bool()])
                
                # Calculate node importance
                P = context_embedding(new_x, new_edge_index, k=args.depth, alpha=args.alpha)
                output = model(new_x, new_edge_index, None)
                S = F.softmax(output, dim=1)
                
                # for training nodes 
                individual_importance = calculate_importance(P[tmp_data_train_mask], P[tmp_data_meta_mask], 
                                                             S[tmp_data_train_mask], S[tmp_data_meta_mask], 
                                                             new_Y[tmp_data_train_mask], new_Y[tmp_data_meta_mask])
                filtering_mask_train = individual_importance > 0.
                
                # for unlabeled nodes
                pesudo_y = torch.max(S, dim=1)[1]
                pesudo_Y = F.one_hot(pesudo_y, num_classes=n_cls)
                individual_importance_unlabeled = calculate_importance(P[tmp_data_unlabeled_mask], P[tmp_data_meta_mask], 
                                                                       S[tmp_data_unlabeled_mask], S[tmp_data_meta_mask],
                                                                       pesudo_Y[tmp_data_unlabeled_mask], new_Y[tmp_data_meta_mask])
                filtering_mask_unlabeled = individual_importance_unlabeled > 0.
                
                # for synthetic nodes
                individual_importance_syn = calculate_importance(P[tmp_data_syn_mask], P[tmp_data_meta_mask], 
                                                                 S[tmp_data_syn_mask], S[tmp_data_meta_mask],
                                                                 new_Y[tmp_data_syn_mask], new_Y[tmp_data_meta_mask])
                filtering_mask_syn = individual_importance_syn > 0.
                
        else:
            tmp_data_train_mask = data_train_mask.clone()
            tmp_data_unlabeled_mask = data_unlabeled_mask.clone()
            new_x = data.x.clone()
            new_edge_index = data.edge_index[:, train_edge_mask].clone()
            new_y = data.y.clone()
            new_Y = Y.clone()
        
        # Train the model
        model.train()
        optimizer.zero_grad()

        output = model(new_x, new_edge_index, None)
        
        # loss for labeled nodes
        loss = F.cross_entropy(output[tmp_data_train_mask][filtering_mask_train],
                               new_y[tmp_data_train_mask][filtering_mask_train])
        
        # loss for synthetic nodes
        if filtering_mask_syn is not None:
            syn_loss_1 = lambdas[filtering_mask_syn, 0] * F.cross_entropy(output[tmp_data_syn_mask][filtering_mask_syn],
                                                                          data.y[center_idx][filtering_mask_syn], reduction='none')
            syn_loss_2 = (1 - lambdas[filtering_mask_syn, 0]) * F.cross_entropy(output[tmp_data_syn_mask][filtering_mask_syn],
                                                                                data.y[aux_idx][filtering_mask_syn], reduction='none')
            syn_loss = torch.mean(syn_loss_1 + syn_loss_2)
            loss += args.syn_scale * syn_loss
        
        # loss for unlabeled nodes
        if pesudo_y is not None and epoch > args.ul_warmup:
            tmp_scale = ul_scale_schedule(epoch, args.ul_decay_epoch) * args.ul_scale
            ul_loss = F.cross_entropy(output[tmp_data_unlabeled_mask][filtering_mask_unlabeled], 
                                      pesudo_y[tmp_data_unlabeled_mask][filtering_mask_unlabeled])
            loss += tmp_scale * ul_loss
        
        loss.backward()
    
        with torch.no_grad():
            model.eval()
            output = model(data.x, data.edge_index[:, train_edge_mask], None)
            val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask])
        
        optimizer.step()
        scheduler.step(val_loss)
        
        # evaluate the model
        accs, baccs, f1s, aurocs, pc_f1s = test()
        train_acc, val_acc, _ = accs
        train_f1, val_f1, _ = f1s
        val_acc_f1 = (val_acc + val_f1) / 2.
        if val_acc_f1 > best_val_acc_f1:
            best_val_acc_f1 = val_acc_f1
            test_acc = accs[2]
            test_bacc = baccs[2]
            test_f1 = f1s[2]
            test_auroc = aurocs[2]
            test_per_class_f1 = pc_f1s[2]
        
    avg_val_acc_f1.append(best_val_acc_f1)
    avg_test_acc.append(test_acc)
    avg_test_bacc.append(test_bacc)
    avg_test_f1.append(test_f1)
    avg_test_auc.append(test_auroc)
    avg_test_per_class_f1s.append(test_per_class_f1)

# calculate statistics
acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
auroc_CI = (statistics.stdev(avg_test_auc) / (repeatition ** (1/2)))
avg_acc = statistics.mean(avg_test_acc)
avg_bacc = statistics.mean(avg_test_bacc)
avg_f1 = statistics.mean(avg_test_f1)
avg_auroc = statistics.mean(avg_test_auc)
avg_val_acc_f1 = statistics.mean(avg_val_acc_f1)

avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, AUROC: {:.4f} +- {:.4f}, Val Acc F1: {:.4f}'
avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_auroc, auroc_CI, avg_val_acc_f1)
print(avg_log)

print('Total time: ', time.time()-start_time) 
