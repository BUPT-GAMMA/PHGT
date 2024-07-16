import pickle
import sys

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
from torch.utils.data import Dataset


def load_data(prefix, args):
    from data_loader import data_loader
    dl = data_loader('data/'+prefix, args)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB' and prefix != 'IMDB-HGB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl


def dataset_config(args):
    if args.dataset == 'DBLP':
        args.edge_dict = {'0-1': 0, '1-0': 1, '1-2': 2, '2-1': 3, '1-3': 4, '3-1': 5}  # DBLP
        args.n_in_dict_ori = {'0': 4057, '1': 14328, '2': 7723, '3': 20}
        args.n_in_dict = {'0': 4057, '1': 14328, '2': 7723, '3': 20}  # DBLP type 3
        args.node_dict = {'0': 0, '1': 1, '2': 2, '3': 3}
        args.n_in_dct_rgcn = {'0-1': 334, '1-0': 4230, '1-2': 4230, '2-1': 50, '1-3': 4230, '3-1': 1}
        args.num_class = 4
        args.meta_path_list = ['0-1-0', '0-1-2', '0-1-3']
        args.meta_path_reach_list = ['0-1', '1-0']
        args.max_seq_len = 128
    elif args.dataset == 'IMDB':
        args.edge_dict = {'0-1': 0, '1-0': 1, '0-2': 2, '2-0': 3, '0-3': 4, '3-0': 5}  # IMDB
        if args.feats_type == 3:
            args.n_in_dict = {'0': 4932, '1': 2393, '2': 6124, '3': 7971}  # IMDB type 3
        elif args.feats_type == 0:
            args.n_in_dict = {'0': 3489, '1': 3341, '2': 3341, '3': 7971}  # IMDB type 0
        args.node_dict = {'0': 0, '1': 1, '2': 2, '3': 3}
        args.num_class = 5
        args.meta_path_list = ['0-1-0', '0-2-0', '0-3-0']
    elif args.dataset == 'Freebase':
        args.edge_dict = {'0-1': 0, '1-0': 1, '0-2': 2, '2-0': 3, '0-3': 4, '3-0': 5}  # Freebase
        args.n_in_dict_ori = {'0': 3492, '1': 33401, '2': 2502, '3': 4459}
        args.n_in_dict = {'0': 3492, '1': 33401, '2': 2502, '3': 4459}  # Freebase
        args.node_dict = {'0': 0, '1': 1, '2': 2, '3': 3}
        args.num_class = 3
        args.meta_path_list = ['0-1-0', '0-2-0', '0-3-0']
    elif args.dataset == 'AMiner':
        args.edge_dict = {'0-1': 0, '1-0': 1, '0-2': 2, '2-0': 3}  #
        args.n_in_dict_ori = {'0': 6564, '1': 13329, '2': 35890}
        args.n_in_dict = {'0': 6564, '1': 13329, '2': 35890}  #
        args.node_dict = {'0': 0, '1': 1, '2': 2}
        args.num_class = 4
        args.meta_path_list = ['0-1-0', '0-2-0']
    elif args.dataset == 'MAG':
        args.n_in_dict_ori = {'0': 128, '1': 128, '2': 128, '3': 128}
        args.n_in_dict = {'0': 128, '1': 128, '2': 128, '3': 128}
        args.node_dict = {'0': 0, '1': 1, '2': 2, '3': 3}
        args.num_class = 349
    elif args.dataset == 'ACM':
        args.node_dict = {'0': 0, '1': 1, '2': 2, '3': 3}
        args.edge_dict = {'0-0': 0, '0-1': 1, '1-0': 2, '0-2': 3, '2-0': 4, '0-3': 5, '3-0': 6}
        args.n_in_dict = {'0': 3025, '1': 5959, '2': 56, '3': 1902}
        args.num_class = 3
        args.meta_path_list = ['0-1-0', '0-2-0']
    else:
        raise ValueError('configuration for {} not assigned!'.format(args.dataset))

    args.num_node_type = len(args.node_dict.keys())

    return args


def model_config(args):
    hetero_models = ['PHGT']
    if args.model in hetero_models:
        args.is_hetero = True
        args.hete_or_homo = 'hete'
    else:
        raise ValueError('Model specification error!')

    if args.hde:
        args.hde_or_not = 'hde'
    else:
        args.hde_or_not = 'nohde'

    args.k_hop = 2
    args.max_dist = args.k_hop + 1
    args.hde_dim = args.num_node_type * (args.max_dist + 1) + args.num_node_type

    return args


class SeqDataset(Dataset):
    def __init__(self, seqs, path_seq, labels):
        super(SeqDataset, self).__init__()
        self.seqs = seqs.cpu()
        self.path_seq = path_seq.cpu()
        self.labels = labels.cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.path_seq[idx], self.labels[idx]
