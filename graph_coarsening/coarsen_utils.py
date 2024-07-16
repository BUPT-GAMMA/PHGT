#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time   : 4/3/23 16:43
# @File   : coarsen_utils.py
# @Author : Zhiyuan Lu
# @Email  : luzy@bupt.edu.cn
from pygsp import graphs
import torch
import dgl
from graph_coarsening.coarsening_utils import coarsen
import numpy as np
import networkx as nx
import pickle


def get_coarsened_graph_from_dgl(hetero_g, args):
    if args.dataset == 'DBLP':
        hetero_g = dgl.remove_nodes(hetero_g, torch.arange(0, hetero_g.num_nodes(ntype='2')), ntype='2')

    if False:
        homo_g = dgl.to_homogeneous(hetero_g)
        adj = homo_g.adj(scipy_fmt='coo')
        G = graphs.Graph(adj)
        C, Gc, _, _ = coarsen(G, K=10, r=args.coarse_r, method='variation_neighborhood')

        # with open('./coarsen_cache.npz', 'wb') as f:
        #     np.save(f, (C, Gc))
        #
        # exit()
        #
        # with open('./coarsen_cache.npz', 'rb') as f:
        #     C, Gc = np.load(f, allow_pickle=True)

        g_coarsened = dgl.from_scipy(Gc.A)

        # g_coarsened.ndata['h'] = torch.matmul(C, homo_g.ndata['h'])
    else:
        file_name = args.dataset + '_' + str(args.num_cluster) + '.pkl'
        with open(file_name, 'rb') as f:
            C = pickle.load(f)
            g_coarsened = None

    C = torch.tensor(C / C.sum(1), dtype=torch.float32)  # .to_sparse()
    # set nan values to 0
    nan_mask = torch.isnan(C)
    C[nan_mask] = 0

    return C, g_coarsened
