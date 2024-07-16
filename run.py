import argparse
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from models.Transformer import GT
from utils.data import load_data, SeqDataset
from utils.pytorchtools import EarlyStopping
from utils.data import model_config, dataset_config
from torch.utils.data import DataLoader
from utils.preprocess import ego_network_sampling_with_truncate, gen_seq_hetero, feature_padding, gen_path_seq_hetero
from graph_coarsening.coarsen_utils import get_coarsened_graph_from_dgl
import datetime
import gc
import json
import argparse
from argparse import Namespace


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append('utils/')


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def infer_batch(model, loader, features_list, e_feat, args):
    logits_list = []
    labels_list = []
    for batch_node, batch_path, batch_label in loader:
        logits = model(features_list, e_feat, batch_node, batch_path, args.l2norm)
        logits_list.append(logits)
        labels_list.append(batch_label)
    logits_list = torch.cat(logits_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)
    return logits_list, labels_list


def run_model(args, hgnn_params):
    torch.use_deterministic_algorithms(True)
    seed_all(args.seed)  # fix random seed
    print(args)
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    checkpoint_path = 'checkpoint/' + post_fix + '/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    device = args.device

    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset, args)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
        hgnn_params['in_dim'] = in_dims
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3 or feats_type == 6 or feats_type == 7:
        in_dims = [features.shape[0] for features in features_list]
        hgnn_params['in_dim'] = in_dims
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
        if feats_type == 6:
            features_list = feature_padding(features_list)
            in_dims = [features.shape[1] for features in features_list]
        if feats_type == 7:
            in_dims = [args.hidden_dim for _ in range(len(features_list))]

    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k + 1 + len(dl.links['count'])

    g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    # Graph coarsening
    if args.global_token_flag:
        coarsen_mat, g_coarsened = get_coarsened_graph_from_dgl(dl.hg, args)
    else:
        coarsen_mat, g_coarsened = (torch.tensor([]), torch.tensor([]))

    # ego-network sampling
    sub_g_list, target_list, nid_list = ego_network_sampling_with_truncate(dl.hg, k=args.ego_radius, args=args)
    node_seq = gen_seq_hetero(dl.hg, sub_g_list, target_list, nid_list, args, seq_len=args.node_len)
    if args.path_token_flag:
        path_seq = gen_path_seq_hetero(dl.hg, sub_g_list, target_list, nid_list, labels, train_val_test_idx, args, seq_len=args.path_len)
    else:
        path_seq = torch.zeros([node_seq.size(0), node_seq.size(1), 3], dtype=torch.long)

    # use official split in HGB
    train_seq = node_seq[train_idx]
    val_seq = node_seq[val_idx]
    test_seq = node_seq[test_idx]

    train_path_seq = path_seq[train_idx]
    val_path_seq = path_seq[val_idx]
    test_path_seq = path_seq[test_idx]

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    num_classes = dl.labels_train['num_classes']
    hgnn_params['num_etypes'] = len(dl.links['count'])*2+1

    train_dataset = SeqDataset(train_seq, train_path_seq, labels[train_idx])
    val_dataset = SeqDataset(val_seq, val_path_seq, labels[val_idx])
    test_dataset = SeqDataset(test_seq, test_path_seq, torch.tensor(dl.labels_test['data'][test_idx]))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=0)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=0)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=0)

    for i in range(args.repeat):
        net = GT(num_classes, in_dims, args.hidden_dim, args.num_layers,  args.num_heads, args.dropout, temper=args.temperature, hg=dl.hg, coarsen_mat=coarsen_mat, args=args, hgnn_params=hgnn_params, g=g, id_dim=args.id_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=(checkpoint_path + 'Transformer_{}_{}_{}.pt').format(args.dataset, args.num_layers, args.device))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            train_loss_sum = 0
            for seq_batch, seq_path_batch, labels_batch in train_loader:
                seq_batch = seq_batch.to(device)
                seq_path_batch = seq_path_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits = net(features_list, e_feat, seq_batch, seq_path_batch, args.l2norm)
                if args.dataset == 'IMDB':
                    train_loss = F.binary_cross_entropy(torch.sigmoid(logits), labels_batch.float())
                else:
                    logp = F.log_softmax(logits, 1)
                    train_loss = F.nll_loss(logp, labels_batch)


                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_loss_sum += train_loss.item()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss_sum, t_end-t_start))

            t_start = time.time()

            # validation
            net.eval()
            with torch.no_grad():
                logits, labels_val = infer_batch(net, val_loader, features_list, e_feat, args)
                if args.dataset == 'IMDB':
                    val_loss = F.binary_cross_entropy(torch.sigmoid(logits), labels_val.to(args.device).float())
                    pred = (logits.cpu().numpy() > 0).astype(int)
                    eval_result = dl.evaluate_valid(pred, labels_val)
                else:
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp, labels_val.to(args.device))
                    pred = logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    pred = onehot[pred]
                    eval_result = dl.evaluate_valid(pred, F.one_hot(labels_val, num_classes=args.num_class))
                print(eval_result)
    
            scheduler.step(val_loss)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            (checkpoint_path + 'Transformer_{}_{}_{}.pt').format(args.dataset, args.num_layers, args.device)))
        net.eval()
        with torch.no_grad():
            logits, labels_test = infer_batch(net, test_loader, features_list, e_feat, args)
            test_logits = logits
            if args.mode == 1:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                if args.dataset == 'IMDB':
                    pred = (logits.cpu().numpy() > 0).astype(int)
                else:
                    pred = test_logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    pred = onehot[pred]
                result = dl.evaluate_valid(pred, labels_test)
                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']

    print('Micro-f1: %.4f, std: %.4f' % (micro_f1.mean().item(), micro_f1.std().item()))
    print('Macro-f1: %.4f, std: %.4f' % (macro_f1.mean().item(), macro_f1.std().item()))

    # log the configurations
    with open('./exp_log.txt', 'a') as f:
        out1 = str(args) + '\r\n'
        out2 = args.model + '_' + args.dataset + '_' + str(args.seed) \
              + ':' + ' micro-f1: ' + str(micro_f1.mean().item()) + ' std: ' + str(micro_f1.std().item())\
              + ' macro-f1: ' + str(macro_f1.mean().item()) + ' std: ' + str(macro_f1.std().item())\
              + ' micro-f1-each-repeat: ' + str(micro_f1.tolist()) + ' macro-f1-each-repeat: ' + str(macro_f1.tolist()) + '\r\n'
        f.write(out1 + out2)

    # empty cache
    del optimizer, net, features_list, e_feat
    gc.collect()
    torch.cuda.empty_cache()

    return micro_f1.mean().item()


def load_params(json_file):
    json_file += '.json'
    json_file = os.path.join('./configs', json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params

def dict_to_namespace(config_dict):
    return Namespace(**config_dict)

 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Transformer')
    ap.add_argument('--config', type=str, default='DBLP', help='JSON config file.')

    args = ap.parse_args()
    args = load_params(args.config)
    args = dict_to_namespace(args)

    args = dataset_config(args)
    args = model_config(args)
    args.device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    hgnn_params = {'num_layers': args.gnn_layers,
                   'dim': args.gnn_dim,
                   'dropout': args.gnn_dropout,
                   'num_heads': args.gnn_heads}
    hgnn_params['num_heads'] = (hgnn_params['num_layers'] - 1) * [hgnn_params['num_heads']] + [1]

    run_model(args, hgnn_params)
