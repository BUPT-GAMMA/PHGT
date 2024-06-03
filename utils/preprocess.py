import numpy as np
import scipy.sparse
import networkx as nx
import torch
from utils.HDE import add_dist_feature
from tqdm import tqdm
import dgl
import os
import pickle
import pandas as pd


def get_metapath_adjacency_matrix(adjM, type_mask, metapath):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param metapath
    :return: a list of metapath-based adjacency matrices
    """
    out_adjM = scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[0], type_mask == metapath[1])])
    for i in range(1, len(metapath) - 1):
        out_adjM = out_adjM.dot(scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])]))
    return out_adjM.toarray()


# networkx.has_path may search too
def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    outs = []
    for metapath in expected_metapaths:
        # consider only the edges relevant to the expected metapath
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)
        partial_g_nx = nx.from_numpy_matrix((M * mask).astype(int))

        # only need to consider the former half of the metapath
        # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
        metapath_to_target = {}
        for source in (type_mask == metapath[0]).nonzero()[0]:
            for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:
                # check if there is a possible valid path from source to target node
                has_path = False
                single_source_paths = nx.single_source_shortest_path(
                    partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)
                if target in single_source_paths:
                    has_path = True

                #if nx.has_path(partial_g_nx, source, target):
                if has_path:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]
                    if len(shortests) > 0:
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests
        metapath_neighbor_paris = {}
        for key, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        p1 + p2[-2::-1]]
        outs.append(metapath_neighbor_paris)
    return outs


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    G_list = []
    for metapaths in neighbor_pairs:
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])
                edge_count += 1
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
        print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array


def get_new_set_index_hetero(data_list, ntypes):
    set_index_list = []
    for data in data_list:
        is_target_data_list = []
        for etype in ntypes:
            is_target_data_list.append(data.nodes[etype].data['is_target'])
        src = is_target_data_list[0].numpy()
        src_index = np.argwhere(src == True).reshape([-1])
        assert len(src_index) == 1
        set_index_list.append(int(src_index))

    return set_index_list


def get_new_set_index_homo(data_list):
    set_index_list = []
    for data in data_list:
        is_target = data.ndata['is_target'].numpy().reshape([-1])
        target_index = np.argwhere(is_target == True).reshape([-1])
        assert len(target_index) == 1
        set_index_list.append(target_index)

    return set_index_list


def ego_network_sampling_with_truncate(g, k, args):
    file_name_pre = 'subgraph_cache_' + args.dataset + '_' + str(k) + 'hop' + '_' + args.hete_or_homo + '_' + args.hde_or_not + '_dropedge' + str(args.drop_edge)
    file_name_pkl = file_name_pre + '.pkl'

    for node_type in g.ntypes:
        g.nodes[node_type].data['is_target'] = torch.zeros((g.num_nodes(ntype=node_type),)).bool()
        g.nodes[node_type].data['hde'] = torch.zeros((g.num_nodes(ntype=node_type), args.hde_dim))

    if not os.path.exists(file_name_pkl):
        glist = []
        target_list = []
        nid_list = []
        for node in tqdm(g.nodes(ntype='0'), desc='Ego-network sampling ...'):
            sub_g, new_idx = dgl.khop_out_subgraph(g, {'0': node}, k=k, relabel_nodes=True)
            nid_list.append(sub_g.ndata[dgl.NID])
            sub_g.nodes['0'].data['is_target'][new_idx['0']] = 1
            glist.append(sub_g)
            target_list.append(new_idx)

        t_list = get_new_set_index_hetero(glist, '0')
        g_list = []
        total = len(t_list)
        for data, idx in tqdm(zip(glist, t_list), total=total, desc='Computing HDE feature ...'):
            if args.hde:
                g = add_dist_feature(idx, data, args)
                g_list.append(g)
            else:
                g_list.append(data)

        if not args.is_hetero:
            t_list = get_new_set_index_homo(g_list)

        with open(file_name_pkl, 'wb') as h:
            pickle.dump((g_list, t_list, nid_list), h)

    with open(file_name_pkl, 'rb') as h:
        g_list, t_list, nid_list = pickle.load(h)

    return g_list, t_list, nid_list


def get_key(dct, value):
    index = None
    for key in dct.keys():
        if np.isin(value, dct[key]).any():
            index = np.argwhere(dct[key] == value)[0]
            out_key = key
    assert index is not None
    return out_key, index


def idx_mapping(ntype, type_idx, hg):
    base_num = 0
    ntypes = list(hg.ntypes)
    n_t_idx = ntypes.index(ntype)
    for n_t in ntypes[:n_t_idx]:
        base_num += hg.num_nodes(ntype=n_t)

    return base_num + type_idx


def gen_seq_hetero(hg, data_list, target_list, nid_list, args, seq_len=128):
    seq_list = []
    for idx, (data, target, nid) in tqdm(enumerate(zip(data_list, target_list, nid_list)),
                                         total=len(data_list), desc='Truncating ...'):
        seq = []
        for n_t in data.ntypes:
            type_id = nid[n_t]
            glob_id = idx_mapping(n_t, type_id, hg)
            seq.append(glob_id)
        seq = torch.cat(seq)

        # swap the first node and the target node
        tar = torch.tensor(target)
        target_idx = seq[tar].item()
        seq[tar] = seq[0]
        seq[0] = target_idx

        if len(seq) > seq_len:
            seq = seq[:seq_len]

        pad_len = seq_len - len(seq)
        pad_idx = torch.zeros((pad_len,), dtype=torch.int) - 1
        seq = torch.cat([seq, pad_idx])

        seq_list.append(seq)

    seq_list = torch.stack(seq_list)
    return seq_list


def get_edge_type_list(path):
    n_t_list = path.split('-')
    e_t_list = []
    for i in range(len(n_t_list)-1):
        e_t_list.append(n_t_list[i] + '-' + n_t_list[i+1])

    return e_t_list


def get_path_instance(g, metapath_list):
    etype_idx_dict = {}
    for etype in g.etypes:
        edges_idx_i = g.edges(etype=etype)[0].cpu().numpy()
        edges_idx_j = g.edges(etype=etype)[1].cpu().numpy()
        etype_idx_dict[etype] = pd.DataFrame([edges_idx_i, edges_idx_j]).T
        _etype = etype.split('-')
        etype_idx_dict[etype].columns = [_etype[0], _etype[1]]

    res = {}
    for metapath in metapath_list:
        res[metapath] = None
        _metapath = metapath.split('-')
        for i in range(1, len(_metapath) - 1):
            if i == 1:
                res[metapath] = etype_idx_dict['-'.join(_metapath[:i + 1])]
            feat_j = etype_idx_dict['-'.join(_metapath[i:i + 2])]
            col_i = res[metapath].columns[-1]
            col_j = feat_j.columns[0]
            res[metapath] = pd.merge(res[metapath], feat_j,
                                     left_on=col_i,
                                     right_on=col_j,
                                     how='inner')
            if col_i != col_j:
                res[metapath].drop(columns=col_j, inplace=True)
        res[metapath] = res[metapath].values

    return res


def get_path_instance_with_src(g, metapath_list, src_index):
    etype_idx_dict = {}
    for etype in g.etypes:
        edges_idx_i = g.edges(etype=etype)[0].cpu().numpy()
        edges_idx_j = g.edges(etype=etype)[1].cpu().numpy()
        etype_idx_dict[etype] = pd.DataFrame([edges_idx_i, edges_idx_j]).T
        _etype = etype.split('-')
        etype_idx_dict[etype].columns = [_etype[0], _etype[1]]

    res = {}
    for metapath in metapath_list:
        res[metapath] = None
        _metapath = metapath.split('-')
        for i in range(1, len(_metapath) - 1):
            if i == 1:
                res[metapath] = etype_idx_dict['-'.join(_metapath[:i + 1])]
            feat_j = etype_idx_dict['-'.join(_metapath[i:i + 2])]
            col_i = res[metapath].columns[-1]
            col_j = feat_j.columns[0]
            res[metapath] = pd.merge(res[metapath], feat_j,
                                     left_on=col_i,
                                     right_on=col_j,
                                     how='inner')
            if col_i != col_j:
                res[metapath].drop(columns=col_j, inplace=True)
        res[metapath] = res[metapath].values
        for k, ins in reversed(list(enumerate(res[metapath]))):
            if ins[0] != src_index:
                res[metapath] = np.delete(res[metapath], k, axis=0)
        res[metapath] = torch.tensor(res[metapath])

    return res


def idx_mapping_ins(ntype, type_idx, hg, seq, meta_path):
    path_node_types = meta_path.split('-')
    seq_out = []
    for ins in seq:
        ins_glob = torch.zeros(len(ins))
        for i, (sub_id, n_t) in enumerate(zip(ins, path_node_types)):
            ins_glob[i] = type_idx[n_t][sub_id]
            # get the shift of the node index
            n_t_idx = ntype.index(n_t)
            base_num = 0
            for n_t_i in ntype[:n_t_idx]:
                base_num += hg.num_nodes(ntype=n_t_i)
            ins_glob[i] += base_num
        seq_out.append(ins_glob)
    seq_out = torch.stack(seq_out)
    return seq_out


def gen_path_seq_hetero(hg, data_list, target_list, nid_list, label_list, train_val_test_idx, args, seq_len=128):
    seq_list = []
    for idx, (data, target, nid, label) in tqdm(enumerate(zip(data_list, target_list, nid_list, label_list)),
                                                total=len(data_list), desc='Truncating ...'):
        seq = []
        path_instance = get_path_instance_with_src(data, args.meta_path_list, target)
        for path in path_instance.keys():
            sub_seq_type = path_instance[path]
            if sub_seq_type.size(0) == 0:
                continue
            sub_seq_type = idx_mapping_ins(data.ntypes, nid, hg, sub_seq_type, path)
            seq.append(sub_seq_type)
        seq = torch.cat(seq, dim=0)

        if len(seq) > seq_len-1:
            seq = seq[:seq_len-1]

        ego_token = torch.tensor(idx).repeat(3).unsqueeze(0)
        seq = torch.cat([ego_token, seq])
        pad_len = seq_len - len(seq)
        pad_tensor = torch.zeros((pad_len, seq.shape[1]), dtype=torch.int) - 1
        seq = torch.cat([seq, pad_tensor], dim=0)
        seq_list.append(seq)

    seq_list = torch.stack(seq_list).long()
    return seq_list


def feature_padding(feature_list):
    len_list = []
    for feature in feature_list:
        len_list.append(feature.size(1))
    max_len = max(len_list)
    for i, feature in enumerate(feature_list):
        pad_len = max_len - feature.size(1)
        pad_tensor = torch.zeros((feature.size(0), pad_len), device=feature.device)
        feature_list[i] = torch.cat([feature.to_dense(), pad_tensor], dim=1)

    return feature_list


def gen_node_and_ins_seq(hg, data_list, target_list, nid_list, label_list, train_val_test_idx, args, seq_len=128):
    seq_list = []
    for idx, (data, target, nid, label) in tqdm(enumerate(zip(data_list, target_list, nid_list, label_list)),
                                                total=len(data_list),
                                                desc='Truncating ...'):
        seq = []
        path_instance = get_path_instance_with_src(data, args.meta_path_list, target)
        for path in path_instance.keys():
            sub_seq_type = path_instance[path]
            sub_seq_type = idx_mapping_ins(data.ntypes, nid, hg, sub_seq_type, path)
            seq.append(sub_seq_type)
        seq = torch.cat(seq, dim=0)

        if len(seq) > seq_len - 1:
            seq = seq[:seq_len - 1]

        ego_token = torch.tensor(idx).repeat(3).unsqueeze(0)
        seq = torch.cat([ego_token, seq])
        pad_len = seq_len - len(seq)
        pad_tensor = torch.zeros((pad_len, seq.shape[1]), dtype=torch.int) - 1
        seq = torch.cat([seq, pad_tensor], dim=0)
        seq_list.append(seq)

    seq_list = torch.stack(seq_list).long()
    return seq_list
