import dgl
import networkx as nx
import numpy as np
import torch


def dgl2nx(dgl_g):
    '''
    example:
    >> nx_g, canonical_etypes = dgl2nx(hg)
    Parameters
    ----------
    dgl_g

    Returns
    -------

    '''
    nx_g = nx.Graph()

    for edge in dgl_g.canonical_etypes:
        src_type = edge[0]
        dst_type = edge[2]
        edge_type = edge[1]
        src, dst = dgl_g.edges(etype=edge_type)
        src = src.numpy().astype(np.str_)
        dst = dst.numpy().astype(np.str_)

        src_str = np.array([src_type + src_i for src_i in src])
        dst_str = np.array([dst_type + dst_i for dst_i in dst])
        edge_list = [tuple(edge) for edge in zip(src_str, dst_str)]
        nx_g.add_edges_from(edge_list)

    # add feature to the networkx graph
    for node in list(nx_g.nodes):
        node_type = node[0]
        node_idx = int(node[1:])
        node_feature = dgl_g.nodes[node_type].data['hde'][node_idx]
        nx_g.nodes[node]['hde'] = node_feature
        node_feature = dgl_g.nodes[node_type].data['is_target'][node_idx]
        nx_g.nodes[node]['is_target'] = node_feature
        node_feature = dgl_g.nodes[node_type].data['h'][node_idx]
        nx_g.nodes[node]['h'] = node_feature

    return nx_g, dgl_g.canonical_etypes


def get_feature_dim(nx_g, node_type, fea_name):
    dim = -1
    for node in list(nx_g.nodes):
        if node[0] == node_type:
            tmp = nx_g.nodes[node][fea_name].size()
            if len(tmp) == 0:
                dim = 1
            else:
                dim = tmp[0]
            break
        else:
            pass

    return dim


def node_edge_matching(src, dst, canonical_etypes):
    etype = ''
    for canonical_etype in canonical_etypes:
        if canonical_etype[0] == src and canonical_etype[2] == dst:
            etype = canonical_etype
    assert etype != ''

    return etype


# check if one nx_g has a specific node type
def has_node_type(nx_g, node_type):
    has_node_type_flag = False
    for node in list(nx_g.nodes):
        if node[0] == node_type:
            has_node_type_flag = True
            break
        else:
            pass

    return has_node_type_flag


def nx2dgl(nx_g, canonical_etypes):
    '''
    example:
    >> dgl_g = nx2dgl(nx_g, canonical_etypes)
    if the node doesn't
    Parameters
    ----------
    nx_g
    canonical_etypes

    Returns
    -------

    '''
    metagraph_data = {}
    node_type_list = []
    for edge_type in canonical_etypes:
        metagraph_data[edge_type] = ([], [])
        node_type_list.append(edge_type[0])
        node_type_list.append(edge_type[2])
    node_type_list = list(set(node_type_list))

    edge_list = list(nx_g.edges)
    src_list = []
    dst_list = []
    for edge in edge_list:
        src_idx = int(edge[0][1:])
        dst_idx = int(edge[1][1:])
        src_type = edge[0][0]
        dst_type = edge[1][0]
        edge_type = node_edge_matching(src_type, dst_type, canonical_etypes)
        metagraph_data[edge_type][0].append(src_idx)
        metagraph_data[edge_type][1].append(dst_idx)
        edge_type = node_edge_matching(dst_type, src_type, canonical_etypes)
        metagraph_data[edge_type][0].append(dst_idx)
        metagraph_data[edge_type][1].append(src_idx)

    dgl_g = dgl.heterograph(metagraph_data)


    # add feature to the dgl graph
    cnt = 0
    for node_type in node_type_list:
        if has_node_type(nx_g, node_type):
            tmp_feature = torch.randn(dgl_g.number_of_nodes(node_type), get_feature_dim(nx_g, node_type, 'hde'))
            dgl_g.nodes[node_type].data['hde'] = tmp_feature
            tmp_feature = torch.zeros(dgl_g.number_of_nodes(node_type), get_feature_dim(nx_g, node_type, 'is_target')).bool()
            dgl_g.nodes[node_type].data['is_target'] = tmp_feature
            tmp_feature = torch.zeros(dgl_g.number_of_nodes(node_type), get_feature_dim(nx_g, node_type, 'h')).float()
            dgl_g.nodes[node_type].data['h'] = tmp_feature
        for node in dgl_g.nodes(ntype=node_type):
            node_idx = int(str(node)[7:-1])
            node_str = node_type + str(node_idx)
            if nx_g.has_node(node_str):
                node_feature = nx_g.nodes[node_str]['hde']
                dgl_g.nodes[node_type].data['hde'][node_idx] = node_feature
                node_feature = nx_g.nodes[node_str]['is_target']
                dgl_g.nodes[node_type].data['is_target'][node_idx] = node_feature
                node_feature = nx_g.nodes[node_str]['h']
                dgl_g.nodes[node_type].data['h'][node_idx] = node_feature
            else:
                cnt = cnt + 1
    # print("nx2dgl(): MISSING NODE IN GRAPH COUNT:", cnt)

    return dgl_g


def add_dist_feature(set_index, g, args):
    nx_g, etype = dgl2nx(g)

    target_a = '0' + str(set_index)

    for node in nx_g.nodes:
        type_feature = type_encoder(node, args)
        type_feature = torch.tensor(type_feature)
        if args.hde:
            tmp_a = dist_encoder(target_a, node, nx_g, args)
            dist_feature = torch.tensor(tmp_a)
            feature = torch.cat([dist_feature, type_feature])
        else:
            feature = type_feature
        nx_g.nodes[node]['hde'] = feature

    dgl_g = nx2dgl(nx_g, etype)

    if args.is_hetero:
        out_g = dgl_g
    else:
        out_g = dgl.to_homogeneous(dgl_g, ndata=['hde', 'is_target', 'h'])

    return out_g


def dist_encoder(src, dest, G, args):
    NODE_TYPE = args.num_node_type
    type2idx = args.node_dict
    max_dist = args.max_dist

    # 计算在各个类型下的SPD=最少出现次数
    paths = list(nx.all_simple_paths(G, src, dest, cutoff=max_dist+1))
    cnt = [max_dist] * NODE_TYPE  # 超过max_spd的默认截断
    for path in paths:
        res = [0] * NODE_TYPE
        for i in range(1, len(path)):
            tmp = path[i][0]
            res[type2idx[tmp]] += 1
        # print(path, res)
        for k in range(NODE_TYPE):
            cnt[k] = min(cnt[k], res[k])

    one_hot_list = [np.eye(max_dist + 1, dtype=np.float64)[cnt[i]] for i in range(NODE_TYPE)]

    return np.concatenate(one_hot_list)


def type_encoder(node, args):
    NODE_TYPE = args.num_node_type
    type2idx = args.node_dict
    res = [0] * NODE_TYPE
    res[type2idx[node[0]]] = 1.0
    return res
