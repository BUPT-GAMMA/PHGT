import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.simplehgn import myGAT


class GTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, dropout=0.5, temperature=1.0):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.dropout = dropout

        self.head_dim = self.embeddings_dimension // self.nheads

        self.temper = temperature

        self.linear_k = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_v = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_q = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout_att = nn.Dropout(self.dropout)
        self.dropout_mlp = nn.Dropout(self.dropout)
        self.dropout_msa = nn.Dropout(self.dropout)
        self.dropout_ffn = nn.Dropout(self.dropout)

        self.activation = nn.LeakyReLU(0.2)

        self.FFN1 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.FFN2 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.LN1 = nn.LayerNorm(embeddings_dimension)
        self.LN2 = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, mask=None, e=1e-12):
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)
        batch_size = k.size()[0]

        q_ = q.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)
        k_ = k.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)
        v_ = v.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)

        k_t = k_.transpose(2, 3)
        score = (q_ @ k_t) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = F.softmax(score / self.temper, dim=-1)
        score = self.dropout_att(score)
        context = score @ v_

        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.nheads)
        h_sa = self.activation(self.linear_final(h_sa))

        h_sa = self.dropout_msa(h_sa)
        h1 = self.LN1(h_sa + h)

        hf = self.activation(self.FFN1(h1))
        hf = self.dropout_mlp(hf)
        hf = self.FFN2(hf)
        hf = self.dropout_ffn(hf)

        h2 = self.LN2(h1+hf)
        return h2


class GT(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64,  num_layers=8, nheads=2, dropout=0, temper=1.0, hg=None, coarsen_mat=None, args=None, hgnn_params=None, g=None, id_dim=64):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''
        super(GT, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.id_dim = id_dim
        self.hgnn = myGAT(g,
                          hgnn_params['dim'],
                          hgnn_params['num_etypes'],
                          hgnn_params['in_dim'],
                          hgnn_params['dim'],
                          embeddings_dimension,
                          hgnn_params['num_layers'],
                          hgnn_params['num_heads'],
                          F.elu,
                          hgnn_params['dropout'],
                          hgnn_params['dropout'],
                          0.05,
                          True,
                          0.05)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension, bias=True) for in_dim in input_dimensions])
        self.dropout = dropout
        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension+self.id_dim, self.nheads, self.dropout, temperature=temper))
        self.predictor = nn.Linear(embeddings_dimension+id_dim, num_class, bias=False)
        self.hg = hg.to(args.device)
        self.coarsen_mat = coarsen_mat.to(args.device)
        self.global_token_flag = args.global_token_flag
        self.path_token_flag = args.path_token_flag
        self.node_token_flag = args.node_token_flag
        self.dataset = args.dataset
        self.target_node_id = nn.Embedding(1, id_dim)
        self.node_token_id = nn.Embedding(1, id_dim)
        self.path_token_id = nn.Embedding(1, id_dim)
        self.global_token_id = nn.Embedding(1, id_dim)

    def forward(self, features_list, e_feat, seqs, path_seqs, norm=False):
        features_list = self.hgnn(features_list, e_feat)

        h = []
        # for fc, feature in zip(self.fc_list, features_list):
        #     h.append(fc(feature))
        h.append(features_list)

        h.append(torch.zeros(1, self.embeddings_dimension, device=h[0].device))
        h = torch.cat(h, 0)
        h_coarse = h[:self.hg.num_nodes()]
        # separate process the DBLP dataset
        if self.dataset == 'DBLP':
            mask = (torch.arange(h_coarse.size(0)) < 18385) | (torch.arange(h_coarse.size(0)) >= 26108)
            h_coarse = h_coarse[mask]
        elif self.dataset == 'ACM':
            mask = (torch.arange(h_coarse.size(0)) < 3025) | (torch.arange(h_coarse.size(0)) >= 8984)
            h_coarse = h_coarse[mask]
        elif self.dataset == 'IMDB':
            mask = (torch.arange(h_coarse.size(0)) < 13449) | (torch.arange(h_coarse.size(0)) >= 21420)
            h_coarse = h_coarse[mask]

        graph_seq = torch.zeros([seqs.size(0), 0, self.embeddings_dimension + self.id_dim], device=h.device)

        # node token
        if self.node_token_flag:
            h_node = h[seqs]
            node_token_id = self.node_token_id.weight.repeat([h_node.size(0), h_node.size(1), 1])
            h_node = torch.cat([h_node, node_token_id], dim=2)
            graph_seq = torch.cat([graph_seq, h_node], dim=1)

        # path token
        if self.path_token_flag:
            h_ins = h[path_seqs]
            h_ins = torch.mean(h_ins, dim=2)
            path_token_id = self.path_token_id.weight.repeat([h_ins.size(0), h_ins.size(1), 1])
            h_ins = torch.cat([h_ins, path_token_id], dim=2)
            graph_seq = torch.cat([graph_seq, h_ins], dim=1)

        # global token
        if self.global_token_flag:
            global_seq = torch.matmul(self.coarsen_mat, h_coarse).repeat([seqs.size(0), 1, 1])
            global_token_id = self.global_token_id.weight.repeat([global_seq.size(0), global_seq.size(1), 1])
            global_seq = torch.cat([global_seq, global_token_id], dim=2)
            graph_seq = torch.cat([graph_seq, global_seq], dim=1)

        h = graph_seq
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        output = self.predictor(h[:, 0, :])

        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output
