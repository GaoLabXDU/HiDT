import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def DSN2(t):
    a = t.sum(dim=1, keepdim=True)
    b = t.sum(dim=0, keepdim=True)
    lamb = torch.cat([a.squeeze(), b.squeeze()], dim=0).max()
    r = t.shape[0] * lamb - t.sum(dim=0).sum(dim=0)
    a = a.expand(-1, t.shape[1])
    b = b.expand(t.shape[0], -1)
    tt = t + (lamb ** 2 - lamb * (a + b) + a * b) / r
    ttmatrix = tt / tt.sum(dim=0)[0]
    ttmatrix = torch.where(t > 0, ttmatrix, t)
    return ttmatrix


def DSN(x):
    """Doubly stochastic normalization"""
    p = x.shape[0]
    y1 = []
    for i in range(p):
        y1.append(DSN2(x[i]))
    y1 = torch.stack(y1, dim=0)
    return y1


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """

    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # Encourage to use the below code when a deterministic result is
    # needed (reproducibility). However, the code below is with low efficiency.

    # tensor = torch.zeros(num_segments, data.shape[1], device=data.device)
    # for index in range(num_segments):
    #     tensor[index, :] = torch.sum(data[segment_ids == index, :], dim=0)
    # return tensor

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], device=data.device)).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape, device=data.device).scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor

# reference Beaconet (https://github.com/GaoLabXDU/Beaconet)
class BatchSpecificNorm(nn.Module):
    def __init__(self, n_batches, feature_dim, eps=1e-8):
        super(BatchSpecificNorm, self).__init__()
        self.scale = nn.Embedding(n_batches, feature_dim)
        self.shift = nn.Embedding(n_batches, feature_dim)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        self.eps = eps

    def forward(self, x, batch_idx):
        scale = self.scale(batch_idx)
        shift = self.shift(batch_idx)
        return x * scale + shift

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_attr):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = e * edge_attr
        attention = DSN(e)
        h_prime = []
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(attention[i], Wh))
        if self.concat:
            h_prime = torch.cat(h_prime, dim=1)
            return F.elu(h_prime), e
        else:
            h_prime = torch.stack(h_prime, dim=0)
            h_prime = torch.sum(h_prime, dim=0)
            return h_prime, e

    # compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""
    def __init__(self,
                 node_hidden_sizes,
                 input_size):

        super(GraphAggregator, self).__init__()
        self._node_hidden_sizes = node_hidden_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._graph_transform_sizes = node_hidden_sizes[-1]
        self._input_size = input_size
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes
        node_hidden_sizes[-1] = self._graph_state_dim * 2
        layer = [nn.Linear(self._input_size[0], 64)]
        layer.append(nn.ReLU())
        layer.append(nn.Linear(64, node_hidden_sizes[0]))
        MLP1 = nn.Sequential(*layer)
        layer = []
        layer.append(nn.Linear(self._graph_state_dim, 32))
        layer.append(nn.ReLU())
        layer.append(nn.Linear(32, 16))
        MLP2 = nn.Sequential(*layer)
        return MLP1, MLP2

    def forward(self, node_states, graph_idx):
        """Compute aggregated graph representations."""
        node_states_g = self.MLP1(node_states)
        gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
        node_states_g = node_states_g[:, self._graph_state_dim:] * gates
        n_graphs = max(graph_idx) + 1
        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)
        graph_states = self.MLP2(graph_states)
        return graph_states


class EdgeGNN(nn.Module):
    def __init__(self, node_hidden_dims, edge_feature_dim, dropout, alpha, nheads):
        super(EdgeGNN, self).__init__()
        self.attentions = [GraphAttentionLayer(node_hidden_dims[0], node_hidden_dims[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(node_hidden_dims[1] * nheads[0] * edge_feature_dim, node_hidden_dims[2], dropout=dropout, alpha=alpha,
                                           concat=False)
        self.bs_norm = BatchSpecificNorm(n_batches=8, feature_dim=16)
        self.bn1 = nn.BatchNorm1d(node_hidden_dims[2])
        self.bn2 = nn.BatchNorm1d(node_hidden_dims[1] * nheads[0] * edge_feature_dim)
        self.aggregator = GraphAggregator(node_hidden_sizes=[node_hidden_dims[3]], input_size=[node_hidden_dims[2]])
        self.batch_norm = nn.BatchNorm1d(node_hidden_dims[2])
    def forward(self, node_features, edge_features, graph_idx, depth_idx):
        x = node_features
        n_nodes = x.shape[0]
        splits = edge_features.split(1, dim=1)
        edge_attr = [split.view(n_nodes, n_nodes) for split in splits]
        edge_attr = torch.stack(edge_attr, dim=0)
        edge_attr = DSN(edge_attr)
        
        temp_x = []
        for att in self.attentions:
            inn_x, edge_attr = att(x, edge_attr)
            temp_x.append(inn_x)
        
        x = torch.cat(temp_x, dim=1)
        x, edge_attr = self.out_att(x, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.bs_norm(x, depth_idx)
        x = F.relu(x)
        graph_states = self.aggregator(x, graph_idx)
        graph_states = self.batch_norm(graph_states)
        return graph_states, edge_attr 