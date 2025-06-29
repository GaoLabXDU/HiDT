import numpy as np
import pandas as pd
import torch

def load_TAD_region(TADfile1, TADfile2):
    # load diffTADs
    region1 = pd.read_csv(TADfile1, sep='\t')
    region2 = pd.read_csv(TADfile2, sep='\t')
    region2 = region2[['chrom', 'start', 'end']]
    # combine
    all_TAD_region = pd.concat([region1, region2], ignore_index=True)
    all_TAD_region = all_TAD_region.sort_values(by=['chrom', 'start'])
    return all_TAD_region

def Slice_matrix(mat, start, end):
    # Getting matrix slice with a TAD
    row = mat.shape[0]
    if start > row or end > row:
        print(start, end, row)
        raise ValueError("invalid TAD boundary")
    else:
        cut_mat = mat[start:end + 1, start:end + 1]
    return cut_mat.toarray()

def filter_matrix(mat):
    # filter invalid TADs (with outlier)
    col_sum = np.sum(mat, axis=0)
    rows = np.shape(mat)[0]
    if np.any(col_sum < 0.1):
        return False
    else:
        return True
    
def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def get_graphs(graphs, n_features, graph_idx, depth_idx, labels, edge_features):
    adj = torch.FloatTensor(graphs)
    flattend_adj = adj.view(-1)
    reshaped_adj = flattend_adj.view(-1, 1)
    edge_features = reshaped_adj.repeat(1, edge_features)
    node_features = torch.FloatTensor(n_features)
    graph_idx = torch.from_numpy(graph_idx).long()
    depth_idx = torch.from_numpy(depth_idx).long()
    labels = torch.FloatTensor(labels)
    return edge_features, node_features, graph_idx, depth_idx, labels

def pack_batches(graphs, depths):
    n_graph = len(graphs)
    # init adj matrix
    sum_node = 0
    for i in range(n_graph):
        cur_graph = graphs[i][0]
        cur_node = np.shape(cur_graph)[0]
        sum_node += cur_node
    combine_adj = np.zeros((sum_node*2, sum_node*2))
    # add
    graph_idx = []
    depth_idx = []
    cur_row = 0
    idx = 0
    for i in range(n_graph):
        graph_1 = graphs[i][0]
        graph_2 = graphs[i][1]
        cur_depth1 = depths[i][0]
        cur_depth2 = depths[i][1]
        cur_node = np.shape(graph_1)[0]
        combine_adj[cur_row:cur_row + cur_node, cur_row:cur_row + cur_node] = graph_1
        cur_row += cur_node
        combine_adj[cur_row:cur_row + cur_node, cur_row:cur_row + cur_node] = graph_2
        cur_row += cur_node
        graph_idx.append(np.ones(cur_node, dtype=np.int32) * idx)
        depth_idx.append(np.ones(cur_node, dtype=np.int32) * cur_depth1)
        idx += 1
        graph_idx.append(np.ones(cur_node, dtype=np.int32) * idx)
        depth_idx.append(np.ones(cur_node, dtype=np.int32) * cur_depth2)
        idx += 1
        
    depth_idx = np.concatenate(depth_idx, axis=0)
    graph_idx = np.concatenate(graph_idx, axis=0)
    node_features = np.ones((sum_node*2, 8), dtype=np.float64)
    return combine_adj, node_features, graph_idx, depth_idx

def generate_valid_batches(idx, graphs, labels, depths, bs):
    batch_graphs = graphs[idx:idx + bs]
    batch_depths = depths[idx:idx + bs]
    batch_graphs, batch_features, graphs_idx, depth_idx = pack_batches(batch_graphs, batch_depths)
    batch_labels = labels[idx:idx + bs]
    return batch_graphs, batch_features, batch_labels, graphs_idx, depth_idx