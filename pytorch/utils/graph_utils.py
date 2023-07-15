import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle
import random

def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def incidence_matrix(adj_list):
    """
    adj_list: List of sparse adjacency matrices
    """
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def ssp_multigraph_to_dgl(graph):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl = dgl.from_networkx(g_nx,edge_attrs=['type'])
    g_dgl.ndata['idx'] = torch.LongTensor(np.arange(g_dgl.num_nodes()))
    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs, r_labels, g_labels = map(list, zip(*samples))
    
    batched_graph = dgl.batch(graphs)
    
    return batched_graph, r_labels, g_labels


def move_batch_to_device_dgl(batch, device,multi_type=0):
    g_dgl_pos, r_labels_pos, targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    if multi_type==1:
        r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    elif multi_type==2:
        r_labels_pos = torch.FloatTensor(r_labels_pos).to(device=device)
    g_dgl_pos = g_dgl_pos.to(device)

    return g_dgl_pos, r_labels_pos, targets_pos

def _sp_row_vec_from_idx_list(idx_list, dim):
    """
    Create sparse vector of dimensionality dim from a list of indices.
    """
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def _get_neighbors(adj, nodes):
    """
    Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph
    """
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)

def get_neighbors(dgl_graphs, nodes):
    src,dst,eid = dgl_graphs.out_edges(nodes, form='all')
    sampled_edges = torch.cat([src.unsqueeze(1),dst.unsqueeze(1),eid.unsqueeze(1)],dim=1).to(device=nodes.device)
    new_nodes, new_index = torch.unique(sampled_edges[:,1], dim=0, sorted=True, return_inverse=True)
    
    return new_nodes,sampled_edges
