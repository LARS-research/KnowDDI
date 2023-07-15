import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.sparse as ssp
from utils.data_utils import process_files_ddi, process_files_decagon
from utils.graph_utils import incidence_matrix, remove_nodes, serialize, _bfs_relational

def generate_subgraph_datasets(params):
    splits=['train', 'valid', 'test']
        
    BKG_file = '../data/{}/{}.txt'.format(params.dataset,params.BKG_file_name)
    print(BKG_file)
    if params.dataset == 'drugbank':
        adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(params.file_paths, BKG_file)
    elif params.dataset == 'BioSNAP':
        adj_list,triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = process_files_decagon(params.file_paths, BKG_file)

    files_data = {}
    for split_name in splits:
        if params.dataset == 'drugbank':
            files_data[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}
        elif params.dataset == 'BioSNAP':
            files_data[split_name] = {'triplets': triplets_mr[split_name], 'max_size': params.max_links, "polarity_mr": polarity_mr[split_name]}

    for split_name, split in files_data.items():
        max_size = split['max_size']
        if max_size < len(split['triplets']):
            perm = np.random.permutation(len(split['triplets']))[:max_size]
            split['triplets'] = split['triplets'][perm]
    
    links2subgraphs(adj_list, files_data, params)

def links2subgraphs(adj_list, files_data, params, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(files_data.values())[0]['triplets'], adj_list, params) * 1.5
    links_length = 0
    for split_name, split in files_data.items():
        links_length += len(split['triplets']) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(adj_list, links, g_labels, split_env):
        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))
        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(adj_list, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in files_data.items():
        logging.info(f"Extracting enclosing subgraphs for links in {split_name} set")
        if params.dataset == 'BioSNAP':
            g_labels = np.array(split["polarity_mr"])
        else:
            g_labels = np.ones(len(split['triplets']))
        db_name_pos = split_name + '_subgraph'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(adj_list, split['triplets'], g_labels, split_env)


    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))



def get_average_subgraph_size(sample_size, links, adj_list, params):
    total_size = 0
    lst = np.random.choice(len(links), sample_size)
    for idx in lst:
        (n1, n2, r_label) = links[idx]
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, adj_list, params.hop, params.enclosing_subgraph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(adj_list, params, max_label_value):
    global adj_list_, params_, max_label_value_
    adj_list_, params_, max_label_value_ = adj_list, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    pruned_subgraph_nodes, pruned_node_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, adj_list_, params_.hop, params_.enclosing_subgraph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        pruned_node_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in pruned_node_labels])

    datum = {'nodes': pruned_subgraph_nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': pruned_node_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

def get_neighbor_nodes(roots, adj, hop=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(hop):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def subgraph_extraction_labeling(ind, rel, A_list, hop=1, enclosing_subgraph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T
    ind = list(ind)
    ind[0], ind[1] = int(ind[0]), int(ind[1])
    ind = (ind[0], ind[1])
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, hop, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, hop, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_subgraph:
        if ind[0] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        if ind[0] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
    
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=hop)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes



