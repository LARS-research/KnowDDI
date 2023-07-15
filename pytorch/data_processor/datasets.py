from torch.utils.data import Dataset
import os
import lmdb
import numpy as np
import json
import dgl
import torch
from utils.data_utils import process_files_ddi,process_files_decagon
from utils.graph_utils import deserialize,get_neighbors,ssp_multigraph_to_dgl
import scipy.sparse as ssp

def get_kge_embeddings(dataset, kge_model):
    """
    Use pre embeddings from pretrained models
    """
    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name, raw_data_paths= None, add_traspose_rels=None,  use_pre_embeddings=False, dataset='', kge_model='',ssp_graph = None, id2entity= None, id2relation= None, rel= None,  global_graph = None, dig_layer=4,BKG_file_name=''):
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db = self.main_env.open_db(db_name.encode())
        self.db_path = db_path
        self.db_name = db_name
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_pre_embeddings else (None, None)
        BKG_file = '../data/{}/{}.txt'.format(dataset,BKG_file_name)

        if not ssp_graph:
            if dataset == 'drugbank' or dataset == 'drugbank_sub':
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(raw_data_paths, BKG_file)
            elif dataset == 'BioSNAP':
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = process_files_decagon(raw_data_paths, BKG_file)

            self.num_rels = rel
            print('number of relations:%d'%(self.num_rels))

            # Add transpose matrices to handle both directions of relations.
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

            #add self loops
            ssp_graph.append(ssp.identity(len(id2entity)))

            # the effective number of relations after adding symmetric adjacency matrices and/or self connections
            self.aug_num_rels = len(ssp_graph)
            self.global_graph = ssp_multigraph_to_dgl(ssp_graph)
            self.ssp_graph = ssp_graph
        else:
            self.num_rels = rel
            self.aug_num_rels = len(ssp_graph)
            self.global_graph = global_graph
            self.ssp_graph = ssp_graph

        self.id2entity = id2entity
        self.id2relation = id2relation
        self.num_entity = len(id2entity)
        self.dig_layer = dig_layer
        with self.main_env.begin(db=self.db) as txn:
            self.num_graphs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')


    def __getitem__(self, index):
        with self.main_env.begin(db=self.db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes, r_label, g_label, n_labels = deserialize(txn.get(str_id)).values()
            directed_subgraph = self._prepare_subgraphs(nodes, n_labels)
            return directed_subgraph, r_label, g_label
           
    def __len__(self):
        return self.num_graphs

    def _prepare_subgraphs(self, nodes, n_labels):
        subgraph = self.global_graph.subgraph(nodes)
        subgraph.edata['type'] = self.global_graph.edata['type'][self.global_graph.subgraph(nodes).edata[dgl.EID]]
        subgraph.ndata['idx'] = torch.LongTensor(np.array(nodes))
        subgraph = self._prepare_features(subgraph, n_labels)
        _,_,edges_btw_roots = subgraph.edge_ids(0, 1,return_uv=True)
        subgraph.remove_edges(edges_btw_roots)
        
        directed_subgraph = self.extract_r_digraph(subgraph)
        return directed_subgraph
    
    def extract_r_digraph(self,graph):
        """
        Extract subgraphs using the algorithm proposed in the paper
        """
        head_nodes = (graph.ndata['id'] == 1).nonzero().squeeze(1)
        tail_nodes = (graph.ndata['id'] == 2).nonzero().squeeze(1)

        total_nodes = torch.cat([head_nodes,tail_nodes])
        raw_layer_edges = {}
        for i in range(self.dig_layer):
            head_nodes, head_edges =  get_neighbors(graph,head_nodes)
            raw_layer_edges[i]=head_edges
            
        layer_edges_id = torch.LongTensor([])

        for i in reversed(range(self.dig_layer)):
            select = torch.nonzero(torch.eq(raw_layer_edges[i][:,1],tail_nodes.unsqueeze(1)))
            l_edge = raw_layer_edges[i][select[:,1]]

            layer_edges_id=torch.cat([layer_edges_id,l_edge[:,2]]) 
            tail_nodes=torch.unique(l_edge[:,0])

        total_edges = torch.unique(layer_edges_id, dim=0, sorted=True)
        if total_edges.numel():
            r_digraph = dgl.edge_subgraph(graph,total_edges)
        else:
            # If the extracted subgraph has no edges, then the returned graph only has head and tail
            r_digraph = dgl.node_subgraph(graph,total_nodes)

        return r_digraph

    def _prepare_features(self, subgraph, n_labels):
        n_nodes = subgraph.number_of_nodes()
        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph#, h__, t__

