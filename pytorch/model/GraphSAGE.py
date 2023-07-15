import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self,params):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(params.gcn_dropout)
        self.activation = F.relu
        self.aggregator_type = params.gcn_aggregator_type
        self.num_nodes = params.num_nodes
        self.num_rels = params.aug_num_rels
        self.emb_dim = params.emb_dim
        self.num_gcn_layers = params.num_gcn_layers

        self.pre_embed = nn.Parameter(torch.Tensor(self.num_nodes, self.emb_dim), requires_grad = True)
        nn.init.xavier_uniform_(self.pre_embed, gain=nn.init.calculate_gain('relu'))

        self.rel_weght = nn.Parameter(torch.Tensor(self.num_rels, self.emb_dim), requires_grad = True)
        nn.init.xavier_uniform_(self.rel_weght, gain=nn.init.calculate_gain('relu'))

        for i in range(self.num_gcn_layers):
            self.layers.append(SAGEConv(self.emb_dim, self.emb_dim, self.aggregator_type))

    def forward(self, g):
        h = self.pre_embed[g.ndata['idx']]
        h = self.dropout(h)
        edge_weight = self.rel_weght[g.edata['type']]
        for l, layer in enumerate(self.layers):
            g.ndata['h'] = layer(g, h, edge_weight = edge_weight)
            if l != len(self.layers) - 1:
                g.ndata['h'] = self.activation(g.ndata['h'])
                g.ndata['h'] = self.dropout(g.ndata['h'])
            h = g.ndata['h']

            if l == 0:
                x = torch.cat([self.pre_embed[g.ndata['idx']], g.ndata['h']], dim = 1)
                g.ndata['repr'] = x.unsqueeze(1).reshape(-1, 2, self.emb_dim)
            else:
                g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)

        return g.ndata.pop('h')