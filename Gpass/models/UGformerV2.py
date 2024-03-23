import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GraphTransformerNet(nn.Module):

    def __init__(self, feature_dim_size, ff_hidden_size,
                 num_self_att_layers, num_GNN_layers, nhead):
        super(GraphTransformerNet, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers
        self.num_GNN_layers = num_GNN_layers
        self.nhead = nhead
        self.lst_gnn = torch.nn.ModuleList()
        self.ugformer_layers = torch.nn.ModuleList()
        # for _layer in range(self.num_GNN_layers):
        #     encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=self.nhead,
        #                                              dim_feedforward=self.ff_hidden_size,
        #                                              dropout=0.3)
        #     self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
        #     self.lst_gnn.append(GCNConv(self.feature_dim_size, self.ff_hidden_size))

        for _layer in range(self.num_GNN_layers):
            self.lst_gnn.append(GCNConv(self.feature_dim_size, self.ff_hidden_size))

    def forward(self, edge_index, node_features):
        input_Tr = node_features
        # for layer_idx in range(self.num_GNN_layers):
        #     input_Tr = torch.unsqueeze(input_Tr, 0)
        #     input_Tr = self.ugformer_layers[layer_idx](input_Tr)
        #     input_Tr = torch.squeeze(input_Tr, 0)
        #     input_Tr = self.lst_gnn[layer_idx](x=input_Tr, edge_index=edge_index)

        for layer_idx in range(self.num_GNN_layers):
            input_Tr = self.lst_gnn[layer_idx](x=input_Tr, edge_index=edge_index)


        return torch.sum(input_Tr, dim=0) / math.sqrt(input_Tr.shape[0])
