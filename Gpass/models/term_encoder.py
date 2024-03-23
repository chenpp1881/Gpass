import torch
import torch.nn as nn
from .UGformerV2 import GraphTransformerNet

# from torch_geometric.nn import TransformerConv
nonterminals = [
    'constr__constr',
    'constructor_rel',
    'constructor_var',
    'constructor_meta',
    'constructor_evar',
    'constructor_sort',
    'constructor_cast',
    'constructor_prod',
    'constructor_lambda',
    'constructor_letin',
    'constructor_app',
    'constructor_const',
    'constructor_ind',
    'constructor_construct',
    'constructor_case',
    'constructor_fix',
    'constructor_cofix',
    'constructor_proj',
    'constructor_ser_evar',
    'constructor_prop',
    'constructor_set',
    'constructor_type',
    'constructor_ulevel',
    'constructor_vmcast',
    'constructor_nativecast',
    'constructor_defaultcast',
    'constructor_revertcast',
    'constructor_anonymous',
    'constructor_name',
    'constructor_constant',
    'constructor_mpfile',
    'constructor_mpbound',
    'constructor_mpdot',
    'constructor_dirpath',
    'constructor_mbid',
    'constructor_instance',
    'constructor_mutind',
    'constructor_letstyle',
    'constructor_ifstyle',
    'constructor_letpatternstyle',
    'constructor_matchstyle',
    'constructor_regularstyle',
    'constructor_projection',
    'bool',
    'int',
    'names__label__t',
    'constr__case_printing',
    'univ__universe__t',
    'constr__pexistential___constr__constr',
    'names__inductive',
    'constr__case_info',
    'names__constructor',
    'constr__prec_declaration___constr__constr____constr__constr',
    'constr__pfixpoint___constr__constr____constr__constr',
    'constr__pcofixpoint___constr__constr____constr__constr',
]


class TermEncoder(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.embedding_constructor = nn.Embedding(num_embeddings=len(nonterminals) + 1,
                                                  embedding_dim=self.opts.fpath_feature_dim)
        self.GTN = GraphTransformerNet(feature_dim_size=self.opts.ast_feature_dim,
                                       ff_hidden_size=self.opts.ast_feature_dim,
                                       num_self_att_layers=1, num_GNN_layers=3, nhead=32)

    def forward(self, term_asts):

        node_feature = []
        node_feature_dict = {}
        begin_node = []
        end_node = []
        node_index_list = []

        def Get_fpath(node, pre_node_index):

            node_feature.append(self.embedding_constructor(
                torch.tensor(nonterminals.index(node.data), dtype=torch.long, device=self.opts.device)))

            if pre_node_index == -1:
                current_node_index = 0
            else:
                current_node_index = max(node_index_list) + 1
                begin_node.append(pre_node_index)
                end_node.append(current_node_index)

            node_index_list.append(current_node_index)

            if node.children == []:
                return

            for c in node.children:
                Get_fpath(node=c, pre_node_index=current_node_index)

        for ast in term_asts:
            Get_fpath(node=ast, pre_node_index=-1)
            node_feature = torch.stack(node_feature)
            edge_index = torch.tensor([begin_node + end_node, end_node + begin_node], dtype=torch.long,
                                      device=self.opts.device)
            node_feature = self.GTN(edge_index=edge_index, node_features=node_feature)
            node_feature_dict[ast] = node_feature

            node_feature = []
            begin_node = []
            end_node = []
            node_index_list = []

        return torch.stack([node_feature_dict[ast] for ast in term_asts])
