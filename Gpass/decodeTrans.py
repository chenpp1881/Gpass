import torch
import torch.nn as nn
from Multihead_Attention import DecoderAttention
from SubLayerConnection import SublayerConnection


class decodeTransformerBlock(nn.Module):

    def __init__(self, hidden, num_layers, dropout, opts):
        super().__init__()
        self.opts = opts
        self.num_layers = num_layers
        self.self_attention = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        self.sublayer_connection1 = nn.ModuleList()
        self.sublayer_connection2 = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.information_integation = nn.ModuleList()
        self.II_connection = nn.ModuleList()
        for _layer in range(self.num_layers):
            self.self_attention.append(DecoderAttention(d_model=hidden))
            self.decoder_attention.append(DecoderAttention(d_model=hidden))
            self.sublayer_connection1.append(SublayerConnection(size=hidden, dropout=dropout))
            self.sublayer_connection2.append(SublayerConnection(size=hidden, dropout=dropout))
            self.linear_layers.append(nn.Linear(in_features=opts.hidden_dim, out_features=opts.hidden_dim))
            self.information_integation.append(DecoderAttention(d_model=hidden))
            self.II_connection.append(SublayerConnection(size=hidden, dropout=dropout))
        self.linear1 = nn.Linear(in_features=opts.hidden_dim + 3, out_features=opts.hidden_dim)
        self.linear2 = nn.Linear(in_features=opts.hidden_dim + 3, out_features=opts.hidden_dim)
        self.linear3 = nn.Linear(in_features=opts.hidden_dim + 3, out_features=opts.hidden_dim)


    def forward(self, action_emb, goal_emb, local_context_emb, environment_emb, term_t, tactic_t, mask):
        goal_emb = self.linear1(goal_emb)
        local_context_emb = self.linear2(local_context_emb)
        environment_emb = self.linear3(environment_emb)
        input_tr = torch.stack([goal_emb, local_context_emb, environment_emb, term_t, tactic_t], dim=1)
        for layer_idx in range(self.num_layers):
            input_tr = self.II_connection[layer_idx](input_tr, lambda _input_tr: self.information_integation[layer_idx].forward(_input_tr, _input_tr, _input_tr))
        for layer_idx in range(self.num_layers):
            action_emb = self.sublayer_connection1[layer_idx](action_emb, lambda _action_emb: self.self_attention[
                layer_idx].forward(_action_emb, _action_emb, _action_emb, mask))
            input_tr = self.sublayer_connection2[layer_idx](input_tr,
                                                            lambda _input_tr: self.decoder_attention[layer_idx].forward(
                                                                _input_tr, action_emb, action_emb))
            input_tr = self.linear_layers[layer_idx](input_tr)
        return input_tr