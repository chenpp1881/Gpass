import torch.nn as nn
import torch.nn.functional as F
import torch


class FPM(nn.Module):
    def __init__(self, pad_idx=0):
        super(FPM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=50265, embedding_dim=256, padding_idx=pad_idx
        )
        self.num_channel = 64
        self.filter_sizes = [3, 5, 7, 11]
        self.hidden_dim = 256
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=self.num_channel,
                                              kernel_size=(filter_size, self.hidden_dim))
                                    for filter_size in self.filter_sizes])

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        conved = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        top_p_max = [torch.topk(conv, 1, 2, True, False).values for conv in conved]
        x_max = torch.cat(top_p_max, -1)
        x_max = x_max.view(-1, len(self.filter_sizes), self.num_channel)
        return x_max


class AFPNet(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.num_channel = 64
        self.filter_sizes = 4
        self.attention = nn.ModuleList()
        self.sublayer_connection = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.fpm = FPM()
        trans_encoder = nn.TransformerEncoderLayer(d_model=self.num_channel, nhead=8)
        self.attention = nn.TransformerEncoder(encoder_layer=trans_encoder, num_layers=6)

    def forward(self, x):
        x = self.fpm(x)
        x_res = x
        x = self.attention(x)
        x = (x_res + x) / 2
        assert self.num_channel * self.filter_sizes == 256
        return x.reshape(-1, self.num_channel * self.filter_sizes)
