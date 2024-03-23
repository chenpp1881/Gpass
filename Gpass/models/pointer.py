import torch.nn as nn
import torch.nn.functional as F

class CopyNet(nn.Module):
    def __init__(self, opts):
        super(CopyNet, self).__init__()
        self.embedding_size = opts.ast_feature_dim
        self.LinearSource = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearTarget = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        self.LinearProb = nn.Linear(self.embedding_size, 2)
    def forward(self, source, traget):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(F.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(-1)
        prob = F.relu(self.LinearProb(traget))
        return genP, prob
