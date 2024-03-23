import os
import pickle
from Embedding import Embedding
import torch.nn as nn
from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from ConvolutionForward import ConvolutionLayer
from Multihead_Combination import MultiHeadedCombination

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.combination = MultiHeadedCombination(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.conv_forward = ConvolutionLayer(dmodel=hidden, layernum=hidden)
        self.s1 = SublayerConnection(size=hidden, dropout=dropout)
        self.s2 = SublayerConnection(size=hidden, dropout=dropout)
        self.s3 = SublayerConnection(size=hidden, dropout=dropout)
        self.s4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, charEm, treemask=None, isTree=False):
        x = self.s1(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.s2(x, lambda _x: self.combination.forward(_x, _x, charEm))
        if isTree:
            x = self.s3(x, lambda _x: self.attention.forward(_x, _x, _x, mask=treemask))
            x = self.s4(x, self.feed_forward)
        else:
            x = self.s3(x, lambda _x:self.conv_forward.forward(_x, mask))
        return self.dropout(x)

class Seq_data_process():
    def __init__(self, Nl_Len, Char_Len, PAD_token):
        self.PAD_token = PAD_token
        self.Nl_Len = Nl_Len
        self.Char_Len = Char_Len
        self.Load_Voc()

    def Load_Voc(self):
        # 加载词库
        if os.path.exists("token_vocab.pickle"):
            self.Token_Voc = pickle.load(open("token_vocab.pickle", "rb"))
        if os.path.exists("gal_vocab.pickle"):
            self.Gal_Voc = pickle.load(open("gal_vocab.pickle", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
        self.Token_Voc["<emptynode>"] = len(self.Token_Voc)
        self.Gal_Voc["<emptynode>"] = len(self.Gal_Voc)

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
        return seq

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans

    def pad_list(self, seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def preProcessData(self, nl_list, voc):
        assert voc in ('tok','gal')
        if voc == 'tok':
            voc = self.Token_Voc
        if voc == 'gal':
            voc = self.Gal_Voc
        inputNl = []
        inputNlChar = []
        for nl in nl_list:
            inputnls = self.Get_Em(nl, voc)
            inputNl.append(self.pad_seq(inputnls, self.Nl_Len))

            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
        return inputNl, inputNlChar

class NlEncoder(nn.Module):
    def __init__(self,char_len,embedding_size,Seq_vocsize):
        super(NlEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.word_len = char_len
        self.char_embedding = nn.Embedding(58, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.token_embedding = Embedding(Seq_vocsize, self.embedding_size)

    def forward(self, input_nl, input_nlchar, mask=None):
        charEm = self.char_embedding(input_nlchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_nl.long())
        for trans in self.transformerBlocks:
            x = trans.forward(x, mask, charEm)
        return x


if __name__ == '__main__':
    Seq_data_process = Seq_data_process(Nl_Len=50,Char_Len=15,PAD_token=0)
    test_list = [['unfold', 'd', 'in', 'elim', 'Der', 'apply', 'not_dans_X_eps', 'assumption', 'unfold', 'f_R_d', 'in', 'assumption',
     'apply', 'dans_map', 'assumption', 'simpl', 'in', 'apply', 'Derive_X', 'assumption', 'trivial', 'trivial', 'apply',
     'word_word_inv', 'auto', 'apply', 'word_word_inv', 'auto', 'auto', 'apply', 'dans_map', 'assumption', 'auto',
     'unfold', 'd', 'in', 'apply', 'Derive_V', 'rewrite', 'eg_r', 'rewrite', 'eg_r', 'assumption', 'trivial', 'trivial',
     'apply', 'Regles_X_V_R'],['intros', 'apply', 'lcos_lg_anga', 'in', 'H', 'apply', 'lcos_lg_anga', 'in', 'H0', 'apply',
    'lcos_lg_anga', 'in', 'H1', 'apply', 'lcos_lg_anga', 'in', 'H2', 'intro', 'apply', 'H3', 'auto', 'apply', 'l10_4', 'auto',
    'intro', 'apply', 'H17', 'apply', 'l10_14', 'intro', 'subst', 'P', 'apply', 'H19']]
    a,b = Seq_data_process.preProcessData(test_list,voc='tok')

