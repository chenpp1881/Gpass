import torch
import torch.nn as nn
import os
from itertools import chain
import sys
import math

sys.path.append(os.path.abspath('..'))
from tac_grammar import CFG
from .tactic_decoder import TacticDecoder
from .term_encoder import TermEncoder
from torch.nn import Embedding
import pickle
from NLdata_process import Seq_data_process, NlEncoder
from .AFPNet import AFPNet

class Prover(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.tactic_decoder = TacticDecoder(CFG(opts.tac_grammar, 'tactic_expr'), opts)
        self.term_encoder = TermEncoder(opts)
        self.tactic_embedding = Embedding(opts.num_tactics, opts.tac_embedding, padding_idx=0)
        self.tac_vocab = pickle.load(open(opts.tac_vocab_file, 'rb'))
        self.cutoff_len = opts.cutoff_len
        self.gal_embedding = Embedding(opts.num_gal, opts.gal_embedding, padding_idx=0)
        self.gal_vocab = pickle.load(open(opts.gal_vocab_file, 'rb'))
        self.gal_cutoff_len = opts.gal_cutoff_len
        self.Seq_data_process = Seq_data_process(Nl_Len=1024, Char_Len=15, PAD_token=0)
        # self.Tok_encoder = NlEncoder(char_len=15, embedding_size=256, Sqe_vocsize=self.opts.num_tactics)
        # self.Gal_encoder = NlEncoder(char_len=15, embedding_size=256, Sqe_vocsize=self.opts.num_gal)
        self.tac_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 30))
        self.gal_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 30))
        self.gal_GRU = nn.GRU(opts.gal_embedding, opts.gal_embedding, 1, batch_first=True)
        self.tactic_GRU = nn.GRU(opts.tac_embedding, opts.tac_embedding, 1, batch_first=True)
        self.Tok_encoder = AFPNet()
        self.Gal_encoder = AFPNet()

    def create_tactic_batch(self, tok_seq):
        mod_tok_seq = []
        if '<unk>' in self.tac_vocab:
            for item in tok_seq:
                mod_item = [self.tac_vocab[i] if i in self.tac_vocab else self.tac_vocab['<unk>'] for i in item]
                mod_tok_seq.append(mod_item)
        else:
            for item in tok_seq:
                mod_item = [self.tac_vocab[i] for i in item if i in self.tac_vocab]
                mod_tok_seq.append(mod_item)

        max_len = min(max([len(item) for item in mod_tok_seq]), self.cutoff_len)
        batch = []
        lens = []
        for item in mod_tok_seq:
            idx = self.cutoff_len - 1  # ex: 29, for len 30
            lens.append(len(item[-idx:]) + 1)
            new_item = [self.tac_vocab['<start>']] + item[-idx:] + [self.tac_vocab['<pad>']] * (
                        max_len - len(item[-idx:]) - 1)
            batch.append(new_item)

        return torch.tensor(batch, device=self.opts.device), lens

    def create_gal_batch(self, gal_seq):
        mod_gal_seq = []
        if '<unk>' in self.gal_vocab:
            for item in gal_seq:
                mod_item = [self.gal_vocab[i] if i in self.gal_vocab else self.gal_vocab['<unk>'] for i in item]
                mod_gal_seq.append(mod_item)
        else:
            for item in gal_seq:
                mod_item = [self.gal_vocab[i] for i in item if i in self.gal_vocab]
                mod_gal_seq.append(mod_item)

        max_len = min(max([len(item) for item in mod_gal_seq]), self.gal_cutoff_len)
        batch = []
        lens = []
        for item in mod_gal_seq:
            idx = self.gal_cutoff_len - 1  # ex: 29, for len 30
            lens.append(len(item[-idx:]) + 1)
            new_item = [self.gal_vocab['<start>']] + item[-idx:] + [self.gal_vocab['<pad>']] * (
                        max_len - len(item[-idx:]) - 1)
            batch.append(new_item)

        return torch.tensor(batch, device=self.opts.device), lens

    def embed_terms(self, environment, local_context, goal, tok_seq=None, gal_seq=None):
        all_asts = list(
            chain([env['ast'] for env in chain(*environment)], [context['ast'] for context in chain(*local_context)],
                  goal))
        all_embeddings = self.term_encoder(all_asts)
        # 每一个ast编码为一个256维向量
        batchsize = len(environment)
        environment_embeddings = []  # 保存ast的编码list[batch_size*tensor[num_ast,256+3]]
        j = 0

        for n in range(batchsize):
            size = len(environment[n])
            environment_embeddings.append(torch.cat([torch.zeros(size, 3, device=self.opts.device),
                                                     all_embeddings[j: j + size]], dim=1))
            environment_embeddings[-1][:, 0] = 1.0
            j += size

        context_embeddings = []
        for n in range(batchsize):
            size = len(local_context[n])
            context_embeddings.append(torch.cat([torch.zeros(size, 3, device=self.opts.device),
                                                 all_embeddings[j: j + size]], dim=1))
            context_embeddings[-1][:, 1] = 1.0
            j += size

        goal_embeddings = []
        for n in range(batchsize):
            goal_embeddings.append(torch.cat([torch.zeros(3, device=self.opts.device), all_embeddings[j]], dim=0))
            goal_embeddings[-1][2] = 1.0
            j += 1
        goal_embeddings = torch.stack(goal_embeddings)

        if tok_seq:
            inputnl, inputnlchar = self.Seq_data_process.preProcessData(tok_seq, voc='tok')
            inputnl = torch.tensor(inputnl, dtype=torch.long, device=self.opts.device)
            # inputnlchar = torch.tensor(inputnlchar, dtype=torch.long, device=self.opts.device)
            # inputnlmask = torch.gt(inputnl, 0)
            # tactic_seq_embeddings = self.Tok_encoder(inputnl,
            #                                          inputnlchar,
            #                                          inputnlmask)
            tactic_seq_embeddings = self.Tok_encoder(inputnl)

        if gal_seq:
            inputnl, inputnlchar = self.Seq_data_process.preProcessData(gal_seq, voc='gal')
            inputnl = torch.tensor(inputnl, dtype=torch.long, device=self.opts.device)
            # inputnlchar = torch.tensor(inputnlchar, dtype=torch.long, device=self.opts.device)
            # inputnlmask = torch.gt(inputnl, 0)
            # gal_seq_embeddings = self.Gal_encoder(inputnl,
            #                                       inputnlchar,
            #                                       inputnlmask)
            gal_seq_embeddings = self.Gal_encoder(inputnl)

            return environment_embeddings, context_embeddings, goal_embeddings, tactic_seq_embeddings, gal_seq_embeddings

        return environment_embeddings, context_embeddings, goal_embeddings

    def forward(self, environment, local_context, goal, actions, teacher_forcing, tok_seq=None, gal_seq=None,
                inference=False):
        environment_embeddings, context_embeddings, goal_embeddings, seq_embeddings, gal_embeddings = \
            self.embed_terms(environment, local_context, goal, tok_seq, gal_seq)

        environment = [{'idents': [v['qualid'] for v in env],
                        'embeddings': environment_embeddings[i],
                        'quantified_idents': [v['ast'].quantified_idents for v in env]}
                       for i, env in enumerate(environment)]
        local_context = [{'idents': [v['ident'] for v in context],
                          'embeddings': context_embeddings[i],
                          'quantified_idents': [v['ast'].quantified_idents for v in context]}
                         for i, context in enumerate(local_context)]
        goal = {'embeddings': goal_embeddings, 'quantified_idents': [g.quantified_idents for g in goal]}
        if inference:
            asts = self.tactic_decoder.inference(environment, local_context, goal, actions, teacher_forcing,
                                                 seq_embeddings, gal_embeddings)
            return asts
        else:
            asts, loss = self.tactic_decoder(environment, local_context, goal, actions, teacher_forcing, seq_embeddings,
                                             gal_embeddings)
        return asts, loss

    def beam_search(self, environment, local_context, goal, tok_seq=None, gal_seq=None):
        environment_embeddings, context_embeddings, goal_embeddings, seq_embeddings, gal_embeddings = \
            self.embed_terms([environment], [local_context], [goal], [tok_seq], [gal_seq])
        environment = {'idents': [v['qualid'] for v in environment],
                       'embeddings': environment_embeddings[0],
                       'quantified_idents': [v['ast'].quantified_idents for v in environment]}
        local_context = {'idents': [v['ident'] for v in local_context],
                         'embeddings': context_embeddings[0],
                         'quantified_idents': [v['ast'].quantified_idents for v in local_context]}
        goal = {'embeddings': goal_embeddings, 'quantified_idents': goal.quantified_idents}
        asts = self.tactic_decoder.beam_search(environment, local_context, goal, seq_embeddings, gal_embeddings)
        return asts
