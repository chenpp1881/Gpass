import torch
import numpy as np
import random
from glob import glob
import argparse
import json
import os
import sys
sys.setrecursionlimit(100000)
sys.path.append(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')))
from hashlib import md5
from utils import log
from progressbar import ProgressBar
from agent import Agent
from models.prover import Prover
import pdb
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str)
    parser.add_argument('eval_id', type=str)
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--projs_split', type=str, default='../projs_split.json')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], type=str, default='test')
    parser.add_argument('--file', type=str)
    parser.add_argument('--proof', type=str)
    parser.add_argument('--file_idx', type=int)
    parser.add_argument('--proj_idx', type=int)
    parser.add_argument('--filter', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_dir', type=str, default='evaluation')
    parser.add_argument('--max_num_tactics', type=int, default=300)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--hammer_timeout', type=int, default=100)
    parser.add_argument('--depth_limit', type=int, default=50)
    parser.add_argument('--beam_width', type=int, default=20)  # lots of timeout when >200
    parser.add_argument('--num_tactic_candidates', type=int, default=20)
    parser.add_argument('--lens_norm', type=float, default=0.5, help='lengths normalization')
    parser.add_argument('--tac_grammar', type=str, default='tactics.ebnf')
    parser.add_argument('--term_embedding_dim', type=int, default=256)
    parser.add_argument('--ast_feature_dim', type=int, default=256)
    parser.add_argument('--fpath_feature_dim', type=int, default=256)
    parser.add_argument('--max_family_path', type=int, default=10)
    parser.add_argument('--size_limit', type=int, default=50)
    parser.add_argument('--embedding_dim', type=int, default=256, help='dimension of the grammar embeddings')
    parser.add_argument('--symbol_dim', type=int, default=256, help='dimension of the terminal/nonterminal symbol embeddings')
    parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of the LSTM controller')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_tactics', type=int, default=15025)
    parser.add_argument('--tac_vocab_file', type=str, default='token_vocab.pickle')
    parser.add_argument('--cutoff_len', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--tac_embedding', type=int, default=256)
    parser.add_argument('--tac_layers', type=int, default=1)
    parser.add_argument('--num_gal', type=int, default=29467)
    parser.add_argument('--gal_vocab_file', type=str, default='gal_vocab.pickle')
    parser.add_argument('--gal_cutoff_len', type=int, default=30)
    parser.add_argument('--gal_embedding', type=int, default=256)
    parser.add_argument('--gal_layers', type=int, default=1)
    opts = parser.parse_args()
    log(opts)
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opts.device.type == 'cpu':
        log('using CPU', 'WARNING')

    torch.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    projs_test = ["weak-up-to", "buchberger", "jordan-curve-theorem", "dblib", "disel", "zchinese", "zfc", "dep-map", "chinese", "UnifySL", "hoare-tut", "huffman", "PolTac", "angles", "coq-procrastination", "coq-library-undecidability", "tree-automata", "coquelicot", "fermat4", "demos", "coqoban", "goedel", "verdi-raft", "verdi", "zorns-lemma", "coqrel", "fundamental-arithmetics"]

    if 'ours' in opts.method:
        t1 = time.time()
        model = Prover(opts)
        log('loading model checkpoint from %s..' % opts.path)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.path, map_location='cpu')
        else:
            checkpoint = torch.load(opts.path)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(opts.device)
        t2 = time.time()
        print("import model takes", (t2-t1) * 1e3, 'ms')
    else:
        model = None

    agent = Agent(model, None, None, opts)

    if opts.file:
        files = [opts.file]
    elif opts.proj_idx is not None:
        files = glob(os.path.join(opts.datapath, '%s/**/*.json' % projs_test[opts.proj_idx]), recursive=True)
    else:
        files = []
        projs = json.load(open(opts.projs_split))['projs_' + opts.split]
        for proj in projs:
            files.extend(glob(os.path.join(opts.datapath, '%s/**/*.json' % proj), recursive=True))

    if opts.file_idx is not None:
        files = [files[opts.file_idx]]

    if opts.filter:
        files = [f for f in files if md5(f.encode('utf-8')).hexdigest().startswith(opts.filter)]

    print(files)
    results = []
    for i, f in enumerate(files):
        print('file: ', f)
        results.extend(agent.evaluate(f, opts.proof))

    oup_dir = os.path.join(opts.output_dir, opts.eval_id)
    if not os.path.exists(oup_dir):
         os.makedirs(oup_dir)
    if opts.filter is None and opts.file is None and opts.file_idx is None:
        oup_file = os.path.join(oup_dir, 'results_%s.json' % str(opts.proj_idx))
    elif opts.file_idx is not None:
        oup_file = os.path.join(oup_dir, 'results_%s_%s.json' % (str(opts.proj_idx), str(opts.file_idx)))
    elif opts.file is None:
        oup_file = os.path.join(oup_dir, '%s.json' % opts.filter)
    elif opts.proof is None:
        oup_file = os.path.join(oup_dir, '%s.json' % os.path.sep.join(opts.file.split(os.path.sep)[2:]).replace(os.path.sep, '-'))
    else:
        oup_file = os.path.join(oup_dir, '%s-%s.json' % (os.path.sep.join(opts.file.split(os.path.sep)[2:]).replace(os.path.sep, '-'), opts.proof))
    opts = vars(opts)
    del opts['device']
    json.dump({'options': opts, 'results': results}, open(oup_file, 'wt'))
    log('results saved to ' + oup_file)