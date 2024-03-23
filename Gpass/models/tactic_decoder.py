import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import random
from copy import deepcopy
from tac_grammar import TerminalNode, NonterminalNode
from lark.lexer import Token
from decodeTrans import decodeTransformerBlock
from .pointer import CopyNet


class AvgLoss:
    'Maintaining the average of a set of losses'

    def __init__(self, device):
        self.sum = torch.tensor(0., device=device)
        self.num = 0

    def add(self, v):
        self.sum += v
        self.num += 1

    def value(self):
        return self.sum / self.num


class Actions_Emb(nn.Module):
    def __init__(self, opts):
        super(Actions_Emb, self).__init__()
        self.opts = opts
        self.Load_Voc()
        self.char_embedding = nn.Embedding(58, opts.term_embedding_dim)
        self.action_embedding = nn.Embedding(99, opts.term_embedding_dim)

    def Load_Voc(self):
        self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))

    def Get_Emb(self, x):
        if isinstance(x, str):
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(self.char_embedding(torch.LongTensor([c_id]).to(self.opts.device)))
            char_emb = torch.cat(tmp, dim=0)
            return torch.sum(char_emb, dim=0) / char_emb.shape[0]
        else:
            return self.action_embedding(torch.LongTensor([x]).to(self.opts.device)).squeeze(0)

    def forward(self, actions):
        proof_step_emb = []
        proof_emb = []

        max_proof_len = 5
        for proof_step in actions:
            for i in range(max_proof_len - 1):
                if i < len(proof_step):
                    proof_step_emb.append(self.Get_Emb(proof_step[i]))
                else:
                    proof_step_emb.append(torch.zeros((self.opts.term_embedding_dim), device=self.opts.device))
            proof_emb.append(torch.stack(proof_step_emb, dim=0))
            proof_step_emb = []
        # begin of sentence
        bos = self.action_embedding(torch.LongTensor([98]).to(self.opts.device)).expand(len(proof_emb), 1,
                                                                                        self.opts.term_embedding_dim)
        return torch.cat([bos, torch.stack(proof_emb)], dim=1)


class TermReader(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.default_term = torch.zeros(self.opts.term_embedding_dim, device=self.opts.device)
        self.linear1 = nn.Linear(opts.hidden_dim + 3, opts.hidden_dim)
        self.linear2 = nn.Linear(opts.hidden_dim, opts.hidden_dim)

    def forward(self, states, embeddings):
        assert states.size(0) == embeddings.size(0)
        term = []
        for state, embedding in zip(states, embeddings):
            if embedding.size(0) == 0:  # no premise
                term.append(self.default_context)
            else:
                weights = torch.matmul(self.linear2(embedding), self.linear1(state))
                weights = F.softmax(weights, dim=0)
                term.append(torch.matmul(embedding.t(), weights).squeeze())
        term = torch.stack(term)
        return term


class ContextReader(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.default_context = torch.zeros(self.opts.term_embedding_dim + 3, device=self.opts.device)
        self.linear1 = nn.Linear(opts.hidden_dim + 3, opts.hidden_dim)
        self.linear2 = nn.Linear(opts.hidden_dim + 3, opts.hidden_dim)

    def forward(self, states, embeddings):
        assert states.size(0) == len(embeddings)
        context = []
        for state, embedding in zip(states, embeddings):
            if embedding.size(0) == 0:  # no premise
                context.append(self.default_context)
            else:
                weights = torch.matmul(self.linear2(embedding), self.linear1(state))
                weights = F.softmax(weights, dim=0)
                context.append(torch.matmul(embedding.t(), weights).squeeze())
        context = torch.stack(context)
        return context


def clear_state(node):
    del node.state


class Porinter_net(nn.Module):
    def __init__(self, opts):
        super(Porinter_net, self).__init__()
        self.embedding_size = opts.ast_feature_dim
        self.LinearSource = nn.Linear(self.embedding_size + 3, self.embedding_size, bias=False)
        self.LinearTarget = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

    def forward(self, target, source):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(target)
        return torch.matmul(sourceLinear, targetLinear)


class TacticDecoder(nn.Module):

    def __init__(self, grammar, opts):
        super().__init__()
        self.opts = opts
        self.grammar = grammar
        self.production_rule_embeddings = nn.Embedding(len(self.grammar.production_rules), opts.embedding_dim)
        self.lex_rule_embeddings = nn.Embedding(len(self.grammar.terminal_symbols), opts.embedding_dim)
        self.decodeTransformerBlock = decodeTransformerBlock(self.opts.hidden_dim, 4, 0.2, opts)
        self.state_decoder = nn.Sequential(nn.Linear(opts.hidden_dim, opts.embedding_dim), nn.Tanh())
        self.context_reader = ContextReader(opts)
        self.env_reader = ContextReader(opts)
        self.term_reader = TermReader(opts)
        self.tactic_reader = TermReader(opts)
        self.pointer_net = Porinter_net(opts)
        self.action_emb = Actions_Emb(opts)
        self.INT_classifier = nn.Linear(opts.hidden_dim, 4)
        self.hint_dbs = ['arith', 'zarith', 'algebra', 'real', 'sets', 'core', 'bool', 'datatypes', 'coc', 'set', 'zfc']
        self.HINT_DB_classifier = nn.Linear(opts.hidden_dim, len(self.hint_dbs))
        self.Pointer = CopyNet(opts)

    def gather_frontier_info(self, frontiers):
        indice = []  # indice for incomplete ASTs

        for i, stack in enumerate(frontiers):
            if stack == []:
                continue
            indice.append(i)

        if indice == []:
            return []
        return indice

    def initialize_trees(self, batchsize):
        asts = [NonterminalNode(self.grammar.start_symbol, parent=None) for i in range(batchsize)]  # partial results
        frontiers = [[asts[i]] for i in
                     range(batchsize)]  # the stacks for DFS, whose top are the next nodes [[ast],[ast],...]
        return asts, frontiers

    def expand_node_set_pred(self, node, rule, stack):
        node.expand(rule)

        # update the links to the predecessor
        for c in node.children[::-1]:
            if isinstance(c, Token):
                continue
            if stack != []:
                stack[-1].pred = c
            stack.append(c)

        if stack != []:
            stack[-1].pred = node

    def expand_nonterminal(self, node, expansion_step, nonterminal_expansion_step, actions_gt, teacher_forcing, stack,
                           inference):
        # selcet a production rule and compute the loss
        applicable_rules = self.grammar.get_applicable_rules(node.symbol)

        if teacher_forcing:
            logits = torch.matmul(self.production_rule_embeddings.weight[applicable_rules],
                                  self.state_decoder(node.state))
            action_idx = actions_gt[expansion_step]
            rule = self.grammar.production_rules[action_idx]  # expand the tree using the ground truth action
            action_gt_onehot = torch.LongTensor([applicable_rules.index(action_idx)]).to(self.opts.device)
            loss = F.cross_entropy(logits.unsqueeze(0), action_gt_onehot)

        else:
            logits = torch.matmul(self.production_rule_embeddings.weight, self.state_decoder(node.state))
            rule_idx = applicable_rules[logits[applicable_rules].argmax().item()]
            rule = self.grammar.production_rules[rule_idx]
            if nonterminal_expansion_step < len(actions_gt):
                action_idx = actions_gt[nonterminal_expansion_step]
                action_gt_onehot = torch.LongTensor([action_idx]).to(self.opts.device)
                loss = F.cross_entropy(logits.unsqueeze(0), action_gt_onehot)
            else:
                loss = 0.

            if expansion_step > self.opts.size_limit:  # end the generation process asap
                rule_idx = applicable_rules[0]
                rule = self.grammar.production_rules[rule_idx]

        self.expand_node_set_pred(node, rule, stack)

        if inference:
            return applicable_rules[logits.argmax()]
        else:
            return loss

    def expand_terminal(self, node, expansion_step, environment, local_context, goal, actions_gt, teacher_forcing,
                        inference):
        loss = 0.
        if teacher_forcing:
            token_gt = actions_gt[expansion_step]

        if node.symbol in ['QUALID', 'LOCAL_IDENT']:
            if node.symbol == 'QUALID':
                candidates = environment['idents'] + local_context['idents']
            else:
                candidates = local_context['idents']
            if candidates == []:
                token = random.choice(['H'] + goal['quantified_idents'])
            else:
                if node.symbol == 'QUALID':
                    candidate_embeddings = torch.cat([environment['embeddings'], local_context['embeddings']])
                else:
                    candidate_embeddings = local_context['embeddings']
                context_scores = self.pointer_net(node.state, candidate_embeddings)
                if teacher_forcing:
                    target = torch.zeros_like(context_scores)
                    if token_gt in candidates:
                        target[candidates.index(token_gt)] = 1.0
                    loss = F.binary_cross_entropy_with_logits(context_scores, target)
                else:
                    token = candidates[context_scores.argmax()]
            if inference:
                if candidates == []:
                    return random.choice(['H'] + goal['quantified_idents'])
                return candidates[context_scores.argmax()]

        elif node.symbol in 'INT':
            cls = self.INT_classifier(node.state)
            if teacher_forcing:
                cls_gt = torch.LongTensor([int(token_gt) - 1]).to(self.opts.device)
                loss = F.cross_entropy(cls.unsqueeze(0), cls_gt)
            else:
                token = str(cls.argmax().item() + 1)
            if inference:
                return ['1', '2', '3', '4'][cls.argmax()]

        elif node.symbol == 'HINT_DB':
            cls = self.HINT_DB_classifier(node.state)
            if teacher_forcing:
                cls_gt = torch.LongTensor([self.hint_dbs.index(token_gt)]).to(self.opts.device)
                loss = F.cross_entropy(cls.unsqueeze(0), cls_gt)
            else:
                token = self.hint_dbs[cls.argmax().item()]
            if inference:
                return self.hint_dbs[cls.argmax()]

        elif node.symbol == 'QUANTIFIED_IDENT':
            if goal['quantified_idents'] == []:
                candidates = ['x']
            else:
                candidates = goal['quantified_idents']
            token = random.choice(candidates)

        # generadddte a token with the lex rule
        node.expand(token_gt if teacher_forcing else token)

        if inference:
            return token
        else:
            return loss

    def expand_partial_tree(self, node, expansion_step, nonterminal_expansion_step, environment, local_context, goal,
                            actions_gt,
                            teacher_forcing, stack, inference=False):
        assert node.state is not None
        if isinstance(node, NonterminalNode):
            return self.expand_nonterminal(node, expansion_step, nonterminal_expansion_step, actions_gt,
                                           teacher_forcing, stack, inference)
        else:
            return self.expand_terminal(node, expansion_step, environment, local_context, goal, actions_gt,
                                        teacher_forcing, inference)

    def forward(self, environment, local_context, goal, actions, teacher_forcing, seq_embeddings=None,
                gal_seq_embeddings=None):
        if not teacher_forcing:
            # when train without teacher forcing, only consider the expansion of non-terminal nodes
            actions = [[a for a in act if isinstance(a, int)] for act in actions]

        loss = AvgLoss(self.opts.device)

        # initialize the trees
        batchsize = goal['embeddings'].size(0)
        # ast: [0,0,0,...] frontiers: [[ast],[ast],...]
        asts, frontiers = self.initialize_trees(batchsize)

        # expand the trees in a depth-first order
        expansion_step = 0
        nonterminal_expansion_step = [0 for i in range(batchsize)]

        action_emb = self.action_emb(actions)
        goal_emb = goal['embeddings']

        local_context_emb = self.context_reader(goal_emb, [torch.cat([i['embeddings']], dim=0) for i in local_context])
        environment_emb = self.env_reader(goal_emb, [torch.cat([i['embeddings']], dim=0) for i in environment])
        term_t = seq_embeddings
        tactic_t = gal_seq_embeddings

        action_mask = torch.tril(torch.ones((batchsize, action_emb.size(1), action_emb.size(1)))).to(
            self.opts.device)
        states = self.decodeTransformerBlock(action_emb, goal_emb, local_context_emb, environment_emb, term_t, tactic_t,
                                             action_mask)

        for step in range(states.size(1)):
            indice = self.gather_frontier_info(frontiers)
            if indice == []:
                break
            for idx in indice:
                stack = frontiers[idx]
                node = stack.pop()
                node.state = states[idx][step]
                g = {k: v[idx] for k, v in goal.items()}
                loss.add(self.expand_partial_tree(node, expansion_step, nonterminal_expansion_step[idx],
                                                  environment[idx], local_context[idx], g, actions[idx],
                                                  teacher_forcing, stack))
                if isinstance(node, NonterminalNode):
                    nonterminal_expansion_step[idx] += 1
            expansion_step += 1

        for ast in asts:
            ast.traverse_pre(clear_state)

        return asts, loss.value()

    def inference(self, environment, local_context, goal, actions, teacher_forcing, seq_embeddings=None,
                  gal_seq_embeddings=None):

        if not teacher_forcing:
            # when train without teacher forcing, only consider the expansion of non-terminal nodes
            actions = [[a for a in act if isinstance(a, int)] for act in actions]

        # initialize the trees
        batchsize = goal['embeddings'].size(0)
        # ast: [0,0,0,...] frontiers: [[ast],[ast],...]
        asts, frontiers = self.initialize_trees(batchsize)
        pre_action = [[] for i in range(batchsize)]

        # expand the trees in a depth-first order
        expansion_step = 0
        nonterminal_expansion_step = [0 for i in range(batchsize)]

        bos = self.action_emb.action_embedding(torch.LongTensor([98]).to(self.opts.device)).expand(batchsize, -1, -1)
        pad = torch.zeros(size=(batchsize, 4, self.opts.hidden_dim)).to(self.opts.device)
        action_emb = torch.cat([bos, pad], dim=1)

        goal_emb = goal['embeddings']
        local_context_emb = self.context_reader(goal_emb,
                                                [torch.cat([i['embeddings']], dim=0) for i in local_context])
        environment_emb = self.env_reader(goal_emb, [torch.cat([i['embeddings']], dim=0) for i in environment])
        term_t = self.term_reader(goal_emb, gal_seq_embeddings)
        tactic_t = self.tactic_reader(goal_emb, seq_embeddings)
        action_mask = torch.tril(torch.ones((batchsize, action_emb.size(1), action_emb.size(1)))).to(
            self.opts.device)

        for step in range(action_emb.size(1)):
            indice = self.gather_frontier_info(frontiers)
            if indice == []:
                break
            if pre_action[0] != []:
                action_emb = self.action_emb(pre_action)

            states = self.decodeTransformerBlock(action_emb, goal_emb, local_context_emb, environment_emb, term_t,
                                                 tactic_t, action_mask)

            # store states and expand nodes
            for j, idx in enumerate(indice):
                stack = frontiers[idx]
                node = stack.pop()
                node.state = states[idx][step]
                g = {k: v[idx] for k, v in goal.items()}
                pre_action[idx].append(self.expand_partial_tree(node, expansion_step, nonterminal_expansion_step[idx],
                                                                environment[idx], local_context[idx], g, actions[idx],
                                                                teacher_forcing, stack, True))
                if isinstance(node, NonterminalNode):
                    nonterminal_expansion_step[idx] += 1
            expansion_step += 1

        for ast in asts:
            ast.traverse_pre(clear_state)

        return asts

    def duplicate(self, ast, stack):
        old2new = {}

        def recursive_duplicate(node, parent=None):
            if isinstance(node, Token):
                new_node = deepcopy(node)
                old2new[node] = new_node
                return new_node
            elif isinstance(node, TerminalNode):
                new_node = TerminalNode(node.symbol, parent)
                new_node.token = node.token
            else:
                assert isinstance(node, NonterminalNode)
                new_node = NonterminalNode(node.symbol, parent)

            old2new[node] = new_node
            new_node.action = node.action
            if node.pred is None:
                new_node.pred = None
            else:
                new_node.pred = old2new[node.pred]
            new_node.state = node.state
            if isinstance(node, NonterminalNode):
                for c in node.children:
                    new_node.children.append(recursive_duplicate(c, new_node))
            return new_node

        new_ast = recursive_duplicate(ast)
        new_stack = [old2new[node] for node in stack]
        return new_ast, new_stack

    def beam_search(self, environment, local_context, goal, seq_embeddings=None, gal_seq_embeddings=None):
        # initialize the trees in the beam
        assert goal['embeddings'].size(0) == 1  # only support batchsize == 1
        beam, frontiers = self.initialize_trees(1)
        log_likelihood = [0.]  # the (unnormalized) objective function maximized by the beam search
        complete_trees = []  # the complete ASTs generated during the beam search

        expansion_step = 0

        pre_action = [[]]

        bos = self.action_emb.action_embedding(torch.LongTensor([98]).to(self.opts.device)).expand(1, -1, -1)
        pad = torch.zeros(size=(1, 4, self.opts.hidden_dim)).to(self.opts.device)
        action_emb = torch.cat([bos, pad], dim=1)

        goal_emb = goal['embeddings']
        local_context_emb = self.context_reader(goal_emb, [torch.cat([local_context['embeddings']], dim=0)])
        environment_emb = self.env_reader(goal_emb, [torch.cat([environment['embeddings']], dim=0)])
        term_t = gal_seq_embeddings
        tactic_t = gal_seq_embeddings
        action_mask = torch.tril(torch.ones((1, action_emb.size(1), action_emb.size(1)))).to(
            self.opts.device)

        for step in range(action_emb.size(1)):
            # collect inputs from all partial trees
            indice = self.gather_frontier_info(frontiers)
            # check if there are complete trees
            for i in range(len(beam)):
                if i not in indice:
                    normalized_log_likelihood = log_likelihood[i] / (
                            expansion_step ** self.opts.lens_norm)  # length normalization
                    beam[i].traverse_pre(clear_state)
                    complete_trees.append((beam[i], normalized_log_likelihood))
            if indice == []:  # all trees are complete, terminate the beam search
                break

            if pre_action[0] != []:
                action_emb = self.action_emb(pre_action)
            states = self.decodeTransformerBlock(action_emb, goal_emb, local_context_emb, environment_emb, term_t,
                                                 tactic_t, action_mask)

            # compute the log likelihood and pick the top candidates
            beam_candidates = []

            for idx in indice:
                stack = frontiers[idx]
                node = stack[-1]
                node.state = states[idx][step]

                if isinstance(node, NonterminalNode):
                    applicable_rules = self.grammar.get_applicable_rules(node.symbol)
                    if expansion_step > self.opts.size_limit:  # end the generation process asap
                        beam_candidates.append((idx, log_likelihood[i], applicable_rules[0]))
                        pre_action[0].append(applicable_rules[0])
                    else:
                        logits = torch.matmul(self.production_rule_embeddings.weight[applicable_rules],
                                              self.state_decoder(node.state))
                        log_cond_prob = logits - logits.logsumexp(dim=0)
                        for n, cand in enumerate(applicable_rules):
                            beam_candidates.append((idx, log_likelihood[idx] + log_cond_prob[n].item(), cand))
                        pre_action[0].append(applicable_rules[logits.argmax()])

                elif node.symbol in ['QUALID', 'LOCAL_IDENT']:
                    if node.symbol == 'QUALID':
                        candidates = environment['idents'] + local_context['idents']
                    else:
                        candidates = local_context['idents']
                    if candidates == []:
                        candidates = ['H'] + goal['quantified_idents']
                        log_cond_prob = - math.log(len(candidates))
                        for cand in candidates:
                            beam_candidates.append((idx, log_likelihood[idx] + log_cond_prob, cand))
                        pre_action[0].append(['H'] + goal['quantified_idents'])

                    else:
                        if node.symbol == 'QUALID':
                            candidate_embeddings = torch.cat([environment['embeddings'], local_context['embeddings']])
                        else:
                            candidate_embeddings = local_context['embeddings']
                        context_scores = self.pointer_net(node.state, candidate_embeddings)
                        log_cond_prob = context_scores - context_scores.logsumexp(dim=0)
                        for n, cand in enumerate(candidates):
                            beam_candidates.append((idx, log_likelihood[idx] + log_cond_prob[n].item(), cand))
                        pre_action[0].append(candidates[context_scores.argmax()])

                elif node.symbol == 'INT':
                    cls = self.INT_classifier(node.state)
                    log_cond_prob = cls - cls.logsumexp(dim=0)
                    for n in range(cls.size(0)):
                        beam_candidates.append((idx, log_likelihood[idx] + log_cond_prob[n].item(), str(n + 1)))
                    pre_action[0].append(['1', '2', '3', '4'][cls.argmax()])

                elif node.symbol == 'HINT_DB':
                    cls = self.HINT_DB_classifier(node.state)
                    log_cond_prob = cls - cls.logsumexp(dim=0)
                    for n in range(cls.size(0)):
                        beam_candidates.append((idx, log_likelihood[idx] + log_cond_prob[n].item(), self.hint_dbs[n]))
                    pre_action[0].append(self.hint_dbs[cls.argmax()])

                elif node.symbol == 'QUANTIFIED_IDENT':
                    if len(goal['quantified_idents']) > 0:
                        candidates = list(goal['quantified_idents'])
                    else:
                        candidates = ['x']
                    log_cond_prob = - math.log(len(candidates))
                    for cand in candidates:
                        beam_candidates.append((idx, log_likelihood[idx] + log_cond_prob, cand))
                    pre_action[0].append(candidates[0])

            # expand the nodes and update the beam
            beam_candidates = sorted(beam_candidates, key=lambda x: x[1], reverse=True)[:self.opts.beam_width]
            new_beam = []
            new_frontiers = []
            new_log_likelihood = []
            for idx, log_cond_prob, action in beam_candidates:
                ast, stack = self.duplicate(beam[idx], frontiers[idx])
                node = stack.pop()
                if isinstance(action, int):  # expand a nonterimial node
                    rule = self.grammar.production_rules[action]
                    self.expand_node_set_pred(node, rule, stack)
                else:  # expand a terminal node
                    node.expand(action)
                new_beam.append(ast)
                new_frontiers.append(stack)
                new_log_likelihood.append(log_likelihood[idx] + log_cond_prob)
            beam = new_beam
            frontiers = new_frontiers
            log_likelihood = new_log_likelihood
            expansion_step += 1

        complete_trees = sorted(complete_trees, key=lambda x: x[1], reverse=True)  # pick the top ASTs
        return [t[0] for t in complete_trees[:self.opts.num_tactic_candidates]]
