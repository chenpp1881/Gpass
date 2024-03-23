from glob import glob
import os
import pickle

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

split = 'train'
proof_steps_path = glob(os.path.join('../proof_step', split, '*.pickle'))
max_family_path = 10
path_dict = {}
current_path = max_family_path * [len(nonterminals)]

def Get_fpath(node, current_path):
    global path_dict
    chirdren_path = current_path
    if node.children == []:
        return
    for c in node.children:
        path_dict[c] = chirdren_path
        current_path = current_path[1:] + [nonterminals.index(c.data)]
        Get_fpath(node=c, current_path=current_path)

for proof_step_path in proof_steps_path:
    proof_step = pickle.load(open(proof_step_path, 'rb'))
    print(proof_step_path)
    proof_step['goal'] = proof_step['goal']['ast']
    proof_step['tactic_actions'] = proof_step['tactic']['actions']
    proof_step['tactic_str'] = proof_step['tactic']['text']
    del proof_step['tactic']
    env_ast = [env['ast'] for env in proof_step['env']]
    context_ast = [context['ast'] for context in proof_step['local_context']]
    goal_ast = [proof_step['goal']]
    all_ast = env_ast + context_ast + goal_ast
    for ast in all_ast:
        path_dict[ast] = current_path
        current_path = current_path[1:] + [nonterminals.index(ast.data)]
        Get_fpath(node=ast,current_path=current_path)

open("ppath.pkl", "wb").write(pickle.dumps(path_dict))


