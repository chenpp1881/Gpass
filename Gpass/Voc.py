from glob import glob
import os
import pickle

# datapath = '../proof_step'
# proof_steps = glob(os.path.join(datapath, 'train', '*.pickle'))
#
# for proof_step_path in proof_steps:
#     proof_step = pickle.load(open(proof_step_path, 'rb'))
#     a = proof_step['prev_tokens']


a = pickle.load(open('char_voc.pkl', 'rb'))
a['@'] = 57
open("char_voc.pkl", "wb").write(pickle.dumps(a))