import json
import os.path as op
import numpy as np
import scipy as sp

import mne
from mne.report import Report
from mne.stats import (permutation_cluster_1samp_test as pc_1samp_test)

cfg = json.load(open(op.join('/', 'Users', 'teonbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
# parameters
random_state = 42
# decoding parameters
tmin, tmax = -.1, 1
n_folds = 5
# baseline
bmin, bmax = -.2, -.1
reject = cfg['reject']
c_names = ['word/target/primed', 'word/target/unprimed']
chance = .5

# setup group
fname_group_scores = op.join(cfg['project_path'], 'output', 'group',
                             f'group_{task}_sensor_priming_scores.npy')
fname_group_patterns = op.join(cfg['project_path'], 'output', 'group',
                               f'group_{task}_sensor_priming_patterns.npy')


####################
# Group Statistics #
####################
group_scores = np.load(fname_group_scores)

########################
# run a TD clustering  #
########################
# remove chance from the gats
group_scores -= chance
group_stats = pc_1samp_test(group_scores, n_permutations=10000,
                            threshold=1.96, tail=0,
                            seed=42, n_jobs=-1)
