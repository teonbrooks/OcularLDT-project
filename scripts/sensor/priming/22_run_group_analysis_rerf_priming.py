import os.path as op
import json
from tokenize import group
import numpy as np

import mne
from mne.stats import spatio_temporal_cluster_1samp_test as stc_1samp_test
from mne_bids import BIDSPath, get_entity_vals


# parameters
cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['project_name']
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
adjacency, _ = mne.channels.read_ch_adjacency('KIT-208')
# setup group
group_template = op.join(cfg['project_path'], 'output', 'results',
                         f'group_OcularLDT_sensor_repr_%s.json')

subjects = get_entity_vals(cfg['bids_root'], entity_key='subject')
c_names = ['word/target/unprimed', 'word/target/primed']

group_rerfs = dict(uncorrected = list(),
                   corrected = list())
for subject in subjects:
    bids_path.update(subject=subject)
    print(cfg['banner'] % subject)
    # define filenames
    subject_template = op.join(cfg['bids_root'], f'sub-{subject}',
                               'meg', f'sub-{subject}_task-{task}')
    fname_rerf_uncorrected = f'{subject_template}_priming_uncorrected_ave.fif'
    fname_rerf_corrected = f'{subject_template}_priming_corrected_ave.fif'

    rerfs = list()
    for fname in [fname_rerf_uncorrected, fname_rerf_corrected]:
        evokeds = mne.read_evokeds(fname)
        for ii, ev in enumerate(evokeds):
            if ev.comment == c_names[0]:
                idx_unprimed = ii
            elif ev.comment == c_names[1]:
                    idx_primed = ii
            else:
                raise ValueError('Missing a condition')
        rerfs.append(mne.evoked.combine_evoked([evokeds[idx_primed],
                                                evokeds[idx_unprimed]],
                                               weights=[1, -1]))
    group_rerfs['uncorrected'].append(rerfs[0].data.T)
    group_rerfs['corrected'].append(rerfs[1].data.T)


#############################
# run a spatio-temporal REG #
#############################
group_rerfs['uncorrected'] = np.array(group_rerfs['uncorrected'])
group_rerfs['corrected'] = np.array(group_rerfs['corrected'])

for comparison, group_data in group_rerfs.items():
    stats = stc_1samp_test(group_data, n_permutations=10000,
                        threshold=1.96, tail=0,
                        adjacency=adjacency,
                        seed=42, n_jobs=-1)

    json.dump(stats, open(group_template % comparison, 'w'),
              sort_keys=True, indent=4)
