from glob import glob
import tomllib as toml
import os.path as op
from pathlib import Path
import numpy as np

import eyelinkio as eio
from scripts._helper import recode_events, find_eyetrack_events

cfg = toml.load(open(Path('./config.toml'), 'rb'))
task = cfg['task']
bids_root = cfg['bids_root']
project_path = cfg['project_path']
fname_group = Path('.') / 'output' / 'group' / f'group_{task}_region_times.tsv'
group_ds = list()

for subject, exps in cfg['exp_list'].items():
    print(cfg['banner'] % subject)

    # Define output
    basename = op.join(bids_root, f"sub-{subject}", 'eyetrack')
    fname = op.join(basename, f'{subject}_{task}_region_times.tsv')

    ds = [list() for _ in range(8)]

    for ii, exp in enumerate(exps, 1):
        if exp == 'n/a':
            continue
        # Define filenames
        fname_trial = glob(op.join(basename,
                           f'*_task-{task}_run-{ii:02d}_log.dat'))[0]
        file_raw = op.join(basename,
                           f'sub-{subject}_task-{task}_run-{ii:02d}_eyetrack.edf')
        # extracting triggering info from datasource file.
        # prime trigger is index 8, target trigger is index 10
        dat = np.loadtxt(fname_trial, dtype=str, delimiter='\t')
    
        prime_triggers = dat[:, 8].astype(int)
        target_triggers = dat[:,10].astype(int)
        prime_exp = np.array([(x & 2 ** 5) >> 5 for x in prime_triggers],
                                dtype=bool)
        target_exp = np.array([(x & 2 ** 5) >> 5 for x in target_triggers],
                                dtype=bool)

        prime_triggers = prime_triggers[prime_exp]
        target_triggers = target_triggers[target_exp]

        # extracting fixation times from the edf file.
        raw = eio.read_edf(file_raw)
        times = list()
        triggers = list()
        trialids = list()
        ias = list()

        # first for the primes
        pat = '!V TRIAL_VAR TIME_PRIME'
        msgs = find_eyetrack_events(raw, pat)
        prime_times = np.array([int(x[(len(pat) + 1):])
                                for x in msgs], int)
        prime_times = prime_times[prime_exp]
        assert prime_triggers.shape[0] == prime_times.shape[0]
        times.append(prime_times)
        triggers.append(prime_triggers)
        # trial ids for primes
        trialids.append(np.arange(len(prime_triggers)) + ii * 240)
        ias.append(['prime'] * len(prime_triggers))

        # then for the targets
        pat = '!V TRIAL_VAR TIME_TARGET'
        msgs = find_eyetrack_events(raw, pat, 1)
        target_times = np.array([int(x[(len(pat) + 1):])
                                 for x in msgs], int)
        target_times = target_times[target_exp]
        assert target_triggers.shape[0] == target_times.shape[0]
        times.append(target_times)
        triggers.append(target_triggers)
        #trial ids for targets
        trialids.append(np.arange(len(target_triggers)) + ii * 240)
        ias.append(['target'] * len(target_triggers))

        # let's do some re-arranging
        times = np.hstack(times)
        triggers = np.hstack(triggers)
        trialids = np.hstack(trialids)
        ias = np.hstack(ias)

        # recode triggers for consistency across the experiment
        triggers = recode_events(triggers)
        priming = (triggers & 2 ** 2) >> 2
        words = (triggers  & 2 ** 3) >> 3

        # dummy label
        sub_labels = [subject] * triggers.shape[0]
        blocks = np.ones(triggers.shape[0], int) * ii

        ds[0].append(sub_labels)
        ds[1].append(blocks)
        ds[2].append(trialids)
        ds[3].append(triggers)
        ds[4].append(words)
        ds[5].append(priming)
        ds[6].append(times)
        ds[7].append(ias)

    header = ['subject', 'block', 'trial', 'trigger',
              'word', 'priming', 'dur', 'ia']
    ds = [np.hstack(d) for d in ds]
    ds = np.vstack(ds).T
    ds = np.vstack((header, ds))

    np.savetxt(fname, ds, fmt='%s', delimiter='\t')
    group_ds.append(ds)

group_ds = np.vstack(group_ds)
np.savetxt(fname_group, group_ds, fmt='%s', delimiter='\t')
