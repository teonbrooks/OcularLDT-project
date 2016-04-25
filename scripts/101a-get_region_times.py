from glob import glob
import os.path as op
import numpy as np

import pyeparse as pp
import config
import config_raw
from _recode_events import _recode_events


path = config.drive
redo = config.redo
exp = config.exp

group_ds = list()
fname_group = op.join(path, 'group', 'group_%s_region_times.txt' % exp)

for subject, experiments in config_raw.subjects.items():
    print config.banner % subject

    # Define output
    fname = op.join(path, subject, 'edf', subject + '_%s_region_times.txt' % exp)

    ds = [list(), list(), list(), list(),
          list(), list(), list(), list(), list()]
    if exp == 'OLDT':
        exps = [experiments[0], experiments[2]]
    else:
        exps = [experiments[1]]
    if 'n/a' in exps:
        exps.pop(exps.index('n/a'))
    for ii, exp in enumerate(exps):
        # Define filenames
        fname_trial = glob(op.join(path, subject, 'edf',
                           '*_Sime_Sent_*BLOCKTRIAL.dat'))[0]
        file_raw = op.join(path, subject, 'edf', '%s_%s.edf' % (subject, exp))
        # extracting triggering info from datasource file.
        # prime trigger is index 8, target trigger is index 10
        dat = np.loadtxt(fname_trial, dtype=str, delimiter='\t')
        if exp.startswith('OLDT'):
            prime_triggers = dat[:, 8].astype(int)
            target_triggers = dat[:,10].astype(int)
            prime_exp = np.array([(x & 2 ** 5) >> 5 for x in prime_triggers],
                                 dtype=bool)
            target_exp = np.array([(x & 2 ** 5) >> 5 for x in target_triggers],
                                  dtype=bool)
        if exp.startswith('SENT'):
            prime_triggers = dat[:, 4].astype(int)
            target_triggers = dat[:,8].astype(int)
            prime_exp = np.array([(x & 2 ** 4) >> 4 for x in prime_triggers],
                                 dtype=bool)
            target_exp = np.array([(x & 2 ** 4) >> 4 for x in target_triggers],
                                  dtype=bool)

        prime_triggers = prime_triggers[prime_exp]
        target_triggers = target_triggers[target_exp]

        # extracting fixation times from the edf file.
        raw = pp.read_raw(file_raw)
        times = list()
        triggers = list()
        trialids = list()
        ias = list()
        # first for the primes
        pat = '!V TRIAL_VAR TIME_PRIME'
        msgs = raw.find_events(pat, 1)[:,-1]
        prime_times = np.array([int(x[(len(pat) + 1):])
                                for x in msgs], int)
        prime_times = prime_times[prime_exp]
        assert prime_triggers.shape[0] == prime_times.shape[0]
        times.append(prime_times)
        triggers.append(prime_triggers)
        trialids.append(np.arange(len(prime_triggers)) + ii * 240)
        ias.append(['prime'] * len(prime_triggers))
        # then for the targets
        pat = '!V TRIAL_VAR TIME_TARGET'
        msgs = raw.find_events(pat, 1)[:,-1]
        target_times = np.array([int(x[(len(pat) + 1):])
                                 for x in msgs], int)
        target_times = target_times[target_exp]
        assert target_triggers.shape[0] == target_times.shape[0]
        times.append(target_times)
        triggers.append(target_triggers)
        trialids.append(np.arange(len(target_triggers)) + ii * 240)
        ias.append(['target'] * len(target_triggers))

        # let's do some re-arranging
        times = np.hstack(times)
        triggers = triggers_old = np.hstack(triggers)
        trialids = np.hstack(trialids)
        ias = np.hstack(ias)
        # coding trigger events
        evts = np.zeros((triggers.shape[0], 3))
        evts[:, 2] = triggers
        if exp == 'OLDT':
            evts, fix_idx, primes_idx, targets_idx, \
                semantic_idx, nonwords_idx = _recode_events(exp, evts)
        else:
            evts, fix_idx, primes_idx, targets_idx, \
                semantic_idx = _recode_events(exp, evts)
        triggers = evts[:, -1]

        semantics = np.zeros(len(triggers), int)
        semantics[semantic_idx] = 1
        words = np.ones(len(triggers), int)
        if exp == 'OLDT':
            words[nonwords_idx] = 0

        # dummy label
        labels = [subject] * triggers.shape[0]
        blocks = np.ones(triggers.shape[0], int) * ii

        ds[0].append(labels)
        ds[1].append(blocks)
        ds[2].append(trialids)
        ds[3].append(triggers)
        ds[4].append(triggers_old)
        ds[5].append(words)
        ds[6].append(semantics)
        ds[7].append(times)
        ds[8].append(ias)

    header = ['subject', 'block', 'trial', 'trigger', 'trigger_old',
              'word', 'priming', 'dur', 'ia']
    ds = [np.hstack(d) for d in ds]
    ds = np.vstack(ds).T
    ds = np.vstack((header, ds))

    np.savetxt(fname, ds, fmt='%s', delimiter=',')
    group_ds.append(ds)

group_ds = np.vstack(group_ds)
np.savetxt(fname_group, group_ds, fmt='%s', delimiter=',')
