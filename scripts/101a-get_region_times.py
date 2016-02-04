from glob import glob
import os.path as op
import numpy as np

import pyeparse as pp
import config
from _recode_events import _recode_events


path = config.drive
redo = config.redo

group_ds = list()
fname_group = op.join(path, 'group', 'group_OLDT_region_times.txt')

for subject in config.subjects:
    print config.banner % subject

    # Define output
    file_ds = op.join(path, subject, 'edf',
                      subject + '_OLDT_region_times.txt')

    ds = [list(), list(), list(), list(), list(), list(), list()]
    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    if 'n/a' in exps:
        exps.pop(exps.index('n/a'))
    for ii, exp in enumerate(exps):
        # Define filenames
        fname_trial = glob(op.join(path, subject, 'edf',
                           '*_%s_*BLOCKTRIAL.dat' % exp))[0]
        file_raw = op.join(path, subject, 'edf', '%s_%s.edf' % (subject, exp))
        # extracting triggering info from datasource file.
        # prime trigger is index 8, target trigger is index 10
        dat = np.loadtxt(fname_trial, dtype=str, delimiter='\t')
        prime_triggers = dat[:, 8].astype(int)
        prime_exp = np.array([(x & 2 ** 5) >> 5 for x in prime_triggers],
                             dtype=bool)
        prime_triggers = prime_triggers[prime_exp]
        target_triggers = dat[:,10].astype(int)
        target_exp = np.array([(x & 2 ** 5) >> 5 for x in target_triggers],
                              dtype=bool)
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
        triggers = np.hstack(triggers)
        trialids = np.hstack(trialids)
        ias = np.hstack(ias)
        # coding trigger events
        evts = np.zeros((triggers.shape[0], 3))
        evts[:, 2] = triggers
        evts, fix_idx, primes_idx, targets_idx, \
            semantic_idx, nonwords_idx = _recode_events(exp, evts)
        triggers = evts[:, -1]

        semantics = np.zeros(len(triggers), bool)
        semantics[semantic_idx] = 1
        words = np.ones(len(triggers), bool)
        words[nonwords_idx] = 0

        # dummy label
        labels = [subject] * triggers.shape[0]

        ds[0].append(labels)
        ds[1].append(trialids)
        ds[2].append(triggers)
        ds[3].append(words)
        ds[4].append(semantics)
        ds[5].append(times)
        ds[6].append(ias)

    header = ['subject', 'trial', 'trigger', 'word', 'priming',
              'dur', 'ia']
    ds = [np.hstack(d) for d in ds]
    ds = np.vstack(ds).T
    ds = np.vstack((header, ds))

    np.savetxt(file_ds, ds, fmt='%s', delimiter=',')
    group_ds.append(ds)

group_ds = np.vstack(group_ds)
np.savetxt(file_ds, ds, fmt='%s', delimiter=',')
