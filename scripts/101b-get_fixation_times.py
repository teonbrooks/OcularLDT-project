from glob import glob
import os.path as op
import numpy as np
import config
import pyeparse as pp
from pyeparse import interest_areas

path = config.drive
redo = config.redo

group_ds = list()
# Define OLDT interest areas
file_ia = op.join(path, 'group', 'OLDT_IAs.txt')
ia_coords = interest_areas.read_ia(file_ia)
ia_words = ['fixation', 'prime', 'target', 'post']
for subject in config.subjects:
    print config.banner % subject


    # Define output
    file_ds = op.join(path, subject, 'edf',
                      subject + '_OLDT_fixation_times.txt')
    if not op.exists(file_ds) or redo:
        ds = [list(), list(), list(), list(), list(), list(), list(), list()]
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
            raw = pp.RawEDF(file_raw)
            ias = interest_areas.Reading(raw, ia_coords, ia_words)
            prime_times = ias.get_first_fix(ia=1)
            prime_times = interest_areas.get_gaze_duration(ias['target'])
            # first for the primes
            pat = '!V TRIAL_VAR TIME_PRIME'
            msgs = raw.find_events(pat, 1)[:,-1]
            prime_times = np.array([int(x[(len(pat) + 1):])
                                    for x in msgs], int)
            prime_times = prime_times[prime_exp]
            assert prime_triggers.shape[0] == prime_times.shape[0]
            times.append(prime_times)
            triggers.append(prime_triggers)
            trialids.append(np.arange(len(prime_triggers)) + 1 + ii * 240)
            # then for the targets
            pat = '!V TRIAL_VAR TIME_TARGET'
            msgs = raw.find_events(pat, 1)[:,-1]
            target_times = np.array([int(x[(len(pat) + 1):])
                                     for x in msgs], int)
            target_times = target_times[target_exp]
            assert target_triggers.shape[0] == target_times.shape[0]
            times.append(target_times)
            triggers.append(target_triggers)
            trialids.append(np.arange(len(target_triggers)) + 1 + ii * 240)

            # let's do some re-arranging
            times = np.hstack(times)
            triggers = np.hstack(triggers)
            trialids = np.hstack(trialids)
            # coding trigger events
            semantics = np.array([(x & 2 ** 4) >> 4
                                  for x in triggers], dtype=int)
            nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2
                                    for x in triggers])
            current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0
                                    for x in triggers])

            # defining word vs. nonword
            words = np.zeros(triggers.shape[0], int)
            idx = np.where(nonword_pos - current_pos != 0)[0]
            idy = np.where(current_pos < nonword_pos)[0]
            idy2 = np.where(nonword_pos == 0)[0]
            idy = np.unique(np.hstack((idy, idy2)))
            words_idx = np.intersect1d(idx, idy)
            words[words_idx] = 1

            # dummy label
            labels = [subject] * triggers.shape[0]

            # make outlier params
            means = np.ones(times.shape[0]) * times.mean()
            stds = np.ones(times.shape[0]) * times.std()

            ds[0].append(labels)
            ds[1].append(trialids)
            ds[2].append(triggers)
            ds[3].append(words)
            ds[4].append(semantics)
            ds[5].append(times)
            ds[6].append(means)
            ds[7].append(stds)

        header = ['subject', 'trialid', 'trigger', 'word', 'semantics',
                  'duration', 'mean', 'std']
        ds = [np.hstack(d) for d in ds]
        ds = np.vstack(ds).T
        ds = np.vstack((header, ds))
        group_ds.append(ds)

        np.savetxt(file_ds, ds, fmt='%s', delimiter='\t')
