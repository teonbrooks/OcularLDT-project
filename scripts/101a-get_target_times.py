import pyeparse
import os.path as op
import numpy as np
import pandas
from scipy.stats import ttest_ind
import config
from glob import glob


path = config.drive
redo = config.redo


for subject in config.subjects:
    print config.banner % subject

    # Define output
    file_ds = op.join(path, subject, 'edf',
                      subject + '_OLDT_target_times.txt')
    if not op.exists(file_ds) or redo:
        ds = [list(), list(), list(), list(), list(), list(), list(), list()]
        exps = [config.subjects[subject][0], config.subjects[subject][2]]
        for exp in exps:
            # Define filenames
            fname_trial = glob(op.join(path, subject, 'edf',
                               '*_%s_*BLOCKTRIAL.dat' % exp))[0]
            file_raw = op.join(path, subject, 'edf', '%s_%s.edf' % (subject, exp))
            # extracting triggering info from datasource file.
            # prime trigger is index 8, target trigger is index 10
            dat = np.loadtxt(fname_trial, dtype=str, delimiter='\t')
            prime_triggers = dat[:, 8].astype(int)
            target_triggers = dat[:,10].astype(int)

            # extracting fixation times from the edf file.
            raw = pyeparse.RawEDF(file_raw)
            times = list()
            triggers = list()
            # first for the primes
            pat = '!V TRIAL_VAR TIME_PRIME'
            msgs = raw.find_events(pat, 1)[:,-1]
            prime_times = np.array([int(x[(len(pat) + 1):])
                                    for x in msgs], int)
            assert prime_triggers.shape[0] == prime_times.shape[0]
            times.append(prime_times)
            triggers.append(prime_triggers)
            # then for the targets
            pat = '!V TRIAL_VAR TIME_TARGET'
            msgs = raw.find_events(pat, 1)[:,-1]
            target_times = np.array([int(x[(len(pat) + 1):])
                                     for x in msgs], int)
            assert target_triggers.shape[0] == target_times.shape[0]
            times.append(target_times)
            triggers.append(target_triggers)


            times = np.hstack(times)
            triggers = np.hstack(triggers)
            # coding trigger events
            semantics = np.array([(x & 2 ** 4) >> 4
                                  for x in triggers], dtype=bool)
            nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2
                                    for x in triggers])
            current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0
                                    for x in triggers])

            # defining word vs. nonword
            words = np.zeros(triggers.shape[0])
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

            # trial id
            trialids = np.arange(triggers.shape[0]) + 1

            ds[0].append(labels)
            ds[1].append(trialids)
            ds[2].append(triggers)
            ds[3].append(words)
            ds[4].append(semantics)
            ds[5].append(times)
            ds[6].append(means)
            ds[7].append(stds)

        header = ['subject', 'trialids', 'triggers', 'words', 'semantics',
                  'duration', 'mean', 'std']
        ds = [np.hstack(d) for d in ds]
        ds = np.vstack(ds).T
        ds = np.vstack((header, ds))

        # # clean empty
        # idx = np.where(times != -1)[0]
        # ds = ds[idx]
    
        np.savetxt(file_ds, ds, fmt='%s', delimiter='\t')
