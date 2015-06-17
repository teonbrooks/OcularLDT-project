import pyeparse
import os.path as op
import numpy as np
import pandas
from scipy.stats import ttest_ind
import config
from glob import glob


path = config.drive
ds = [list(), list(), list(), list(), list(), list(), list()]
for subject in config.subjects:
    print subject
    for exp in [config.subjects[subject][0], config.subjects[subject][2]]:
        raw_file = op.join(path, subject, 'edf', '%s_%s.edf' % (subject, exp))

        # extracting triggering info from datasource file.
        # target trigger is index 10
        trial_file = glob(op.join(path, subject, 'edf', '*_%s_*BLOCKTRIAL.dat' % exp))[0]
        triggers = np.loadtxt(trial_file, dtype=str, delimiter='\t')[:,10].astype(int)

        # extracting fixation times from the edf file.
        raw = pyeparse.RawEDF(raw_file)
        pat = '!V TRIAL_VAR TIME_TARGET'
        msgs = raw.find_events(pat, 1)[:,-1]
        times = np.array([int(x[(len(pat) + 1):]) for x in msgs], int)
        assert triggers.shape[0] == times.shape[0]

        # # clean empty
        # idx = np.where(times != -1)[0]
        # triggers = triggers[idx]
        # times = times[idx]


        # coding trigger events
        semantics = np.array([(x & 2 ** 4) >> 4 for x in triggers], dtype=bool)
        nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2 for x in triggers])
        current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0 for x in triggers])

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

        ds[0].append(labels)
        ds[1].append(triggers)
        ds[2].append(words)
        ds[3].append(semantics)
        ds[4].append(times)
        ds[5].append(means)
        ds[6].append(stds)

    ds_file = op.join(path, subject, 'edf', '%s_OLDT_target_times.txt' % subject)
    ds = [np.hstack(d) for d in ds]
    ds = np.vstack(ds).T
    np.savetxt(ds_file, ds, fmt='%s')

