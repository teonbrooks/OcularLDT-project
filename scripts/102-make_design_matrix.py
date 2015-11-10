import os.path as op
import numpy as np
from pandas import read_table
import config


path = config.drive
exp = 'OLDT'
redo = config.redo

for subject in config.subjects:
    print config.banner % subject

    fname_meg = op.join(path, subject, 'mne', '%s_%s_trials.txt'
                        % (subject, exp))
    fname_em = op.join(path, subject, 'edf', '%s_%s_fixation_times.txt'
                       % (subject, exp))
    fname_dm = op.join(path, subject, 'mne', '%s_%s_design_matrix.txt'
                       % (subject, exp))

    if not op.exists(fname_dm) or redo:
        meg_ds = read_table(fname_meg, sep='\t')
        em_ds = read_table(fname_em, sep=',')
        em_ds = em_ds[em_ds['ia'] == 'target']

        lookup = {key: idx for key, idx in zip(zip(em_ds['trial'],
                  em_ds['trigger']), range(len(em_ds['trigger'])))}

        interest = zip(meg_ds['trialid'] - 1, meg_ds['trigger'])
        durations = list()
        for ii in interest:
            durations.append(em_ds.irow(lookup[ii])['ffd'])
        intercepts = np.ones(len(durations))

        design_matrix = np.vstack((intercepts, durations)).T
        np.savetxt(fname_dm, design_matrix, fmt='%s', delimiter='\t')
