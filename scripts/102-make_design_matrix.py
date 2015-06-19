import os.path as op
import numpy as np
import pandas
import config


path = config.drive
exp = 'OLDT'

for subject in config.subjects:
    print subject
    meg_fname = op.join(path, subject, 'mne', '%s_%s_trials.txt'
                        % (subject, exp))
    em_fname = op.join(path, subject, 'edf', '%s_%s_target_times.txt'
                       % (subject, exp))
    dm_fname = op.join(path, subject, 'mne', '%s_%s_design_matrix.txt'
                       % (subject, exp))

    meg_ds = pandas.read_table(meg_fname, sep='\t')
    em_ds = pandas.read_table(em_fname, sep='\t')

    coreg = em_ds.loc[meg_ds['TrialID']]
    durations = coreg['duration']
    intercepts = np.ones(len(durations))

    design_matrix = np.vstack((intercepts, durations)).T
    np.savetxt(dm_fname, design_matrix, fmt='%s', delimiter='\t')