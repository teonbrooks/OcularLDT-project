import os.path as op
import numpy as np
from pandas import read_table
import config
from mne import write_events


path = config.drive
exp = config.exp
redo = config.redo
analysis = 'fixation'
depmeas = 'ffd'

for subject in config.subjects:
    print config.banner % subject

    fname_template = op.join(path, subject, '%s', '_'.join((subject, exp)))
    fname_meg = fname_template % 'mne' + '_meg_trial_struct.txt'
    fname_em = fname_template % 'edf' + '_%s_times.txt' % analysis
    fname_dm = fname_template % 'mne' + '_%s_design_matrix.txt' % analysis
    fname_eve = fname_template % 'mne' + '_%s_coreg-eve.txt' % analysis

    meg_ds = read_table(fname_meg, sep=',')
    meg_ds = meg_ds[meg_ds['trigger'] != 128]
    em_ds = read_table(fname_em, sep=',')

    lookup = {key: idx for idx, key in
              enumerate(zip(meg_ds['trial'], meg_ds['trigger']))}

    interest = zip(em_ds['trial'], em_ds['trigger'])
    i_starts = list()
    durations = list()
    triggers = list()
    for ii, dur in zip(interest, em_ds[depmeas]):
        try:
            i_starts.append(meg_ds.iloc[lookup[ii]]['i_start'])
            durations.append(dur)
            triggers.append(meg_ds.iloc[lookup[ii]]['trigger'])
        except KeyError:
            pass
    intercepts = np.ones(len(durations))
    # create and write out design matrix
    design_matrix = np.vstack((intercepts, durations)).T
    np.savetxt(fname_dm, design_matrix, fmt='%s', delimiter='\t')

    # create and write out a co-registered event file
    evts = np.zeros((len(i_starts), 3), int)
    evts[:, 0] = i_starts
    evts[:, 2] = triggers
    write_events(fname_eve, evts)
