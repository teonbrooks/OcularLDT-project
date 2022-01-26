import os.path as op
import numpy as np
from pandas import read_table
import config
from mne import write_events


path = config.drive
exp = config.exp
redo = config.redo

for subject in config.subjects:
    print config.banner % subject

    fname_template = op.join(path, subject, '%s', '_'.join((subject, exp)))
    fname_meg = fname_template % 'mne' + '_meg_trial_struct.txt'
    fname_em = fname_template % 'edf' + '_region_times.txt'
    fname_dm = fname_template % 'mne' + '_region_design_matrix.txt'
    fname_eve = fname_template % 'mne' + '_region_coreg-eve.txt'

    if not op.exists(fname_dm) or redo:
        meg_ds = read_table(fname_meg, sep=',')
        meg_ds = meg_ds[meg_ds['trigger'] != 128]
        em_ds = read_table(fname_em, sep=',')

        lookup = {key: idx for idx, key in
                  enumerate(zip(meg_ds['trial'], meg_ds['trigger']))}

        interests = zip(em_ds['trial'], em_ds['trigger'])
        i_starts = list()
        durations = list()
        triggers = list()
        for ii, dur in zip(interests, em_ds['dur']):
            try:
                trial = meg_ds.iloc[lookup[ii]]
                i_starts.append(trial['i_start'])
                triggers.append(trial['trigger'])
                durations.append(dur)
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
