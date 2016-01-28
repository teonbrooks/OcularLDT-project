import os.path as op
import numpy as np
from pandas import read_table
import config
from mne import write_events


path = config.drive
exp = 'OLDT'
redo = config.redo

for subject in config.subjects:
    print config.banner % subject

    fname_meg = op.join(path, subject, 'mne', '%s_%s_meg_trial_struct.txt'
                        % (subject, exp))
    fname_em = op.join(path, subject, 'edf', '%s_%s_fixation_times.txt'
                       % (subject, exp))
    fname_dm = op.join(path, subject, 'mne', '%s_%s_fixation_design_matrix.txt'
                       % (subject, exp))
    fname_eve = op.join(path, subject, 'mne', '%s_%s_coreg-eve.txt'
                        % (subject, exp))

    if not op.exists(fname_dm) or redo:
        meg_ds = read_table(fname_meg, sep=',')
        meg_ds = meg_ds[meg_ds['recoded_trigger'] != 128]
        em_ds = read_table(fname_em, sep=',')
        # em_ds = em_ds[em_ds['ia'] == 'target']

        lookup = {key: idx for key, idx in zip(zip(meg_ds['trial'],
                  meg_ds['trigger']), range(len(meg_ds['trigger'])))}

        interest = zip(em_ds['trial'], em_ds['trigger'])
        i_starts = list()
        durations = list()
        triggers = list()
        for ii, dur in zip(interest, em_ds['ffd']):
            try:
                i_starts.append(meg_ds.iloc[lookup[ii]]['i_start'])
                durations.append(dur)
                triggers.append(meg_ds.iloc[lookup[ii]]['recoded_trigger'])
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
