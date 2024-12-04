# TODO: update note at the top of doc
"""
Creating Event files

EM data are in a trial structure while MEG data are continuous. In order to
begin looking at these two data streams, we need to co-register in a common
reference frame.
Here, we're creating two files for all of the MEG analyses:

01. An event file with all the fixation crosses, primes, and targets.
02. A co-registering event file to match EM data for fix, prime, target regions.
"""
import os
import os.path as op
import numpy as np
from pandas import DataFrame
import mne
import config
from _recode_coreg_events import _recode_events

redo = config.redo
path = config.drive
exp = config.exp
filt = config.filt

# update this to use the bids format
for subject in config.subjects:
    print(config.banner % subject)

    fname_template = op.join(path, subject, 'mne', '_'.join((subject, exp)))
    fname_evts = fname_template + '-eve.txt'
    fname_raw = fname_template + '_calm_' + filt + '_filt-raw.fif'
    # write the co-registration file
    fname_trial = fname_template + '_meg_trial_struct.txt'

    if not op.exists(fname_evts) or redo:
        raw = mne.io.read_raw_fif(fname_raw)

        # E-MEG alignment
        evts = mne.find_stim_steps(raw, merge=-2)
        # select non-zero events
        idx = np.nonzero(evts[:, 2])[0]
        evts = evts[idx]
        # recode events
        triggers = evts[:, 2]
        
        i_start, prev_trig, trig, new = evt
        triggers = _recode_events(triggers)
        evts = DataFrame({'fix_pos': triggers['fix_pos'],
                          'stime': evts[:, 0],
                          'prev_trig': evts[:, 1],
                          'old_trig': triggers['old_trig'],
                          'trig': triggers['trig']})
        # now dropping practice trials
        evts = evts[evts['trig'] == 0]
        evts = evts.sort_values(by='stime')

        """
        Writing the co-registration event file
        --------------------------------------
        The MEG and EM files must be aligned to have a proper decoding.
        This arranges the triggers in events of interests: prime, target.
        These events are consistent and identifiable in both data types.

        They are selected and rearranged according to timestamps. This allows
        for ease of matching later on.

        """

        coreg_evts = list()

        # TODO: just use pandas
        # TODO: add a trial column and update it with the logic
        with open(fname_trial, 'w') as FILE:
            FILE.write('# MEG coregistration file' + os.linesep)
            header = ['trial_no', 'fix_pos,' 'stime', 'old_trigger', 'trigger']
            FILE.write(','.join(header) + os.linesep)
            ii = -1
            for _, evt in evts.iterrows():
                # at the start of the Experiment, the trigger reset to zero.
                # starting trial at 0
                if evt['prev_trig'] == 255:
                    continue
                elif evt['prev_trig'] == 0:
                    ii += 1
                coreg_evts.append(evt)
                trial = [ii, evt['fix_pos'], evt['stime'],
                         evt['old_trig'], evt['trig']]
                trial = ','.join(map(str, trial)) + '\n'
                FILE.write(trial)

        # coreg event list
        coreg_evts = DataFrame(coreg_evts)
        evts = coreg_evts[['trial_no', 'stime', 'trigger']].to_numpy()
