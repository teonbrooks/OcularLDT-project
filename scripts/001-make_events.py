"""
Creating Event files

EM data are in a trial structure while MEG data are continuous. In order to
begin looking at these two data streams, we need to co-register in a common
reference frame.
Here, we're creating two files for all of the MEG analyses:

01. An event file with all the fixation crosses, primes, and targets.
02. A co-registering event file to match EM data for fix, prime, target regions.
"""

import os.path as op
from copy import deepcopy
import numpy as np
import mne
import config
from _recode_events import _recode_events

redo = config.redo
path = config.drive
exp = 'OLDT'
filt = config.filt


for subject in config.subjects:
    print config.banner % subject

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
        evts, fix_idx, primes_idx, targets_idx, \
            semantic_idx, nonwords_idx = _recode_events(exp, evts)

        """
        Writing the co-registration event file
        --------------------------------------
        The MEG and EM files must be aligned to have a proper decoding.
        This arranges the triggers in events of interests: prime, target.
        These events are consistent and identifable in both data types.

        They are selected and rearranged according to timestamps. This allows
        for ease of matching later on.

        """

        idx = np.hstack((fix_idx, primes_idx, targets_idx))
        evts = evts[idx]
        idx = zip(evts[:, 0], np.arange(evts.size))
        idx = list(zip(*sorted(idx))[-1])
        evts = evts[idx]
        coreg_evts = list()

        with open(fname_trial, 'w') as FILE:
            header = ['trial', 'i_start', 'old_trigger', 'trigger']
            FILE.write(','.join(header) + '\n')
            ii = -1
            for evt in evts:
                i_start, prev_trig, trig, new = evt
                # at the start of the Experiment, the trigger reset to zero.
                # starting trial at 0
                if prev_trig == 255:
                    continue
                elif prev_trig == 0:
                    ii += 1
                coreg_evts.append(evt)
                trial = [ii, i_start, trig, new]
                trial = ','.join(map(str, trial)) + '\n'
                FILE.write(trial)
        # coreg event list
        evts = np.array(coreg_evts)[:, [0, 1, -1]]

        mne.write_events(fname_evts, evts)
