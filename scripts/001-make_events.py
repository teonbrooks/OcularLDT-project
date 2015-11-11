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

redo = config.redo
path = config.drive


def make_events(raw, subject, exp):
    # E-MEG alignment
    evts = mne.find_stim_steps(raw, merge=-2)
    # trial alignment
    trials = deepcopy(evts)

    if exp.startswith('OLDT'):
        expt = np.array([(x & 2 ** 5) >> 5 for x in evts[:, 2]], dtype=bool)
        evts = evts[expt]
        idx = np.nonzero(evts[:, 2])[0]
        evts = evts[idx]
        triggers = evts[:, 2]

        semantic = np.array([(x & 2 ** 4) >> 4 for x in triggers], dtype=bool)
        nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2 for x in triggers])
        current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0 for x in triggers])

        idx = np.where(nonword_pos - current_pos != 0)[0]
        idy = np.where(current_pos < nonword_pos)[0]
        idy2 = np.where(nonword_pos == 0)[0]
        idy3 = np.where(current_pos != 0)[0]
        idy = np.unique(np.hstack((idy, idy2, idy3)))
        n_idx = np.where(nonword_pos - current_pos == 0)[0]
        n_idy = np.where(nonword_pos != 0)[0]

        # nonwords vs words
        words_idx = np.intersect1d(idx, idy)
        nonwords_idx = np.intersect1d(n_idx, n_idy)
        # fixation
        fix_idx = np.where(current_pos == 0)[0]
        # prime vs target
        primes_idx = np.where(current_pos == 1)[0]
        targets_idx = np.where(current_pos == 2)[0]
        # semantic
        semantic_idx = np.where(semantic)[0]

    elif exp.startswith('SENT'):
        expt = np.array([(x & 2 ** 4) >> 4 for x in evts[:, 2]], dtype=bool)
        evts = evts[expt]
        idx = np.nonzero(evts[:, 2])[0]
        evts = evts[idx]
        triggers = evts[:, 2]

        semantic = np.array([(x & 2 ** 3) >> 3 for x in triggers], dtype=bool)
        current_pos = np.array([(x & (2 ** 2 + 2 ** 1 + 2 ** 0)) >> 0
                                for x in triggers])

        semantic_idx = np.where(semantic)[0]
        fix_idx = np.where(current_pos == 0)[0]
        primes_idx = np.where(current_pos == 1)[0]
        targets_idx = np.where(current_pos == 3)[0]

    else:
        raise ValueError('This function only works for '
                         'OLDTX or SENTX experiments, not %s.' % exp)

    """
    Recoding Events
    ---------------
    To deal with the way MNE deals with events with multiple tagging, we
    devised a way to handle this by repacking tag info in a binary fashion.
    Maybe next time I will code my events just directly as such
    (consult schematic in lab notebook).

    Prime += 1
    Target += 2
    Priming += 4
    Nonword += 8

    This recoding efficiency allows for Hierarchial Event Descriptors
    (HED tags).
    """
    recodeds = np.zeros((evts.shape[0], 1), int)
    evts = np.hstack((evts, recodeds))
    # prime vs target
    evts[primes_idx, -1] += 1
    evts[targets_idx, -1] += 2
    # semantic priming
    evts[semantic_idx, -1] += 4
    if exp.startswith('OLDT'):
        # word vs nonword
        evts[nonwords_idx, -1] += 8
    # fixation
    evts[fix_idx, -1] = 128

    """
    Writing the co-registration event file
    --------------------------------------
    The MEG and EM files must be aligned to have a proper decoding.
    This arranges the triggers in events of interests: prime, target.
    These events are consistent and identifable in both data types.

    They are selected and rearranged according to timestamps. This allows for
    ease of matching later on.

    """

    idx = np.hstack((fix_idx, primes_idx, targets_idx))
    evts = evts[idx]
    idx = zip(evts[:, 0], np.arange(evts.size))
    idx = list(zip(*sorted(idx))[-1])
    evts = evts[idx]
    path = op.dirname(raw.info['filename'])

    # write the co-registration file
    fname_trial = op.join(path, '..', 'mne', '%s_%s_meg_trial_struct.txt'
                          % (subject, exp))
    fname_coreg = op.join(path, '..', 'mne', '%s_%s-eve.txt'
                          % (subject, exp))
    coreg_evts = list()

    with open(fname_trial, 'w') as FILE:
        header = ['trial', 'i_start', 'trigger', 'recoded_trigger']
        FILE.write(','.join(header) + '\n')
        ii = -1
        for evt in evts:
            i_start, _, trig, new = evt
            # at the start of the Experiment, the trigger reset to zero.
            if evt[1] == 255:
                continue
            elif evt[1] == 0:
                ii += 1
            coreg_evts.append(evt)
            trial = [ii, i_start, trig, new]
            trial = ','.join(map(str, trial)) + '\n'
            FILE.write(trial)
    # coreg event list
    coreg_evts = np.array(coreg_evts)
    evts = coreg_evts[:, [0, -1]]
    starts = np.zeros(len(evts), int)
    evts = np.vstack((evts[:, 0], starts, evts[:, -1])).T
    mne.write_events(fname_coreg, evts)

for subject in config.subjects:
    print config.banner % subject

    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    evt_file = op.join(path, subject, 'mne', subject + '_OLDT-eve.txt')

    if not op.exists(evt_file) or redo:
        if 'n/a' in exps:
            exps.pop(exps.index('n/a'))
            raw = config.kit2fiff(subject=subject, exp=exps[0],
                                  path=path, dig=False, preload=False)
        else:
            raw = config.kit2fiff(subject=subject, exp=exps[0],
                                  path=path, dig=False, preload=False)
            raw2 = config.kit2fiff(subject=subject, exp=exps[1],
                                  path=path, dig=False, preload=False)
            mne.concatenate_raws([raw, raw2])
        make_events(raw, subject, 'OLDT')
