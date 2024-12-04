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
import numpy as np
from pandas import DataFrame


def _recode_events(df):
    df_trig = df[df['metric'] == 'ttl']
    triggers = np.array(df_trig['msg'].str.extract(r'TTL ([0-9]+)'))
    triggers = triggers.astype(int).ravel()
    trial_nos = np.array(df_trig['trial_no'])
    stimes = np.array(df_trig['stime'])
    
    # remove practice
    expt = np.array([(x & 2 ** 5) >> 5 for x in triggers], dtype=bool)
    triggers = triggers[expt]

    # unpack trigger information
    semantic = np.array([(x & 2 ** 4) >> 4 for x in triggers],
                        dtype=bool)
    nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2
                            for x in triggers])
    current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0
                            for x in triggers])

    idx = np.where(nonword_pos - current_pos != 0)[0]
    # word region
    idy = np.where(np.logical_and(current_pos > 0,
                                  current_pos < nonword_pos))[0]
    # nonword is absent
    idy2 = np.where(nonword_pos == 0)[0]
    # not fixation cross
    idy3 = np.where(current_pos != 0)[0]
    idy = np.unique(np.hstack((idy, idy2, idy3)))
    # find nonwords
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


    ############
    # Recoding #
    ############
    recodeds = np.zeros(triggers.shape[0], int)
    # prime vs target
    recodeds[primes_idx] += 1
    recodeds[targets_idx] += 2
    # semantic priming
    recodeds[semantic_idx] += 4
    # word vs nonword
    recodeds[nonwords_idx] += 8
    # fixation
    recodeds[fix_idx] = 128

    new_trig = {'trial': trial_nos,
                'fix_pos': current_pos,
                'old_trig': triggers,
                'trig': recodeds,
                'i_start': stimes}
    new_trig = DataFrame(new_trig)
    new_trig = new_trig.sort_values(by=['trial_no', 'fix_pos'])
        
    return new_trig