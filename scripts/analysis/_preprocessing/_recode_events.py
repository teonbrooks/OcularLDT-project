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


def _recode_events(exp, evts, idx=True):
    evts = evts.astype(int)
    if exp.startswith('OLDT'):
        # remove practice
        expt = np.array([(x & 2 ** 5) >> 5 for x in evts[:, 2]], dtype=bool)
        evts = evts[expt]
        # force type to int
        triggers = evts[:, 2]

        semantic = np.array([(x & 2 ** 4) >> 4 for x in triggers],
                            dtype=bool)
        nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2
                                for x in triggers])
        current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0
                                for x in triggers])

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

    else:
        raise ValueError('This function only works for '
                         'OLDTX experiments, not %s.' % exp)

    ############
    # Recoding #
    ############
    recodeds = np.zeros((evts.shape[0], 1), int)
    evts = np.hstack((evts, recodeds))
    # prime vs target
    evts[primes_idx, -1] += 1
    evts[targets_idx, -1] += 2
    # semantic priming
    evts[semantic_idx, -1] += 4
    # fixation
    evts[fix_idx, -1] = 128
    if exp.startswith('OLDT'):
        # word vs nonword
        evts[nonwords_idx, -1] += 8
        return evts, fix_idx, primes_idx, targets_idx, semantic_idx, nonwords_idx
    else:
        return evts, fix_idx, primes_idx, targets_idx, semantic_idx
