import os.path as op
from copy import deepcopy
import numpy as np
import mne


def make_events(raw, subject, exp):
    # E-MEG alignment
    evts = mne.find_stim_steps(raw)
    alignment = deepcopy(evts)
    idx = np.nonzero(alignment[:, 2])[0]
    alignment = alignment[idx]
    current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0 for x in
                            alignment[:, 2]])
    idx = np.where(current_pos == 1)[0]
    alignment[idx, 2] = 50
    alignment = alignment[idx]

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
        idy = np.unique(np.hstack((idy, idy2)))
        n_idx = np.where(nonword_pos - current_pos == 0)[0]
        n_idy = np.where(nonword_pos != 0)[0]

        semantic_idx = np.where(semantic)[0]
        words_idx = np.intersect1d(idx, idy)
        nonwords_idx = np.intersect1d(n_idx, n_idy)
        fix_idx = np.where(current_pos == 0)[0]
        primes_idx = np.where(current_pos == 1)[0]
        targets_idx = np.where(current_pos == 2)[0]

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

    if exp.startswith('OLDT'):
        # word vs nonword
        words = deepcopy(evts)
        words[words_idx, 2] = 2
        words[nonwords_idx, 2] = 1
        idx = np.hstack((nonwords_idx, words_idx))
        words = words[idx]

    # semantic priming condition
    priming = deepcopy(evts)
    primed_idx = np.intersect1d(semantic_idx, targets_idx)
    unprimed_idx = np.setdiff1d(targets_idx, primed_idx)
    priming[primed_idx, 2] = 4
    priming[unprimed_idx, 2] = 3
    idx = np.hstack((unprimed_idx, primed_idx))
    priming = priming[idx]

    # prime vs target
    pos = deepcopy(evts)
    pos[targets_idx, 2] = 6
    pos[primes_idx, 2] = 5
    idx = np.hstack((primes_idx, targets_idx))
    pos = pos[idx]

    # fixation
    fix = deepcopy(evts)
    fix[fix_idx, 2] = 99
    fix = fix[fix_idx]

    if exp.startswith('OLDT'):
        evts = np.vstack((words, priming, pos, fix, alignment))
    else:
        evts = np.vstack((priming, pos, fix, alignment))
    idx = zip(evts[:, 0], np.arange(evts.size))
    idx = list(zip(*sorted(idx))[-1])
    evts = evts[idx]
    path = op.dirname(raw.info['filename'])
    mne.write_events(op.join(path, '..', 'mne', '%s_%s-eve.txt')
                     % (subject, exp), evts)

    # key
    # 1: nonword, 2: word, 3: unprimed, 4: primed, 5: prime, 6: target, 50: alignment, 99: fixation
