import mne
import numpy as np
import os.path as op
from copy import deepcopy
import argparse
import sys
import config


path = config.data_dir
parser = argparse.ArgumentParser(description='Make Events.')
parser.add_argument('--subject', dest='subject', help='Subject Name',
                  metavar='str')
parser.add_argument('--exp', dest='expt', help='Experiment Name',
                  metavar='str')
parser.add_argument('--datadir', dest='datadir', help='Data Director Name',
                  metavar='str', default=path)

options = parser.parse_args()
subject = options.subject
expt = options.expt

make_OLDT_events(subject, expt)
    if not expt.startswith('OLDT'):
        raise ValueError('This script only works for'
                         'OLDTX experiments, not %s.' % expt)
    if subject is None:
        parser.print_help()
        sys.exit(1)
    datadir = options.datadir

    datadir = op.join(datadir, subject, 'mne')
    raw_file = '%s_%s_calm_lp40-raw.fif' % (subject, expt)
    raw = mne.io.Raw(op.join(datadir, raw_file), verbose=False, preload=False)

    evts = mne.find_stim_steps(raw)
    exp = np.array([(x & 2 ** 5) >> 5 for x in evts[:, 2]], dtype=bool)
    evts = evts[exp]
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
    primes_idx = np.intersect1d(np.where(current_pos == 1)[0], words_idx)
    targets_idx = np.intersect1d(np.where(current_pos == 2)[0], words_idx)


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

    evts = np.vstack((words, priming, pos, fix))
    idx = zip(evts[:, 0], np.arange(evts.size))
    idx = list(zip(*sorted(idx))[-1])
    evts = evts[idx]
    mne.write_events(op.join(datadir, '%s_%s-eve.txt') % (subject, expt), evts)

    # key
    # 1: nonword, 2: word, 3: unprimed, 4: primed, 5: prime, 6: target, 99: fixation


make_SENT_events(subject, expt)
    if not expt.startswith('SENT'):
        raise ValueError('This script only works for'
                         'SENTX experiments, not %s.' % expt)
    if subject is None:
        parser.print_help()
        sys.exit(1)
    datadir = options.datadir

    datadir = op.join(datadir, subject, 'mne')
    raw_file = '%s_%s_calm_lp40-raw.fif' % (subject, expt)
    raw = mne.io.Raw(op.join(datadir, raw_file), verbose=False, preload=False)

    evts = mne.find_stim_steps(raw)
    exp = np.array([(x & 2 ** 5) >> 5 for x in evts[:, 2]], dtype=bool)
    evts = evts[exp]
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
    primes_idx = np.intersect1d(np.where(current_pos == 1)[0], words_idx)
    targets_idx = np.intersect1d(np.where(current_pos == 2)[0], words_idx)


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

    evts = np.vstack((words, priming, pos, fix))
    idx = zip(evts[:, 0], np.arange(evts.size))
    idx = list(zip(*sorted(idx))[-1])
    evts = evts[idx]
    mne.write_events(op.join(datadir, '%s_%s-eve.txt') % (subject, expt), evts)

    # key
    # 1: nonword, 2: word, 3: unprimed, 4: primed, 5: prime, 6: target, 99: fixation