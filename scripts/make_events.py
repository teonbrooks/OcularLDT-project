import mne
import numpy as np
import os.path as op


subject = 'A0149'
path = '/Users/teon/Google Drive/E-MEG/data/%s/mne/' % subject
raw_file = '%s_OLDT_calm_lp40-raw.fif' % subject
raw = mne.io.Raw(op.join(path, raw_file), verbose=False, preload=False)

evts = mne.find_stim_steps(raw)
exp = np.array([(x & 2 ** 5) >> 5 for x in evts[:, 2]], dtype=bool)
evts = evts[exp]
idx = np.where(evts[:, 1] == 0)[0]
idy = np.nonzero(evts[:, 2])[0]
idx = np.intersect1d(idx, idy)
evts = evts[idx]
triggers = evts[:, 2]

semantic = np.array([(x & 2 ** 4) >> 4 for x in triggers], dtype=bool)
nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2 for x in triggers])
current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0 for x in triggers])

idx = np.where(nonword_pos - current_pos != 0)[0]
idy = np.where(current_pos < nonword_pos)[0]
idy2 = np.where(nonword_pos == 0)[0]
idy = np.unique(np.hstack((idy, idy2)))

semantic_idx = np.where(semantic)[0]
words_idx = np.intersect1d(idx, idy)
nonwords_idx = np.where(nonword_pos - current_pos == 0)[0]
primes_idx = np.intersect1d(np.where(current_pos == 1)[0], words_idx)
targets_idx = np.intersect1d(np.where(current_pos == 2)[0], words_idx)

# word vs nonword
words = evts
words[words_idx, 2] = 2
words[nonwords_idx, 2] = 1
idx = np.hstack((nonwords_idx, words_idx))
words = words[idx]

# semantic priming condition
priming = evts
primed_idx = np.intersect1d(semantic_idx, targets_idx)
priming[primed_idx, 2] = 4
unprimed_idx = np.setdiff1d(targets_idx, primed_idx)
priming[unprimed_idx, 2] = 3
idx = np.hstack((unprimed_idx, primed_idx))
priming = priming[idx]

# prime vs target
pos = evts
pos[primes_idx, 2] = 5
pos[targets_idx, 2] = 6
idx = np.hstack((primes_idx, targets_idx))
pos = pos[idx]

evts = np.vstack((words, priming, pos))
idx = zip(evts[:, 0], np.arange(evts.size))
idx = list(zip(*sorted(idx))[-1])
evts = evts[idx]
mne.write_events(op.join(path, '%s_OLDT-eve.txt') % subject, evts)

# key
# 1: nonword, 2: word, 3: unprimed, 4: primed
