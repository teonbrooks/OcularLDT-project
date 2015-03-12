import mne
import numpy as np
import os.path as op


path = '/Volumes/teon-backup/Experiments/E-MEG/data/A0148/mne/'
raw_file = 'A0148_OLDT_calm_lp40-raw.fif'
raw = mne.io.Raw(op.join(path, raw_file), verbose=False, preload=False)

evts = mne.find_stim_steps(raw)
exp_idx = np.array([(x & 2 ** 5) >> 5 for x in evts[:, 2]], dtype=bool)
evts = evts[exp_idx]
idx = np.where(evts[:,1] == 0)[0]
idy = np.nonzero(evts[:, 2])[0]
idx = np.intersect1d(idx, idy)
evts = evts[idx]
triggers = evts[:,2]


priming_idx = np.array([(x & 2 ** 4) >> 4 for x in triggers], dtype=bool)
nonword_pos = np.array([(x & (2 ** 3 + 2 ** 2)) >> 2 for x in triggers])
current_pos = np.array([(x & (2 ** 1 + 2 ** 0)) >> 0 for x in triggers])
words_idx = np.where(nonword_pos - current_pos != 0)[0]
nonwords_idx = np.where(nonword_pos - current_pos == 0)[0]
primes_idx = np.intersect1d(np.where(current_pos == 1)[0], words_idx)
targets_idx = np.intersect1d(np.where(current_pos == 2)[0], words_idx)

# word vs nonword
words = evts
words[words_idx,2] = 2
words[nonwords_idx, 2] = 1
idx = np.hstack((nonwords_idx, words_idx))
words = words[idx]

# semantic priming condition
priming = evts
primed_idx = np.intersect1d(np.where(priming_idx)[0], targets_idx)
priming[primed_idx,2] = 4
unprimed_idx = np.setdiff1d(targets_idx, primed_idx)
priming[unprimed_idx,2] = 3
idx = np.hstack((unprimed_idx, primed_idx))
priming = priming[idx]
evts = np.vstack((words, priming))
idx = zip(evts[:,0], np.arange(evts.size))
idx = list(zip(*sorted(idx))[-1])
evts = evts[idx]
mne.write_events(op.join(path, 'A0148_OLDT-eve.txt'), evts)

# key
# 1: nonword, 2: word, 3: unprimed, 4: primed
