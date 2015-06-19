import os.path as op

import mne
import config


subject = 'A0129'
drive = 'google_drive'
exps = [config.subjects[subject][0], config.subjects[subject][2]]
path = config.drives[drive]

raw = config.kit2fiff(subject, exps[0], config.drives[drive])
raw2 = config.kit2fiff(subject, exps[1], config.drives[drive])
raw.append(raw2)
raw.info['bads'] = ['MEG 130']
raw.preload_data()
raw.filter(.1, 40, l_trans_bandwidth=.05, method='fft')
raw.info['bads'] = config.bads[subject]
evts = mne.read_events(op.join(config.drives[drive], subject, 'mne',
                       subject + '_OLDT_coreg-eve.txt'))

# key
# 1: nonword, 2: word, 3: unprimed, 4: primed
# 5: prime, 6: target, 50: alignment, 99: fixation

epochs = mne.Epochs(raw, evts, None, tmin=-.2, tmax=1.6, preload=True)

dm_fname = op.join(path, subject, 'mne', '%s_OLDT_design_matrix.txt' % subject)
design_matrix = np.load_txt(dm_fname)

idx = np.where(design_matrix[:, -1] != -1)[0]
design_matrix = design_matrix[idx]
epochs.drop_epochs(idx)

epochs.drop_channels(epochs.info['bads'])
epochs.plot(fixs=design_matrix[:, -1], block=True, trellis=False, n_epochs=2)