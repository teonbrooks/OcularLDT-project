import scipy as sp
import numpy as np
import mne
import config
import os.path as op


subject = 'A0129'
drive = 'google_drive'
exps = [config.subjects[subject][0], config.subjects[subject][2]]
path = config.drives[drive]
ep_fname = '%s_OLDT_ica_calm_filt_full-epo.fif' % subject
ep_fname = op.join(path, subject, 'mne', ep_fname)
if op.exists(ep_fname):
    epochs = mne.read_epochs(ep_fname)
    epochs = epochs['prime']
else:
    raw = config.kit2fiff(subject, exps[0], config.drives[drive])
    raw2 = config.kit2fiff(subject, exps[1], config.drives[drive])
    raw.append(raw2)
    raw.preload_data()
    raw.filter(.1, 40, l_trans_bandwidth=.05, method='fft')
    raw.info['bads'] = config.bads[subject]
    evts = mne.read_events(op.join(config.drives[drive], subject, 'mne',
                           subject + '_OLDT-eve.txt'))

    # key
    # 1: nonword, 2: word, 3: unprimed, 4: primed
    # 5: prime, 6: target, 50: alignment, 99: fixation
    epochs = mne.Epochs(raw, evts, 5, tmin=-.2, tmax=.6,
                        baseline=(None,0), reject=config.reject,
                        verbose=False, preload=True)
    epochs.save(ep_fname)
    del raw
p = epochs.average().plot()

# First ICA
ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
ep_ica = epochs.crop(-.1, .1, copy=True)
ica.fit(ep_ica)


# try with correlation
ics = ica.get_sources(ep_ica.average())
ics_e = ics.data
ics_median = np.median(ics_e, axis=0)
# ics_median = np.ones(ics_e.shape[-1])
ics_scores = [sp.stats.pearsonr(ics_median, ic) for ic in ics_e]
corr, pvals = zip(*ics_scores)
gof = np.square(corr)
idx = list(np.where(gof < .1)[0])
# ica.plot_sources(ep_ica.average(), exclude=idx)
ica.plot_sources(ep_ica.average(), picks=idx)
ica.exclude = idx
epochs_r = ica.apply(epochs, copy=True)
epochs_r.average().plot()


#
#
# # Recursive ICA step
# ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
# ep_r_ica = epochs_r.crop(-.1, .1, copy=True)
# ica.fit(ep_r_ica)
#
# ica.exclude = [4, 8]
# epochs_rr = ica.apply(epochs_r, copy=True)
#
# # epochs.drop_channels(['MEG 130'])
# # ica.plot_sources(epochs.average(), picks=range(5), start=-.1, stop=.1)
# # ica.plot_source()
# # ica.exclude.append(2)
# # ica.save(op.join(config.drives[drive], subject, 'mne',
# #                  subject + '_OLDT-ica.fif'))
# # ica.apply(epochs)
# #
# #
# # p = epochs.average().plot()
