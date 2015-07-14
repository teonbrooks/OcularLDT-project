import os.path as op
import numpy as np
import mne

path = '/Users/teon/Google Drive/E-MEG/data/A0129/mne/'
fname_epo = op.join(path, 'A0129_OLDT_xca_calm_iir_filt-epo.fif')
epochs = mne.read_epochs(fname_epo)
before = epochs.average().plot()

ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
ica.fit(epochs)
ics = ica.get_sources(epochs)

picks = range(len(ics.info['ch_names']))
power, itc = mne.time_frequency.tfr_multitaper(ics, np.arange(10) + 1.,
    (np.arange(10) + 1.) /2, picks=picks, return_itc=True)
itc.crop(-.1, .03, copy=False)

avg = itc.data.mean(axis=2).mean(1)
# descending rank
ranks = np.argsort(avg)[::-1]
avg = avg[ranks]
exclude = ranks[avg > .5]
ica.exclude = exclude
# idx = np.where((itc.data == itc.data.max(axis=0)).all(axis=2))[0]
# itc.plot(idx)
epochs_em = ica.apply(epochs, copy=True)
after = epochs_em.average().plot()
