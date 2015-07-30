import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt

path = '/Users/teon/Google Drive/E-MEG/data/A0129/mne/'
fname_raw = op.join(path, 'A0129_OLDT_calm_iir_filt-raw.fif')
raw = mne.io.read_raw_fif(fname_raw)
fname_epo = op.join(path, 'A0129_OLDT_xca_calm_iir_filt-epo.fif')
epochs = mne.read_epochs(fname_epo)
before = epochs.average()
p_before = before.plot()

ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
ica.fit(epochs)
ics = ica.get_sources(epochs)

picks = range(len(ics.info['ch_names']))
power, itc = mne.time_frequency.tfr_multitaper(ics, np.arange(10) + 1.,
    (np.arange(10) + 1.) /2, picks=picks, return_itc=True)
# itc.crop(-.1, .03, copy=False)

nchan = itc.info['nchan']
columns = 5  # match topo plots
rows = np.ceil(float(nchan) / columns).astype(int)
fig, axes = plt.subplots(rows, columns, figsize=(20, 20))

axes = [ax for axe in axes for ax in axe]
for ii in range(nchan - (columns*rows)):
    axes[-(ii + 1)].axis('off')
itc.plot(picks=picks, axes=axes[:itc.info['nchan']])

fig.savefig('/Applications/packages/E-MEG/output/images/ica-tf.pdf')
ica.exclude = [2, 30]
ica_bads = ica.plot_components(ica.exclude)
ica_bads.savefig('/Applications/packages/E-MEG/output/images/ica-bads.pdf')
after = ica.apply(before, copy=True)
p_after = after.plot()
p_after.savefig('/Applications/packages/E-MEG/output/images/ica-after.pdf')
p_before.savefig('/Applications/packages/E-MEG/output/images/ica-before.pdf')

# select high ITC pre-onset, but low afterwards

# avg = itc.data.mean(axis=2).mean(1)
# # descending rank
# ranks = np.argsort(avg)[::-1]
# avg = avg[ranks]
# exclude = ranks[avg > .5]
# ica.exclude = exclude
# # idx = np.where((itc.data == itc.data.max(axis=0)).all(axis=2))[0]
# # itc.plot(idx)
# epochs_em = ica.apply(epochs, copy=True)
# after = epochs_em.average().plot()
