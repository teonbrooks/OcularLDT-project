import os.path as op
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec

import mne
from mne.report import Report
import config


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'
filt = config.filt
redo = True
reject = config.reject
decim = 4
baseline = (-.2, -.1)
tmin, tmax = -.5, 1
ylim = dict(mag=[-200, 200])
event_id = {'word/prime/unprimed': 1,
            'word/prime/primed': 5,
            'nonword/prime': 9,
           }
subject = 'A0129'
drive = 'google_drive'

# define filenames
path = op.join(drive, subject, 'mne')
fname_rep = op.join(config.results_dir, subject,
                    subject + '_%s_calm_%s_filt_ica-report.html'
                    % (exp, filt))
fname_raw = op.join(path, subject + '_%s_calm_%s_filt-raw.fif'
                    % (exp, filt))

raw = mne.io.read_raw_fif(fname_raw, preload=True)
raw.info['bads'] = config.bads[subject]
evts = mne.read_events(op.join(path, subject, 'mne', subject + '_OLDT-eve.txt'))

epochs = mne.Epochs(raw, evts, event_id, tmin=-.2, tmax=.6,
                    baseline=(None,-.1), reject=config.reject,
                    verbose=False, preload=False)

# p = epochs.average().plot()

# First ICA
ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
ica.fit(raw, decim=decim)

ics = ica.get_sources(epochs)



# # try with correlation
# ics = ica.get_sources(ep_ica.average())
# ics_e = ics.data
# ics_median = np.median(ics_e, axis=0)
# # ics_median = np.ones(ics_e.shape[-1])
# ics_scores = [sp.stats.pearsonr(ics_median, ic) for ic in ics_e]
# corr, pvals = zip(*ics_scores)
# gof = np.square(corr)
# idx = list(np.where(gof < .1)[0])
# # ica.plot_sources(ep_ica.average(), exclude=idx)
# ica.plot_sources(ep_ica.average(), picks=idx)
# ica.exclude = idx
# epochs_r = ica.apply(epochs, copy=True)
# epochs_r.average().plot()


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
