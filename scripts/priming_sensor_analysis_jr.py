import sys
import os
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

import mne
from mne.report import Report
from mne.decoding import ConcatenateChannels, GeneralizationAcrossTime
from mne.stats.regression import linear_regression, linear_regression_raw

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score

import config


# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
analysis = 'priming_sensor_analysis'
random_state = 42
decim = 2
# decoding parameters
tstart, tstop = -.2, .7
# length = 25e-3  # smoothing window
# step = 5e-3
length = 10e-3 * decim
step = 10e-3 * decim

# setup group
fname_group = op.join(config.results_dir, 'group', 'group_OLDT_%s_filt_%s.html'
                      % (filt, analysis))
group_rep = Report()
group_scores = []
group_std_scores = []
group_auc_scores = []

for subject in config.subjects:
    print config.banner % subject
    # define filenames
    fname_rep = op.join(config.results_dir, subject,
                        '%s_%s_%s.html' % (subject, exp, analysis))
    fname_proj = op.join(path, subject, 'mne', '%s_%s_calm_%s_filt-proj.fif'
                         % (subject, exp, filt))
    fname_epo = op.join(path, subject, 'mne',
                        '%s_%s_calm_%s_filt-epo.fif'
                        % (subject, exp, filt))
    fname_raw = op.join(path, subject, 'mne',
                        '%s_%s_calm_%s_filt-raw.fif'
                        % (subject, exp, filt))
    fname_evt = op.join(path, subject, 'mne',
                        '%s_%s_priming_calm_%s_filt-epo.fif'
                        % (subject, exp, filt))
    rep = Report()

    # loading epochs
    epochs = mne.read_epochs(fname_epo, preload=False)['word/target']
    epochs.drop_bad_epochs(reject=config.reject)
    epochs.load_data().crop(tstart, tstop)
    epochs.decimate(decim)
    epochs.info['bads'] = config.bads[subject]

    # add/apply proj
    proj = mne.read_proj(fname_proj)
    epochs.add_proj(proj)
    epochs.apply_proj()

    # limit channels to good
    epochs.pick_types(meg=True, exclude='bads')

    # # currently disabled because of the HED
    # epochs.equalize_event_counts(['unprimed', 'primed'], copy=False)
    # plotting grand average
    p = epochs.average().plot(show=False)
    comment = ("This is a grand average over all the target epochs after "
               "equalizing the numbers in the priming condition.<br>"
               'unprimed: %d, and primed: %d, out of 96 possible events.'
               % (len(epochs['unprimed']), len(epochs['primed'])))
    rep.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                          'Summary', image_format=img, comments=comment)
    # compute/plot difference
    evoked = epochs['primed'].average() - epochs['unprimed'].average()
    p = evoked.plot(show=False)
    rep.add_figs_to_section(p, 'Difference Butterfly',
                          'Evoked Difference Comparison',
                          image_format=img)

    # set up a linear regression model
    design_matrix = np.ones((len(epochs), 2))
    lbl = LabelEncoder()
    # Convert the labels of the data to binary descriptors
    y = lbl.fit_transform(epochs.events[:,-1])
    design_matrix[:, -1] = y
    names = ['intercept', 'priming']
    # run regression
    # stats = linear_regression(epochs, design_matrix, names)
    # s = stats['priming'].mlog10_p_val
    # # plot p-values
    # interval = int(step * 1e3 / decim)   # plot every 5ms
    # times = evoked.times[::interval]
    # figs = list()
    # for time in times:
    #     figs.append(s.plot_topomap(time, vmin=0, vmax=3, unit='',
    #                                scale=1, cmap='Reds', show=False))
    #     plt.close()
    # rep.add_slider_to_section(figs, times, 'Regression Analysis (-log10 p-val)')
    # rep.save(fname_rep, open_browser=False, overwrite=True)

    # #rERF
    # raw = mne.io.read_raw_fif(fname_raw)
    # evts = mne.read_events(fname_evt)
    # rerf = linear_regression_raw(raw, evts, event_id, tmin=-.2, tmax=.6,
    #                              decim=5)

    print 'get ready for decoding ;)'
    train_times = {'start': tstart, 'stop': tstop,
                   'length': length, 'step': step}

    # time-resolved decoding using GAT
    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                                   train_times=train_times)
    gat.fit(epochs, y=y)
    gat.score(epochs, y=y)
    fig = gat.plot(title='Decoding Score on Semantic Priming: '
                   'Unprimed vs. Primed')
    rep.add_figs_to_section(fig, 'Decoding Score on Priming',
                          'Decoding', image_format=img)

    rep.save(fname_rep, open_browser=False, overwrite=True)
