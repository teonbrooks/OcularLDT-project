import sys
import os
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

import mne
from mne.report import Report
from mne.decoding import GeneralizationAcrossTime
from mne.stats.regression import linear_regression, linear_regression_raw

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import config


# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
analysis = 'priming_sensor_analysis'
random_state = 42
decim = 5
# decoding parameters
tstart, tstop = -.2, .7
# smoothing window
# length = 5. * decim * 1e-3
# step = 5. * decim * 1e-3
length = decim * 1e-3
step = decim * 1e-3


# setup group
fname_group = op.join(config.results_dir, 'group', 'group_OLDT_%s_filt_%s.html'
                      % (filt, analysis))
group_rep = Report()
group_scores = []
group_reg = []

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
    # fname_raw = op.join(path, subject, 'mne',
    #                     '%s_%s_calm_%s_filt-raw.fif'
    #                     % (subject, exp, filt))
    fname_evt = op.join(path, subject, 'mne',
                        '%s_%s_priming_calm_%s_filt-epo.fif'
                        % (subject, exp, filt))
    rep = Report()

    # loading epochs
    epochs = mne.read_epochs(fname_epo, preload=False)['word/target']
    # add/apply proj
    proj = mne.read_proj(fname_proj)
    epochs.add_proj(proj)
    epochs.apply_proj()

    # select window of interest
    epochs.crop(tstart, tstop)
    epochs.decimate(decim)

    # limit channels to good
    epochs.info['bads'] = config.bads[subject]
    epochs.pick_types(meg=True, exclude='bads')
    epochs.drop_bad_epochs(reject=config.reject)

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
    ep = epochs.copy()
    ep.apply_baseline((-.2, .1))
    evoked = ep['primed'].average() - ep['unprimed'].average()
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
    stats = linear_regression(epochs, design_matrix, names)
    group_reg.append(stats['priming'].t_val)
    s = stats['priming'].mlog10_p_val
    # plot p-values
    figs = list()
    times = evoked.times
    for time in times:
        figs.append(s.plot_topomap(time, vmin=0, vmax=3, unit='',
                                   scale=1, cmap='Reds', show=False))
        plt.close()
    rep.add_slider_to_section(figs, times, 'Uncorrected Regression Analysis '
                              '(-log10 p-val)')
    rep.save(fname_rep, open_browser=False, overwrite=True)

    # #rERF
    # raw = mne.io.read_raw_fif(fname_raw)
    # evts = mne.read_events(fname_evt)
    # rerf = linear_regression_raw(raw, evts, event_id, tmin=-.2, tmax=.6,
    #                              decim=5)

    print 'get ready for decoding ;)'
    train_times = {'start': tstart,
                   'stop': tstop,
                   'length': length,
                   'step': step
                   }

    # Generalization Across Time
    # # default GAT
    # clf = LogisticRegression()
    # cv = KFold(n_folds=5)
    clf = SVC(kernel='linear', probability=False, random_state=random_state)
    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=random_state)

    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                                   train_times=train_times, clf=clf, cv=cv)
    gat.fit(epochs, y=y)
    gat.score(epochs, y=y)
    fig = gat.plot(title='GAT Decoding Score on Semantic Priming: '
                   'Unprimed vs. Primed')
    rep.add_figs_to_section(fig, 'GAT Decoding Score on Priming',
                          'Decoding', image_format=img)
    fig = gat.plot_diagonal(title='Time Decoding Score on Semantic Priming: '
                            'Unprimed vs. Primed')
    rep.add_figs_to_section(fig, 'Time Decoding Score on Priming',
                          'Decoding', image_format=img)

    rep.save(fname_rep, open_browser=False, overwrite=True)
    group_scores.append(gat.scores_)

# temp hack
group_gat = gat
group_gat.scores_ = np.mean(group_scores, axis=0)

fig = gat.plot(title='GAT Decoding Score on Semantic Priming: '
               'Unprimed vs. Primed')
group_rep.add_figs_to_section(fig, 'GAT Decoding Score on Priming',
                              'Decoding', image_format=img)
fig = gat.plot_diagonal(title='Time Decoding Score on Semantic Priming: '
                        'Unprimed vs. Primed')
group_rep.add_figs_to_section(fig, 'Time Decoding Score on Priming',
                              'Decoding', image_format=img)
group_rep.save(fname_group, open_browser=False, overwrite=True)
