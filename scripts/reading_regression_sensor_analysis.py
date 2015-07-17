import os
import sys
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import itertools

import mne
from mne.report import Report
from mne.decoding import ConcatenateChannels
from mne.stats import linear_regression, spatio_temporal_cluster_1samp_test

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import config


# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
analysis = 'reading_regression_sensor_analysis'
decim = 5
random_state = 42

win = 25e-3  # smoothing window in seconds
plt_interval = 5e-3  # plotting interval
reject = config.reject

# setup group
fname_group = op.join(config.results_dir, 'group', 'group_OLDT_%s_filt_%s.html'
                      % (filt, analysis))
group_rep = Report()
group_scores = []
group_std_scores = []
group_auc_scores = []


# Ranker
def rank_scorer(reg, X, y):
    y_pred = reg.predict(X)
    comb = itertools.combinations(range(len(y)), 2)
    score = 0.
    for k, pair in enumerate(comb):
        i, j = pair
        if y[i] == y[j]:
            continue
        score += np.sign((y[i] - y[j]) * (y_pred[i] - y_pred[j])) > 0.

    return score / float(k)


for subject in config.subjects:
    print config.banner % subject

    # define filenames
    fname_rep = op.join(config.results_dir, subject,
                        subject + '_%s_%s.html' % (exp, analysis))
    fname_proj = op.join(path, subject, 'mne',
                         subject + '_%s_calm_%s_filt-proj.fif' % (exp, filt))
    fname_epo = op.join(path, subject, 'mne',
                        subject + '_%s_coreg_calm_%s_filt-epo.fif' % (exp, filt))
    fname_dm = op.join(path, subject, 'mne',
                       subject + '_%s_design_matrix.txt' % exp)
    rep = Report()

    # loading design matrix, epochs, proj
    design_matrix = np.loadtxt(fname_dm)
    epochs = mne.read_epochs(fname_epo)
    epochs.decimate(decim)
    epochs.info['bads'] = config.bads[subject]
    epochs.pick_types(meg=True, exclude='bads')

    # add and apply proj
    proj = mne.read_proj(fname_proj)
    # proj = [proj[0]]
    epochs.add_proj(proj)
    epochs.apply_proj()

    # epochs rejection: filtering
    # drop based on MEG rejection, must happen first
    epochs.drop_bad_epochs(reject=reject)
    design_matrix = design_matrix[epochs.selection]
    # remove zeros
    idx = design_matrix[:, -1] > 0
    epochs = epochs[idx]
    design_matrix = design_matrix[idx]
    # and outliers
    durs = design_matrix[:, -1]
    mean, std = durs.mean(), durs.std()
    devs = np.abs(durs - mean)
    criterion = 3 * std
    idx = devs < criterion
    epochs = epochs[idx]
    design_matrix = design_matrix[idx]

    assert len(design_matrix) == len(epochs)

    # plotting grand average
    p = epochs.average().plot(show=False)
    comment = ("This is a grand average over all the target epochs after "
               "equalizing the numbers in the priming condition.<br>"
               'Number of epochs: %d.' % (len(epochs)))
    rep.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                            'Summary', image_format=img, comments=comment)

    names = ['intercept', 'fixation']
    stats = linear_regression(epochs, design_matrix, names)

    # run a spatio-temporal linear regression
    X = stats['fixation'].beta.data.swapaxes(1, 2)
    connectivity, ch_names = read_ch_connectivity('KIT-208')
    threshold = 2
    p_accept = 0.05
    cluster_stats = spatio_temporal_cluster_1samp_test(X, n_permutations=1000,
                        threshold=threshold, tail=0, connectivity=connectivity)
    asdf
    # s = stats[names[-1]].mlog10_p_val
    s = stats[names[-1]].t_val
    # plot p-values
    interval = int(plt_interval * 1e3 / decim)   # plot every 5ms
    times = epochs.times[::interval]
    figs = list()
    for time in times:
        # figs.append(s.plot_topomap(time, vmin=0, vmax=3, unit='',
        #                            scale=1, cmap='Reds', show=False))
        figs.append(s.plot_topomap(time, vmin=1, vmax=4, unit='',
                                   scale=1, cmap='Reds', show=False))
        plt.close()
    # rep.add_slider_to_section(figs, times, 'Regression Analysis (-log10 p-val)')
    rep.add_slider_to_section(figs, times, 'Regression Analysis (t-val)')
    rep.save(fname_rep, open_browser=False, overwrite=True)

    print "get ready for decoding ;)"
    # handle the window at the end
    first_samp = int(win * 1e3 / decim)
    last_samp = -first_samp
    times = epochs.times[first_samp:last_samp]
    n_times = len(times)

    scores = np.empty(n_times, np.float32)
    std_scores = np.empty(n_times, np.float32)

    # sklearn pipeline
    scaler = StandardScaler()
    concat = ConcatenateChannels()
    regression = Ridge(alpha=1e-3)  # Ridge Regression

    # Define 'y': what you're predicting
    y = design_matrix[:, -1]
    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=random_state)

    for t, time in enumerate(times):
        # add progress indicator
        progress = (t + 1.) * 100 / len(times)
        sys.stdout.write("\r%f%%" % progress)
        sys.stdout.flush()
        # smoothing window
        ep = epochs.crop(time - win, time + win, copy=True)
        # Pipeline:
        # Concatenate features, shape: (epochs, sensor * time window)
        # Standardize features: mean-centered, normalized by std
        # Run an Regression
        reg = Pipeline([('concat', concat), ('scaler', scaler),
                        ('regression', regression)])
        Xt = ep.get_data()

        # Run cross-validation
        scores_t = cross_val_score(reg, Xt, y, cv=cv, scoring=rank_scorer,
                                   n_jobs=1)
        scores[t] = scores_t.mean(axis=0)
        std_scores[t] = scores_t.std(axis=0)

    scores *= 100  # make it percentage
    std_scores *= 100
    group_scores.append(scores)
    group_std_scores.append(std_scores)

    # Regression Rank CV score
    plt.close('all')
    fig = plt.figure()
    plt.plot(times, scores, label="CV score")
    plt.axhline(50, color='k', linestyle='--', label="Chance level")
    plt.axvline(0, color='r', label='stim onset')
    plt.legend()
    hyp_limits = (scores - std_scores, scores + std_scores)
    plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
                     color='b', alpha=0.5)
    plt.xlabel('Times (ms)')
    plt.ylabel('CV classification score (% correct)')
    plt.ylim([30, 100])
    plt.title('Sensor space decoding using Rank Scorer')

    # decoding fig
    rep.add_figs_to_section(fig, 'Decoding Score on Priming',
                            'Decoding', image_format=img)
    group_rep.add_figs_to_section(fig, '%s: Decoding Score on Priming'
                                % subject, 'Subject Summary',
                                image_format=img)
    rep.save(fname_rep, open_browser=False, overwrite=True)

# group average classification score
first_samp = int(win * 1e3 / decim)
last_samp = -first_samp
times = epochs.times[first_samp:last_samp]
scores = np.array(group_scores).mean(axis=0)
std_scores = np.array(group_scores).std(axis=0)
plt.close('all')
fig = plt.figure()
plt.plot(times, scores, label="Classif. score")
plt.axhline(50, color='k', linestyle='--', label="Chance level")
plt.axvline(0, color='r', label='stim onset')
plt.legend()
hyp_limits = (scores - std_scores, scores + std_scores)
plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
                 color='b', alpha=0.5)
plt.xlabel('Times (ms)')
plt.ylabel('CV classification score (% correct)')
plt.ylim([30, 100])
plt.title('Group Average Sensor space decoding')
group_rep.add_figs_to_section(fig, 'Group Average Decoding Score on Priming',
                            'Group Summary', image_format=img)
group_rep.save(fname_group % exp, open_browser=False, overwrite=True)
