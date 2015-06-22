import os
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import itertools

import mne
from mne.report import Report
from mne.decoding import ConcatenateChannels
from mne.stats.regression import linear_regression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import config


# parameters
path = config.drive
filt = config.filt
exp = 'OLDT'
analysis = 'reading_regression_sensor_analysis'
decim = 4
random_state = 42
img = config.img
win = .050  # smoothing window in seconds
reject = config.reject
alphas = [.001, .01, .1, 1, 10, 100]

group_r = Report()

# define filenames
r_fname = op.join(config.results_dir, '%s', '%s_%s_' + '%s.html' % analysis)
group_fname = op.join(config.results_dir, 'group', 'group_%s_' + '%s.html'
                      % analysis)
proj_fname = op.join(path, '%s', 'mne', '%s_%s-proj.fif')
ep_fname = op.join(path, '%s', 'mne', '%s_%s_coreg_calm_%s_filt-epo.fif')
dm_fname = op.join(path, '%s', 'mne', '%s_%s_design_matrix.txt')


group_scores = []
group_std_scores = []

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

    subject_names = (subject, subject, exp)
    r = Report()

    # loading design matrix, epochs, proj
    design_matrix = np.loadtxt(dm_fname % subject_names)
    epochs = mne.read_epochs(ep_fname % (subject_names + (filt,)),
                             verbose=False)
    epochs.decimate(decim)
    epochs.info['bads'] = config.bads[subject]
    epochs.pick_types(meg=True, exclude='bads')

    # add and apply proj
    proj = mne.read_proj(proj_fname % subject_names)
    proj = [proj[0]]
    epochs.add_proj(proj)
    epochs.apply_proj()

    # epochs rejection: filtering
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
    # # drop based on MEG rejection
    # epochs.reject = reject
    # epochs.drop_bad_epochs()
    # idx = []
    # for msgs in epochs.drop_log:
    #     if not msgs:
    #         idx.append(True)
    #     else:
    #         idx.append(any([False if msg.startswith('MEG')
    #                         else True for msg in msgs]))
    # design_matrix = design_matrix[idx]
    assert len(design_matrix) == len(epochs)

    # plotting grand average
    p = epochs.average().plot(show=False)
    comment = ("This is a grand average over all the target epochs after "
               "equalizing the numbers in the priming condition.<br>"
               'Number of epochs: %d.' % (len(epochs)))
    r.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                          'Summary', image_format=img, comments=comment)

    # run a linear regression
    names = ['intercept', 'fixation']
    stats = linear_regression(epochs, design_matrix, names)
    s = stats[names[-1]].mlog10_p_val
    # plot t-values
    p = s.plot_topomap(np.linspace(0, .20, 10), unit='-log10 p-val',
                       scale=1, vmin=0, vmax=4, cmap='Reds', show=False)
    r.add_figs_to_section(p, '-log10 p-val Topomap 0-200 ms',
                          'Regression Analysis',
                          image_format=img)
    p = s.plot_topomap(np.linspace(.20, .40, 10), unit='-log10 p-val',
                       scale=1, vmin=0, vmax=4, cmap='Reds', show=False)
    r.add_figs_to_section(p, '-log10 p-val Topomap 200-400 ms',
                          'Regression Analysis',
                          image_format=img)
    p = s.plot_topomap(np.linspace(.40, .60, 10), unit='-log10 p-val',
                       scale=1, vmin=0, vmax=4, cmap='Reds', show=False)
    r.add_figs_to_section(p, '-log10 p-val Topomap 400-600 ms',
                          'Regression Analysis',
                          image_format=img)
    r.save(r_fname % subject_names, open_browser=False, overwrite=True)

    print "get ready for decoding ;)"
    for alpha in alphas:
        # handle the window at the end
        last_samp = int(win * 1e3 / decim)
        times = epochs.times[:-last_samp]
        n_times = len(times)

        scores = np.empty(n_times, np.float32)
        std_scores = np.empty(n_times, np.float32)

        # sklearn pipeline
        # scaler = StandardScaler()
        concat = ConcatenateChannels()
        # regression = Ridge(alpha=alpha)  # Ridge Regression
        regression = KernelRidge(kernel='rbf', gamma=alpha)  # KRR

        # Define 'y': what you're predicting
        y = design_matrix[:, -1]

        # Define a monte-carlo cross-validation generator (reduce variance):
        cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=random_state)

        for t, tmin in enumerate(times):
            # smoothing window
            ep = epochs.crop(tmin, tmin + win, copy=True)
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
        plt.plot(times, scores, label="Regression Rank CV score")
        plt.axhline(50, color='k', linestyle='--', label="Chance level")
        plt.axvline(0, color='r', label='stim onset')
        plt.legend()
        hyp_limits = (scores - std_scores, scores + std_scores)
        plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
                         color='b', alpha=0.5)
        plt.xlabel('Times (ms)')
        plt.ylabel('CV classification score (% correct)')
        plt.ylim([30, 100])
        plt.title('Sensor space decoding')

        # decoding fig
        r.add_figs_to_section(fig, 'alpha=%s: Decoding Score on Priming' % alpha,
                              'Decoding', image_format=img)
        group_r.add_figs_to_section(fig, '%s: Decoding Score on Priming'
                                    % subject, 'Subject Summary',
                                    image_format=img)
        r.save(r_fname % subject_names, open_browser=False, overwrite=True)

# # group average classification score
# group_scores = np.array(group_scores).mean(axis=0)
# group_std_scores = np.array(group_std_scores).mean(axis=0)
# plt.close('all')
# fig = plt.figure()
# plt.plot(times, group_scores, label="Classif. score")
# plt.axhline(50, color='k', linestyle='--', label="Chance level")
# plt.axvline(0, color='r', label='stim onset')
# plt.legend()
# hyp_limits = (group_scores - group_std_scores,
#               group_scores + group_std_scores)
# plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
#                  color='b', alpha=0.5)
# plt.xlabel('Times (ms)')
# plt.ylabel('CV classification score (% correct)')
# plt.ylim([30, 100])
# plt.title('Group Average Sensor space decoding')
# group_r.add_figs_to_section(fig, 'Group Average Decoding Score on Priming',
#                             'Group Summary', image_format=img)
# group_r.save(group_fname % exp, open_browser=False, overwrite=True)
