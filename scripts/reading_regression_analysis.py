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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import config


# parameters
path = config.drive
exp = 'OLDT'
analysis = 'reading_regression_analysis'
decim = 5
random_state = 42
img = config.img
win = 10  # smoothing window

group_r = Report()

# define filenames
group_fname = op.join(config.results_dir, 'group', 'group_OLDT_%s.html'
                      % analysis)
r_fname = op.join(config.results_dir, '%s', '%s_OLDT_%s.html')
proj_fname = op.join(path, '%s', 'mne', '%s_OLDT-proj.fif')
ep_fname = op.join(path, '%s', 'mne', '%s_OLDT_coreg_calm_filt-epo.fif')
dm_fname = op.join(path, '%s', 'mne', '%s_OLDT_design_matrix.txt')


group_scores = []
group_std_scores = []

# Ranker
def rank_scorer(clf, X, y):
    y_pred = clf.predict(X)
    comb = itertools.combinations(range(len(y)), 2)
    k = 0
    score = 0.
    for i, j in comb:
        if y[i] == y[j]:
            continue
        score += np.sign((y[i] - y[j]) * (y_pred[i] - y_pred[j])) > 0.
        k += 1

    return score / float(k)


for subject in config.subjects:
    r = Report()

    # loading design matrix, epochs, proj
    design_matrix = np.loadtxt(dm_fname % (subject, subject))
    epochs = mne.read_epochs(ep_fname % (subject, subject))
    epochs.info['bads'] = config.bads[subject]
    epochs.pick_types(meg=True, exclude='bads')
    proj = mne.read_proj(proj_fname % (subject, subject))
    proj = [proj[0]]

    # temporary hack
    epochs._raw_times = epochs.times
    epochs._offset = None
    epochs.detrend = None
    epochs.decim = None

    # back to coding, apply proj
    epochs.add_proj(proj)
    epochs.apply_proj()

    # filtering
    # remove zeros
    idx = np.where(design_matrix[:, -1] == -1)[0]  # drop bads
    epochs.drop_epochs(idx)
    design_matrix = design_matrix[design_matrix[:, -1] != -1]  # save goods
    # and outliers
    durs = design_matrix[:, -1]
    mean = durs.mean()
    std = durs.std()
    devs = np.abs(durs - mean)
    criterion = 3 * std
    idx = np.where(devs > criterion)[0]  # drop bads
    epochs.drop_epochs(idx)
    design_matrix = design_matrix[devs < criterion]  # save goods
    assert len(design_matrix) == len(epochs)

    # plotting grand average
    p = epochs.average().plot(show=False)
    comment = ("This is a grand average over all the target epochs after "
               "equalizing the numbers in the priming condition.<br>"
               'Number of epochs: %d.' % (len(epochs)))
    r.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                          'Group Summary', image_format=img, comments=comment)

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
    r.save(r_fname % subject, open_browser=False, overwrite=True)

    print "get ready for decoding ;)"
    # handle the window at the end
    times = epochs.times[:-win]
    # handle downsampling
    times = times[::decim]
    n_times = len(times)

    scores = np.empty(n_times, np.float32)
    std_scores = np.empty(n_times, np.float32)
    auc_scores = np.empty(n_times, np.float32)

    # sklearn pipeline
    scaler = StandardScaler()
    concat = ConcatenateChannels()
    regression = Ridge(alpha=1)  # Ridge Regression

    # Define 'y': what you're predicting
    y = design_matrix[:, -1]

    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=random_state)

    for t, tmin in enumerate(times):
        # smoothing window
        ep = epochs.crop(tmin, tmin + (decim * win * 1e-3), copy=True)
        # Pipeline:
        # Standardize features: mean-centered, normalized by std
        # Concatenate features, shape: (epochs, sensor * time window)
        # Run an Regression
        clf = Pipeline([('concat', concat), ('scaler', scaler),
                        ('regression', regression)])
        # decimate data
        ep_len = ep.get_data().shape[-1]
        idx = slice(0, ep_len, decim)
        Xt = ep.get_data()[:, :, idx]

        # Run cross-validation
        scores_t = cross_val_score(clf, Xt, y, cv=cv, scoring=rank_scorer,
                                   n_jobs=1)
        scores[t] = scores_t.mean()
        std_scores[t] = scores_t.std()

    scores *= 100  # make it percentage
    std_scores *= 100
    group_scores.append(scores)
    group_std_scores.append(std_scores)

    # CV classification score
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
    plt.title('Sensor space decoding')

    # decoding fig
    r.add_figs_to_section(fig, 'Decoding Score on Priming', 'Decoding',
                          image_format=img)
    group_r.add_figs_to_section(fig, '%s: %s Decoding Score on Priming'
                                % subject, 'Subject Summary',
                                image_format=img)
    if not op.exists(op.dirname(r_fname)):
        os.mkdir(op.dirname(r_fname))
    r.save(r_fname % (subject, subject, analysis), open_browser=False, overwrite=True)

# group average classification score
group_scores = np.array(group_scores).mean(axis=0)
group_std_scores = np.array(group_std_scores).mean(axis=0)
plt.close('all')
fig = plt.figure()
plt.plot(times, group_scores, label="Classif. score")
plt.axhline(50, color='k', linestyle='--', label="Chance level")
plt.axvline(0, color='r', label='stim onset')
plt.legend()
hyp_limits = (group_scores - group_std_scores,
              group_scores + group_std_scores)
plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
                 color='b', alpha=0.5)
plt.xlabel('Times (ms)')
plt.ylabel('CV classification score (% correct)')
plt.ylim([30, 100])
plt.title('Group Average Sensor space decoding')
group_r.add_figs_to_section(fig, 'Decoding Score on Priming', 'Group Summary',
                            image_format=img)
group_r.save(group_fname, open_browser=False, overwrite=True)
