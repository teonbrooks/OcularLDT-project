import sys
import os
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

import mne
from mne.report import Report
from mne.decoding import ConcatenateChannels
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
decim = 5
win = 25e-3  # smoothing window
plt_interval = 5e-3  # plotting interval
random_state = 42


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
                        '%s_%s_calm_%s_filt-raw.fif' % (subject, exp, filt))
    fname_evt = op.join(path, subject, 'mne',
                        subject + '_%s_priming_calm_%s_filt-epo.fif'
                        % (exp, filt))
    event_id = {'unprimed': 3, 'primed': 4}
    rep = Report()

    # loading epochs
    epochs = mne.read_epochs(fname_epo)
    epochs.crop(-.1, .7)
    epochs.drop_bad_epochs(reject=config.reject)
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

    # run a linear regression
    design_matrix = np.ones((len(epochs), 2))
    lbl = LabelEncoder()
    # Convert the labels of the data to binary descriptors
    y = lbl.fit_transform(epochs.events[:,-1])
    design_matrix[:, -1] = y
    names = ['intercept', 'priming']
    stats = linear_regression(epochs, design_matrix, names)
    s = stats['priming'].mlog10_p_val
    # plot p-values
    interval = int(plt_interval * 1e3 / decim)   # plot every 5ms
    times = evoked.times[::interval]
    figs = list()
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
    # time-resolve decoding
    # handle the window at the end
    first_samp = int(win * 1e3 / decim)
    last_samp = -first_samp
    times = epochs.times[first_samp:last_samp]
    n_times = len(times)

    scores = np.empty(n_times, np.float32)
    std_scores = np.empty(n_times, np.float32)
    auc_scores = np.empty(n_times, np.float32)

    # sklearn pipeline
    scaler = StandardScaler()
    concat = ConcatenateChannels()
    # linear SVM
    svc = SVC(kernel='linear', probability=True, random_state=random_state)
    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=random_state)

    for t, tmin in enumerate(times):
        # add progress indicator
        progress = (t + 1.) * 100 / len(times)
        sys.stdout.write("\r%f%%" % progress)
        sys.stdout.flush()
        # smoothing window
        ep = epochs.crop(tmin - win, tmin + win, copy=True)
        # Concatenate features, shape: (epochs, sensor * time window)
        # Standardize features: mean-centered, normalized by std for each feature
        # Run an SVM
        clf = Pipeline([('concat', concat), ('scaler', scaler), ('svm', svc)])
        Xt = ep.get_data()

        # Run cross-validation
        # Note: for sklearn the Xt matrix should be 2d (n_samples x n_features)
        scores_t = cross_val_score(clf, Xt, y, cv=cv, n_jobs=-1)
        scores[t] = scores_t.mean()
        std_scores[t] = scores_t.std()
        # # Run ROC/AUC calculation
        # auc_scores_t = []
        #
        # for i, (train, test) in enumerate(cv):
        #     probas_ = clf.fit(Xt[train], y[train]).predict_proba(Xt[test])
        #     auc_scores_t.append(roc_auc_score(y[test], probas_[:, 1]))
        # auc_scores[t] = np.array(auc_scores_t).mean()

    scores *= 100  # make it percentage
    std_scores *= 100
    # auc_scores *= 100

    # for group average
    group_scores.append(scores)
    # group_auc_scores.append(auc_scores)

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
    plt.ylim([30, 80])
    plt.title('Sensor space decoding')
    # # AUC score
    # plt.close('all')
    # auc_fig = plt.figure()
    # plt.plot(times, auc_scores, label="Classif. score")
    # plt.axhline(50, color='k', linestyle='--', label="Chance level")
    # plt.axvline(0, color='r', label='stim onset')
    # plt.xlabel('Times (ms)')
    # plt.ylabel('AUC')
    # plt.ylim([30, 100])
    # plt.title('Sensor space Area Under ROC')

    # decoding fig
    rep.add_figs_to_section(fig, 'Decoding Score on Priming',
                          'Decoding', image_format=img)
    group_rep.add_figs_to_section(fig, '%s: Decoding Score on Priming'
                                % subject, 'Subject Summary',
                                image_format=img)
    # # auc fig
    # rep.add_figs_to_section(auc_fig, 'AUC Score on Priming',
    #                       'Subject Summary', image_format=img)
    # group_rep.add_figs_to_section(auc_fig, '%s: AUC Score on Priming'
    #                             % subject, 'Subject Summary',
    #                             image_format=img)
    rep.save(fname_rep, open_browser=False, overwrite=True)

# group average classification score
group_scores = np.array(group_scores).mean(axis=0)
group_std_scores = np.array(group_scores).std(axis=0)
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
plt.ylim([30, 80])
plt.title('Group Average Sensor space decoding')
group_rep.add_figs_to_section(fig, 'Group Average Decoding Score on Priming',
                            'Group Summary', image_format=img)

# # group average AUC score
# group_auc_scores = np.asarray(group_auc_scores).mean(axis=0)
# group_std_auc_scores = np.asarray(group_auc_scores).std(axis=0)
# plt.close('all')
# fig = plt.figure()
# plt.plot(times, group_auc_scores, label="Area Under Curve")
# plt.axhline(50, color='k', linestyle='--', label="Chance level")
# plt.axvline(0, color='r', label='stim onset')
# plt.legend()
# hyp_limits = (group_auc_scores - group_std_auc_scores,
#               group_auc_scores + group_std_auc_scores)
# plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1],
#                  color='b', alpha=0.5)
# plt.xlabel('Times (ms)')
# plt.ylabel('AUC')
# plt.ylim([30, 100])
# plt.title('Group Average Sensor space AUC')
# group_rep.add_figs_to_section(fig, 'Group Average AUC on Priming',
#                             'Group Summary', image_format=img)

group_rep.save(fname_group, open_browser=False, overwrite=True)
