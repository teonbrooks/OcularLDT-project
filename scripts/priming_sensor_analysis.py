import os
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

import mne
from mne.report import Report
from mne.decoding import ConcatenateChannels
from mne.stats.regression import linear_regression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.metrics import roc_curve, auc

import config


# parameters
path = config.drive
exp = 'OLDT'
analysis = 'priming_sensor_analysis'
decim = 5
img = config.img
win = 10  # smoothing window
kernels = ['linear']
random_state = 42

group_r = Report()

# define filenames
group_fname = op.join(config.results_dir, 'group', 'group_OLDT_%s.html')
r_fname = op.join(config.results_dir, '%s', '%s_OLDT_%s.html')
proj_fname = op.join(path, '%s', 'mne', '%s_OLDT-proj.fif')
ep_fname = op.join(path, '%s', 'mne', '%s_OLDT_priming_calm_filt-epo.fif')


group_scores = []
group_std_scores = []
for subject in config.subjects:
    r = Report()

    # loading epochs, proj
    epochs = mne.read_epochs(ep_fname % (subject, subject))
    epochs.info['bads'] = config.bads[subject]
    epochs.pick_types(meg=True, exclude='bads')

    # temporary hack
    epochs._raw_times = epochs.times
    epochs._offset = None
    epochs.detrend = None
    epochs.decim = None

    # back to coding
    proj = mne.read_proj(proj_fname % (subject, subject))
    proj = [proj[0]]
    epochs.add_proj(proj)
    epochs.apply_proj()

    epochs.equalize_event_counts(['unprimed', 'primed'], copy=False)
    # plotting grand average
    p = epochs.average().plot(show=False)
    comment = ("This is a grand average over all the target epochs after"
               "equalizing the numbers in the priming condition.<br>"
               'unprimed: %d, and primed: %d, out of 96 possible events.'
               % (len(epochs['unprimed']), len(epochs['primed'])))
    r.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                          'Summary', image_format=img, comments=comment)
    # compute/plot difference
    evoked = epochs['primed'].average() - epochs['unprimed'].average()
    evoked.pick_types()
    p = evoked.plot(show=False)
    r.add_figs_to_section(p, 'Difference Butterfly',
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
    r.save(r_fname % (subject, subject, analysis), open_browser=False, overwrite=True)

    # get ready for decoding ;)
    for kernel in kernels:
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
        # linear SVM 
        svc = SVC(kernel=kernel, probability=True, random_state=random_state)
        # Define a monte-carlo cross-validation generator (reduce variance):
        # cv = ShuffleSplit(len(epochs), 10, test_size=0.2)
        cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=random_state)

        for t, tmin in enumerate(times):
            # smoothing window
            ep = epochs.crop(tmin, tmin + (decim * win * 1e-3), copy=True)
            # Standardize features: mean-centered, normalized by std
            # Concatenate features, shape: (epochs, sensor * time window)
            # Run an SVM
            clf = Pipeline([('concat', concat), ('scaler', scaler), ('svm', svc)])
            # decimate
            ep_len = ep.get_data().shape[-1]
            idx = slice(0, ep_len, decim)
            Xt = ep.get_data()[:, :, idx]

            # Run cross-validation
            # Note: for sklearn the Xt matrix should be 2d (n_samples x n_features)
            scores_t = cross_val_score(clf, Xt, y, cv=cv, n_jobs=-1)
            scores[t] = scores_t.mean()
            std_scores[t] = scores_t.std()
            # Run ROC/AUC calculation
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            all_tpr = []

            for i, (train, test) in enumerate(cv):
                probas_ = clf.fit(Xt[train], y[train]).predict_proba(Xt[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
            mean_tpr /= len(cv)
            mean_tpr[-1] = 1.0
            auc_scores[t] = auc(mean_fpr, mean_tpr)

        scores *= 100  # make it percentage
        std_scores *= 100
        auc_scores *= 100

        # for group average
        group_scores.append(scores)
        group_std_scores.append(std_scores)

        # CV classification score
        plt.close('all')
        fig = plt.figure()
        # plt.plot(times, scores, label="Classif. score")
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
        # AUC score
        plt.close('all')
        auc_fig = plt.figure()
        plt.plot(times, auc_scores, label="Classif. score")
        plt.axhline(50, color='k', linestyle='--', label="Chance level")
        plt.axvline(0, color='r', label='stim onset')
        plt.xlabel('Times (ms)')
        plt.ylabel('AUC')
        plt.ylim([30, 100])
        plt.title('Sensor space Area Under ROC')


        # decoding fig
        r.add_figs_to_section(fig, 'Decoding Score on Priming',
                              'Decoding', image_format=img)
        group_r.add_figs_to_section(fig, '%s: Decoding Score on Priming'
                                    % subject, 'Subject Summary',
                                    image_format=img)
        # auc fig
        r.add_figs_to_section(auc_fig, 'AUC Score on Priming',
                              'Subject Summary', image_format=img)
        group_r.add_figs_to_section(auc_fig, '%s: AUC Score on Priming'
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
group_r.add_figs_to_section(fig, 'Group Average Decoding Score on Priming',
                            'Group Summary', image_format=img)

group_r.save(group_fname % analysis, open_browser=False, overwrite=True)
