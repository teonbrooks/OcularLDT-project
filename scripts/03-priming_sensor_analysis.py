import os
import os.path as op
import warnings
import numpy as np
import matplotlib.pyplot as plt

import config
from make_events import make_events

import mne
from mne.report import Report
from mne.decoding import ConcatenateChannels

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, ShuffleSplit


prep = 'calm_lp40'
drive = 'nyu'
# drive = 'home'
exp = 'OLDT'
decim = 5
# smoothing window
win = 10


group_path = op.join(config.results_dir, 'group_OLDT_priming_sensor_analysis.html')
group_r = Report()


for subject in config.subjects:
    r_path = op.join(config.results_dir, subject, subject + '_OLDT_priming_sensor_analysis.html')
    r = Report()
    path = config.drives[drive]
    exps = config.subjects[subject]
    # load raw
    raw = config.kit2fiff(subject=subject, exp=exps[0],
                          path=path, preload=True)
    raw.filter(1,40)
    raw2 = config.kit2fiff(subject=subject, exp=exps[2],
                          path=path, preload=True)
    raw2.filter(1,40)
    # mne.transform_instances([raw, raw2])
    mne.concatenate_raws([raw, raw2])

    evt_file = op.join(path, '%s_%s-eve.txt' % (subject, 'OLDT'))
    if not op.exists(evt_file):
        make_events(raw, subject, 'OLDT')
    raw.info['bads'] = config.bads[subject]
    evts = mne.read_events(op.join(path, subject, 'mne',
                           '%s_%s-eve.txt' % (subject, 'OLDT')))
    event_id = {'unprimed': 3, 'primed': 4}
    # load epochs
    epochs = mne.Epochs(raw, evts, event_id, tmin=-.1, tmax=.6,
                        baseline=(None,0), reject=config.reject,
                        preload=True, decim=decim, verbose=False)
    epochs.equalize_event_counts(['unprimed', 'primed'], copy=False)
    del raw, raw2
    # plotting grand average
    p = epochs.average().plot(show=False)
    r.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
                          'Summary', image_format='png')
    # # compute/plot difference
    # evoked = epochs['primed'].average() - epochs['unprimed'].average()
    # p = evoked.plot(show=False)
    # r.add_figs_to_section(p, '%s: Difference Butterfly' % subject,
    #                       'Evoked Difference Comparison',
    #                       image_format='png')
    # p = evoked.plot_topomap(np.linspace(0, .25, 10), show=False)
    # r.add_figs_to_section(p, '%s: Difference Topomap 0-250 ms' % subject,
    #                       'Evoked Difference Comparison',
    #                       image_format='png')
    # p = evoked.plot_topomap(np.linspace(.25, .50, 10), show=False);
    # r.add_figs_to_section(p, '%s: Difference Topomap 250-500 ms' % subject,
    #                       'Evoked Difference Comparison',
    #                       image_format='png')

    # get ready for decoding ;)
    n_times = len(epochs.times) - win
    times = epochs.times[:-win]
    scores = np.empty(n_times, np.float32)
    std_scores = np.empty(n_times, np.float32)

    picks = mne.pick_types(epochs.info, meg=True, exclude='bads')

    lbl = LabelEncoder()
    scaler = StandardScaler()
    concat = ConcatenateChannels()
    svc = SVC(C=1, kernel='linear')
    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(len(epochs), 10, test_size=0.2)
    # Convert the labels of the data to binary descriptors
    y = lbl.fit_transform(epochs.events[:,-1])

    for t, tmin in enumerate(times):
        ep = epochs.crop(tmin, tmin+.05, copy=True)

        # epochs_list = [epochs[k] for k in ('unprimed', 'primed')]

        # Standardize features: mean-centered, normalized by std
        # Concatenate features, shape: (epochs, sensor * time window)
        # Run an SVM
        clf = Pipeline([('concat', concat), ('scaler', scaler), ('svm', svc)])
        Xt = ep.get_data()[:, picks, :]

        # Run cross-validation
        # Note: for sklearn the Xt matrix should be 2d (n_samples x n_features)
        scores_t = cross_val_score(clf, Xt, y, cv=cv, n_jobs=3)
        scores[t] = scores_t.mean()
        std_scores[t] = scores_t.std()

    scores *= 100  # make it percentage
    std_scores *= 100

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

    r.add_figs_to_section(fig, '%s: Decoding Score on Priming' % subject,
                          'Decoding', image_format='png')
    group_r.add_figs_to_section(fig, '%s: Decoding Score on Priming' % subject,
                          'Decoding', image_format='png')
    if not op.exists(op.dirname(r_path)):
        os.mkdir(op.dirname(r_path))
    r.save(r_path, open_browser=False, overwrite=True)
group_r.save(group_path, open_browser=False, overwrite=True)