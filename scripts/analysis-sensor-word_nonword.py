import pickle
import os.path as op
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import mne
from mne.decoding import GeneralizationAcrossTime
from mne.stats import linear_regression_raw
from mne.channels import read_ch_connectivity

import config
from analysis_func import group_stats

# parameters
redo = True
path = config.drive
filt = config.filt
img = config.img
exp = config.exp
clf_name = 'logit'
analysis = 'word-nonword_%s_sensor_analysis' % clf_name
decim = 2
event_id = config.event_id
reject = config.reject
c_names = ['word', 'nonword']
subjects = config.subjects

# classifier
clf = make_pipeline(StandardScaler(), LogisticRegression())
n_folds = 5
random_state = 42
# decoding parameters
tmin, tmax = -.2, 1
# smoothing window
length = decim * 1e-3
step = decim * 1e-3

# setup group
group_template = op.join(path, 'group', 'group_%s_%s_filt_%s.%s')
fname_group = group_template % (exp, filt, analysis + '_dict', 'mne')


if redo:
    for subject in subjects:
        print config.banner % subject
        # define filenames
        subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
        fname_proj = subject_template % (exp, '_calm_' + filt + '_filt-proj', 'fif')
        fname_raw = subject_template % (exp, '_calm_' + filt + '_filt-raw', 'fif')
        fname_evts = subject_template % (exp, '-eve', 'txt')
        fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_gat', 'npy')
        fname_rerf = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                         + '_rerf-ave', 'fif')
        fname_cov = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_data-cov', 'fif')
        fname_weights = subject_template % (exp, '_calm_' + filt + '_filt_'
                                            + analysis + '_gat_weights', 'npy')


        # loading events and raw
        evts = mne.read_events(fname_evts)
        # map word, then nonword
        evts = mne.event.merge_events(evts, [1, 2, 5, 6], 99)
        evts = mne.event.merge_events(evts, [9, 10], 100)
        event_id = {c_names[0]: 99, c_names[1]: 100}
        raw = mne.io.read_raw_fif(fname_raw, preload=True, verbose=False)

        # add/apply proj
        proj = [mne.read_proj(fname_proj)[0]]
        raw.add_proj(proj).apply_proj()
        # select only meg channels
        raw.pick_types(meg=True)

        # TO DO: make an issue about equalize events from just the event matrix
        # and event_id. this is needed for linear_regression_raw

        # run a rERF
        rerf = linear_regression_raw(raw, evts, event_id, tmin=tmin, tmax=tmax,
                                     decim=decim, reject=reject)
        mne.write_evokeds(fname_rerf, list(rerf.values()))

        # create epochs for gat
        epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax,
                            baseline=None, decim=decim, reject=reject,
                            preload=True, verbose=False)
        epochs = epochs[[c_names[0], c_names[1]]]
        epochs.equalize_event_counts([c_names[0], c_names[1]], copy=False)
        # Convert the labels of the data to binary descriptors
        lbl = LabelEncoder()
        y = lbl.fit_transform(epochs.events[:,-1])

        print 'get ready for decoding ;)'

        # Generalization Across Time
        # default GAT: LogisticRegression with StratifiedKFold (n=5)
        train_times = {'start': tmin,
                       'stop': tmax,
                       'length': length,
                       'step': step
                       }
        gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=-1,
                                       train_times=train_times, clf=clf,
                                       cv=n_folds)
        gat.fit(epochs, y=y)
        gat.score(epochs, y=y)
        np.save(fname_gat, gat.scores_)

        # store weights
        weights = list()
        for fold in range(n_folds):
            # weights explained: gat.estimator_[time_point][fold].steps[-1][-1].coef_
            weights.append(np.vstack([gat.estimators_[idx][fold].steps[-1][-1].coef_
                                      for idx in range(len(epochs.times))]))
        np.save(fname_weights, np.array(weights))
        cov = mne.compute_covariance(epochs)
        cov.save(fname_cov)


else:
    group_dict = pickle.load(open(fname_group))
    subjects = group_dict['subjects']

####################
# Group Statistics #
####################
group_dict = group_stats(subjects, path, exp, filt, analysis, c_names)

pickle.dump(group_dict, open(fname_group, 'w'))
