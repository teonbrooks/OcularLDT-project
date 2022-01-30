import pickle
import os.path as op
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import mne
from mne.decoding import Scaler, SlidingEstimator
from mne.stats import linear_regression_raw
from mne.channels import read_ch_connectivity

import config
from analysis_func import group_stats


# parameters
bids_path = '/Volumes/teon-backup/Experiments/OcularLDT'
project_path = '~/codespace/OcularLDT-project'
clf_name = 'logit'
analysis = 'priming_%s_no_pca_sensor_analysis' % clf_name
clf = make_pipeline(StandardScaler(), LogisticRegression())
random_state = 42
decim = 2
# decoding parameters
tmin, tmax = -.2, 1
n_folds = 5
# baseline
bmin, bmax = -.2, -.1
# smoothing window
length = decim * 1e-3
step = decim * 1e-3
event_id = config.event_id
reject = config.reject
c_names = ['word/target/primed', 'word/target/unprimed']

# setup group
group_template = op.join(project_path, 'output', 'group',
                         f'group_OcularLDT_sensor_{analysis}.mne')

subjects = config.subjects

for subject in subjects:
    print(config.banner % subject)
    # define filenames
    subject_template = op.join(bids_path, subject, 'meg', f'sub-{subject}_task-{task}')
    fname_raw = f'{subject_template}_meg.fif'
    fname_ica = f'{subject_template}_ica.fif'
    fname_evts = f'{subject_template}_events.tsv'
    fname_rerf = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_rerf-ave', 'fif')
    fname_cov = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                    + '_data-cov', 'fif')
    fname_weights = subject_template % (exp, '_calm_' + filt + '_filt_'
                                        + analysis + '_gat_weights', 'npy')

    # loading events and raw
    evts = mne.read_events(fname_evts)
    raw = mne.io.read_raw_fif(fname_raw, preload=True, verbose=False)

    # add/apply proj
    # proj = [mne.read_proj(fname_proj)[0]]
    # raw.add_proj(proj).apply_proj()
    # select only meg channels
    raw.pick_types(meg=True)

    # TO DO: make an issue about equalize events from just the event matrix
    # and event_id. this is needed for linear_regression_raw

    # run a rERF
    rerf = linear_regression_raw(raw, evts, event_id, tmin=tmin, tmax=tmax,
                                    decim=decim, reject=reject)

    mne.write_evokeds(fname_rerf, rerf.values())

    # create epochs for gat
    epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax,
                        baseline=None, reject=reject, decim=decim,
                        preload=True, verbose=False)
    epochs = epochs[[c_names[0], c_names[1]]]
    epochs.equalize_event_counts([c_names[0], c_names[1]], copy=False)
    # Convert the labels of the data to binary descriptors
    lbl = LabelEncoder()
    y = lbl.fit_transform(epochs.events[:,-1])

    print 'get ready for decoding ;)'

    # Generalization Across Time
    # default GAT: LogisticRegression with KFold (n=5)
    train_times = {'start': tmin,
                    'stop': tmax,
                    'length': length,
                    'step': step
                    }
    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                                    train_times=train_times, clf=clf, cv=n_folds)
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
    subjects = config.subjects

####################
# Group Statistics #
####################
group_dict = group_stats(subjects, path, exp, filt, analysis, c_names)

pickle.dump(group_dict, open(fname_group, 'w'))
