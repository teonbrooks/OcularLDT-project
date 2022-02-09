import pickle
import os.path as op
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import mne
from mne.report import Report
from mne_bids import get_entity_vals, BIDSPath, read_raw_bids
from mne.decoding import (Vectorizer, SlidingEstimator, cross_val_multiscore,
                          Scaler, LinearModel, get_coef)


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
# parameters
random_state = 42
# decoding parameters
tmin, tmax = -.1, 1
n_folds = 5
# baseline
bmin, bmax = -.2, -.1
reject = cfg['reject']
c_names = ['word/target/primed', 'word/target/unprimed']

# setup group
# group_template = op.join(project_path, 'output', 'group',
#                          f'group_OcularLDT_sensor_{analysis}.mne')

subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])

for subject in subjects_list[:1]:
    print(cfg['banner'] % subject)
    bids_path.update(subject=subject)

    # define filenames
    subject_template = op.join(cfg['bids_root'], f"sub-{subject}", 'meg',
                               f"sub-{subject}_task-{task}")
    fname_ica = f"{subject_template}_ica.fif"
    fname_weights = f"{subject_template}_weights.npy"

    # loading events and raw
    raw = read_raw_bids(bids_path)
    raw.pick_types(meg=True)
    events, event_id = mne.events_from_annotations(raw)

    # apply ICA
    ica = mne.preprocessing.read_ica(fname_ica)

    # create epochs for gat
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                        baseline=None, reject=reject,
                        preload=True, verbose=False)
    epochs = epochs[[c_names[0], c_names[1]]]
    epochs.equalize_event_counts([c_names[0], c_names[1]])
    epochs = ica.apply(epochs)

    # Convert the labels of the data to binary descriptors
    lbl = LabelEncoder()
    y = lbl.fit_transform(epochs.events[:,-1])
    X = epochs.get_data()
    clf = make_pipeline(Scaler(scalings='median'),
                        Vectorizer(),
                        LinearModel(LogisticRegression(solver='liblinear',
                                                       max_iter=500)))


    print('get ready for decoding ;)')

    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    scores = cross_val_multiscore(time_decod, X, y=y, cv=n_folds, n_jobs=1)
    scores = np.mean(scores, axis=0)



#     # store weights
#     weights = list()
#     for fold in range(n_folds):
#         # weights explained: gat.estimator_[time_point][fold].steps[-1][-1].coef_
#         weights.append(np.vstack([gat.estimators_[idx][fold].steps[-1][-1].coef_
#                                     for idx in range(len(epochs.times))]))
#     np.save(fname_weights, np.array(weights))
#     cov = mne.compute_covariance(epochs)
#     cov.save(fname_cov)

# else:
#     group_dict = pickle.load(open(fname_group))
#     subjects = config.subjects

# ####################
# # Group Statistics #
# ####################
# group_dict = group_stats(subjects, path, exp, filt, analysis, c_names)

# pickle.dump(group_dict, open(fname_group, 'w'))
