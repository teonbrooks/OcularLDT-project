import tomllib as toml
from pathlib import Path
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from mne.decoding import (Vectorizer, SlidingEstimator, cross_val_multiscore,
                          Scaler, LinearModel, get_coef)


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))
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
fname_group_template= op.join(cfg['project_path'], 'output', 'group',
                              f'group_{task}_sensor_priming_%s.npy')

subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])

group_scores = list()
group_patterns = list()
for subject in subjects_list:
    print(cfg['banner'] % subject)
    bids_path.update(subject=subject)

    # define filenames
    fname_ica = bids_path.update(suffix='ica').fpath
    fname_weights = bids_path.update(suffix='weights', extension='.npy').fpath

    # loading events and raw
    raw = read_raw_bids(bids_path)
    raw.pick_types(meg=True)
    events, event_id = mne.events_from_annotations(raw)

    # apply ICA
    ica = mne.preprocessing.read_ica(fname_ica)

    # create epochs for sliding estimator
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

    # perform cross-validation on the time decoding
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    scores = cross_val_multiscore(time_decod, X, y=y, cv=n_folds, n_jobs=1)
    scores = np.mean(scores, axis=0)
    group_scores.append(scores)

    # retrieve the pattern for the model fit
    time_decod.fit(X, y)
    pattern = get_coef(time_decod, 'patterns_', inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(pattern, epochs.info,
                                      tmin=epochs.times[0])
    group_patterns.append(pattern)

# save all the scores across participants
group_scores = np.vstack(group_scores)
np.save(fname_group_template % 'scores', group_scores)

# save all the spatial patterns across participants
group_patterns = np.vstack(group_patterns)
np.save(fname_group_template % 'patterns', group_patterns)

# save the shared time course
np.save(fname_group_template % 'times', epochs.times)
