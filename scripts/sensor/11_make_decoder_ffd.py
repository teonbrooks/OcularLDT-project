import json
import os.path as op
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

import mne
from mne_bids import get_entity_vals, BIDSPath, read_raw_bids
from mne.decoding import (Vectorizer, SlidingEstimator, cross_val_multiscore,
                          Scaler, LinearModel, get_coef)

cfg = json.load(open(op.join('/', 'Users', 'teonbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
# parameters
random_state = 42
tmin, tmax = -.1, 1
clf_name = 'ridge'
reg_type = 'reg'
event_id = {'word/prime/unprimed': 1,
            'word/target/unprimed': 2,
            'word/prime/primed': 5,
            'word/target/primed': 6,
            }
reject = cfg['reject']
c_name = 'ffd'

# classifier
reg = Ridge(alpha=1e-3)  # Ridge Regression
clf = make_pipeline(Scaler(scalings='median'),
                    Vectorizer(),
                    LinearModel(Ridge(alpha=1e-3)))
# clf = Pipeline([('scaler', StandardScaler()), ('ridge', reg)])

# decoding parameters
# tmin, tmax = -.2, 1
# smoothing window
n_folds = 5

# setup group
fname_group_template= op.join(cfg['project_path'], 'output', 'group',
                              f'group_{task}_sensor_ffd_%s.npy')

subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])


# Ranker function
def rank_scorer(y, y_pred):
    # because of dimensionality issues with the output of sklearn regression
    # one needs to ravel
    y = np.ravel(y)
    y_pred = np.ravel(y_pred)
    n = y.size
    # get the total number of combinations
    n_comb = sp.misc.comb(n, 2)

    """
    Comparisons
    -----------
    - You tile the `y` so you can have make all possible comparisons.
    - When you transpose a copy of `y` and then subtract it,
      you are now doing pairwise comparisons if each combination.
    - The diagonal is a comparison with itself (remove), and above and below
      are mirror of combinations, so you only need half of them.
    """
    y_compare = np.tile(y, (n, 1))
    y_compare = y_compare - y_compare.T

    # do the exact same thing for the y_pred
    y_pred_compare = np.tile(y_pred, (n, 1))
    y_pred_compare = y_pred_compare - y_pred_compare.T

    # positive = correct prediction, negative = incorrect prediction
    score = y_compare * y_pred_compare
    # we need to remove the diagonal from the combinations
    score = (score > 0).sum()/ (2 * n_comb)

    return score

group_scores = list()
group_patterns = list()
for subject in subjects_list[:1]:
    print(cfg['banner'] % subject)

    bids_path.update(subject=subject)
    # # define filenames
    # subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
    # fname_proj = subject_template % (exp, '_calm_' + filt + '_filt-proj', 'fif')
    # fname_raw = subject_template % (exp, '_calm_' + filt + '_filt-raw', 'fif')
    # fname_evts = subject_template % (exp, '_fixation_coreg-eve', 'txt')
    # fname_dm = subject_template % (exp, '_fixation_design_matrix', 'txt')
    # fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
    #                                 + '_gat', 'npy')
    # fname_reg = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
    #                                 + '_reg-ave', 'fif')
    # fname_cov = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
    #                                 + '_data-cov', 'fif')
    # fname_weights = subject_template % (exp, '_calm_' + filt + '_filt_'
    #                                     + analysis + '_gat_weights', 'npy')

    # define filenames
    subject_template = op.join(cfg['bids_root'], f"sub-{subject}", 'meg',
                               f"sub-{subject}_task-{task}")
    fname_ica = f"{subject_template}_ica.fif"
    fname_weights = f"{subject_template}_weights.npy"

    fname_ffd = op.join(cfg['bids_root'], f"sub-{subject}", 'eyetrack',
                        f"{subject}_OLDT_fixation_times.txt")

    # loading events and raw
    raw = read_raw_bids(bids_path)
    raw.picks('meg')
    events, event_id = mne.events_from_annotations(raw)

    # map word, then nonword
    events = mne.event.merge_events(events, [1, 2, 5, 6], 99)
    event_id = {'word': 99}

    # loading design matrix, epochs, proj
    design_matrix = pd.read_csv(fname_ffd)
    reg_names = ('intercept', 'ffd')

    # # let's look at the time around the fixation
    # durs = np.asarray(design_matrix[:, -1] * 1000, int)
    # evts[:, 0] = evts[:, 0] + durs

    raw = read_raw_bids(bids_path)
    raw.pick('meg')

    # apply ICA
    ica = mne.preprocessing.read_ica(fname_ica)

    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)

    # epochs rejection: filtering
    # drop based on MEG rejection, must happen first
    epochs.drop_bad(reject=reject)
    design_matrix = design_matrix.loc[epochs.selection]
    evts = evts[epochs.selection]
    # remove zeros
    idx = design_matrix[:, -1] > 0
    epochs = epochs[idx]
    design_matrix = design_matrix[idx]
    evts = evts[idx]
    # define outliers
    durs = design_matrix[:, -1]
    mean, std = durs.mean(), durs.std()
    devs = np.abs(durs - mean)
    criterion = 3 * std
    # remove outliers
    idx = devs < criterion
    epochs = epochs[idx]
    design_matrix = design_matrix[idx]
    evts = evts[idx]

    # rerf keys
    dm_keys = evts[:, 0]

    assert len(design_matrix) == len(epochs) == len(dm_keys)
    # group_ols[subject] = epochs.average()
    # Define 'y': what you're predicting
    y = design_matrix[:, -1]

    # run a rERF
    covariates = dict(zip(dm_keys, y))
    # linear regression
    reg = linear_regression(epochs, design_matrix, reg_names)
    reg[c_name].beta.save(fname_reg)

    print('get ready for decoding ;)')

    scores = cross_val_multiscore(reg, X, y=y, cv=n_folds, n_jobs=-1)
    # cv = KFold(n=len(y), n_folds=n_folds, random_state=random_state)
    # gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
    #                                 scorer=rank_scorer, clf=clf, cv=cv)
    # gat.fit(epochs, y=y)
    # gat.score(epochs, y=y)
    # print gat.scores_.shape
    # np.save(fname_gat, gat.scores_)

    # store weights
    weights = list()
    for fold in range(n_folds):
        # weights explained: gat.estimator_[time_point][fold].steps[-1][-1].coef_
        weights.append(np.vstack([gat.estimators_[idx][fold].steps[-1][-1].coef_
                                    for idx in range(len(epochs.times))]))
    np.save(fname_weights, np.array(weights))
    cov = mne.compute_covariance(epochs)
    cov.save(fname_cov)

# define a layout
layout = mne.find_layout(epochs.info)
# additional properties
group_dict['layout'] = layout
group_dict['times'] = epochs.times
group_dict['sfreq'] = epochs.info['sfreq']

####################
# Group Statistics #
####################
group_dict.update(group_stats(subjects, path, exp, filt, analysis, c_name,
                              reg_type=reg_type))

pickle.dump(group_dict, open(fname_group, 'w'))
