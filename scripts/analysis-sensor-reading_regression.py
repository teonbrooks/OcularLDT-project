import pickle
import os.path as op
import numpy as np
import scipy as sp

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

import mne
from mne.decoding import GeneralizationAcrossTime
from mne.stats import (linear_regression_raw,
                       spatio_temporal_cluster_1samp_test as stc_1samp_test)
from mne.channels import read_ch_connectivity

import config


# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
clf_name = 'ridge'
analysis = 'reading_%s_regression_sensor_analysis'
random_state = 42
decim = 4
# decoding parameters
tmin, tmax = -.2, 1
# smoothing window
length = decim * 1e-3
step = decim * 1e-3
event_id = {'word/prime/unprimed': 1,
            'word/target/unprimed': 2,
            'word/prime/primed': 5,
            'word/target/primed': 6,
            }
reject = config.reject
# classifier
reg = Ridge(alpha=1e-3)  # Ridge Regression
clf = Pipeline([('scaler', StandardScaler()), ('ridge', reg)])
# clustering
connectivity, ch_names = read_ch_connectivity('KIT-208')
threshold = 1.96
p_accept = 0.05

# setup group
group_template = op.join(path, 'group', 'group_%s_%s_filt_%s_%s.%s')
fname_group_gat = group_template % (exp, filt, analysis, 'gat', 'mne')

group_gat = dict()
group_rerf = dict()
group_rerf_diff = list()
group_ols = dict()


# Ranker function
def rank_scorer(y, y_pred):
    # because of dimensionality issues with the output of sklearn regression
    # one needs to ravel
    y = np.ravel(y)
    y_pred = np.ravel(y_pred)
    n = y.size
    n_comb = sp.misc.comb(n, 2)

    y_compare = np.tile(y, (n, 1))
    y_compare = y_compare - y_compare.T

    y_pred_compare = np.tile(y_pred, (n, 1))
    y_pred_compare = y_pred_compare - y_pred_compare.T

    # positive = correct prediction, negative = incorrect prediction
    score = y_compare * y_pred_compare
    # we need to remove the diagonal from the combinations
    score = (score > 0).sum()/ (2 * n_comb)

    return score


for subject in config.subjects:
    print config.banner % subject
    # define filenames
    subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
    fname_proj = subject_template % (exp, '_calm_' + filt + '_filt-proj', 'fif')
    fname_raw = subject_template % (exp, '_calm_' + filt + '_filt-raw', 'fif')
    fname_evts = subject_template % (exp, '_fixation_coreg-eve', 'txt')
    fname_dm = subject_template % (exp, '_fixation_design_matrix', 'txt')
    # loading events and raw
    evts = mne.read_events(fname_evts)

    # map word, then nonword
    evts = mne.event.merge_events(evts, [1, 2, 5, 6], 99)
    event_id = {'word': 99}

    # loading design matrix, epochs, proj
    design_matrix = np.loadtxt(fname_dm)

    # # let's look at the time around the fixation
    # durs = np.asarray(design_matrix[:, -1] * 1000, int)
    # evts[:, 0] = evts[:, 0] + durs

    raw = mne.io.read_raw_fif(fname_raw, preload=True, verbose=False)

    # add/apply proj
    proj = [mne.read_proj(fname_proj)[0]]
    raw.add_proj(proj).apply_proj()
    # select only meg channels
    raw.pick_types(meg=True)

    epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax,
                        baseline=None, decim=decim, reject=reject,
                        preload=True, verbose=False)

    dm_keys = evts[:, 0]
    # epochs rejection: filtering
    # drop based on MEG rejection, must happen first
    epochs.drop_bad_epochs(reject=reject)
    design_matrix = design_matrix[epochs.selection]
    dm_keys = dm_keys[epochs.selection]
    # remove zeros
    idx = design_matrix[:, -1] > 0
    epochs = epochs[idx]
    design_matrix = design_matrix[idx]
    dm_keys = dm_keys[idx]
    # define outliers
    durs = design_matrix[:, -1]
    mean, std = durs.mean(), durs.std()
    devs = np.abs(durs - mean)
    criterion = 3 * std
    # remove outliers
    idx = devs < criterion
    epochs = epochs[idx]
    design_matrix = design_matrix[idx]
    dm_keys = dm_keys[idx]

    assert len(design_matrix) == len(epochs) == len(dm_keys)
    # group_ols[subject] = epochs.average()
    # Define 'y': what you're predicting
    y = design_matrix[:, -1]

    # run a rERF
    covariates = dict(zip(dm_keys, y))
    rerf = linear_regression_raw(raw, evts, event_id, tmin=tmin, tmax=tmax,
                                 decim=decim, reject=reject,
                                 covariates=covariates)
    rerf_diff = mne.evoked.combine_evoked([rerf[c_names[0]], rerf[c_names[1]]],
                                          weights=[1, -1])
    # take the magnitude of the difference so that the t-val is interpretable
    group_rerf_diff.append(np.abs(rerf_diff.data.T))
    group_rerf[subject] = rerf


    # #############################
    # # run a spatio-temporal OLS #
    # #############################
    # names = ['intercept', 'fixation']
    # stats = linear_regression(epochs, design_matrix, names)
    #
    # # can you do this at the group level?
    # # run a spatio-temporal linear regression
    # X = stats['fixation'].t_val.data.swapaxes(1, 2)
    # cluster_stats = spatio_temporal_cluster_1samp_test(X, n_permutations=1000,
    #                     threshold=threshold, tail=0, connectivity=connectivity)

    print 'get ready for decoding ;)'

    train_times = {'start': tmin,
                   'stop': tmax,
                   'length': length,
                   'step': step
                   }
    cv = KFold(n=len(y), n_folds=5, random_state=random_state)
    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=-1,
                                   train_times=train_times, scorer=rank_scorer,
                                   clf=clf, cv=cv)
    gat.fit(epochs, y=y)
    gat.score(epochs, y=y)
    group_gat[subject] = np.array(gat.scores_)

# define a layout
layout = mne.find_layout(epochs.info)
# additional properties
group_gat['layout'] = group_rerf['layout'] = layout
group_gat['times'] = group_rerf['times'] = epochs.times
group_gat['sfreq'] = group_rerf['sfreq'] = epochs.info['sfreq']

# temp hack
gat.scores_ = np.array([group_gat[subject] for subject
                        in config.subjects]).mean(axis=0)
group_gat['group'] = gat

#########################
# run a GAT clustering  #
#########################
group_gat_diff = np.array([group_gat[subject] for subject
                           in config.subjects]) - .5
n_subjects = len(config.subjects)
n_chan = raw.info['nchan']
connectivity, ch_names = read_ch_connectivity('KIT-208')

threshold = 1.96
p_accept = 0.05
group_gat['stats'] = stc_1samp_test(group_gat_diff, n_permutations=1000,
                                    threshold=threshold, tail=0,
                                    seed=random_state)

# h5io.write_hdf5(fname_group_gat, group_gat)
pickle.dump(group_gat, open(fname_group_gat, 'w'))
