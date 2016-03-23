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
from mne.stats import (linear_regression_raw, linear_regression,
                       spatio_temporal_cluster_1samp_test as stc_1samp_test)
from mne.channels import read_ch_connectivity

import config


# parameters
path = config.drive
filt = config.filt
redo = True
img = config.img
exp = 'OLDT'
clf_name = 'ridge'
analysis = 'reading_%s_regression_10ms_sensor_analysis' % clf_name
random_state = 42
decim = 2
# decoding parameters
tmin, tmax = -.2, 1
# smoothing window
length = 10 * 1e-3
step = 10 * 1e-3
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
group_reg = dict()
group_reg_stats = list()
group_ols = dict()


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

if redo:
    for subject in config.subjects:
        print config.banner % subject
        # define filenames
        subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
        fname_proj = subject_template % (exp, '_calm_' + filt + '_filt-proj', 'fif')
        fname_raw = subject_template % (exp, '_calm_' + filt + '_filt-raw', 'fif')
        fname_evts = subject_template % (exp, '_fixation_coreg-eve', 'txt')
        fname_dm = subject_template % (exp, '_fixation_design_matrix', 'txt')
        fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_gat', 'npy')
        fname_reg = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_reg-ave', 'fif')
        # loading events and raw
        evts = mne.read_events(fname_evts)

        # map word, then nonword
        evts = mne.event.merge_events(evts, [1, 2, 5, 6], 99)
        event_id = {'word': 99}

        # loading design matrix, epochs, proj
        design_matrix = np.loadtxt(fname_dm)
        reg_names = ('intecept', 'ffd')

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

        # epochs rejection: filtering
        # drop based on MEG rejection, must happen first
        epochs.drop_bad_epochs(reject=reject)
        design_matrix = design_matrix[epochs.selection]
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
        reg['ffd'].beta.save(fname_reg)

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
        np.save(fname_gat, gat.scores_)

    # define a layout
    layout = mne.find_layout(epochs.info)
    # additional properties
    group_dict['layout'] = layout
    group_dict['times'] = epochs.times
    group_dict['sfreq'] = epochs.info['sfreq']

else:
    group_dict = pickle.load(open(fname_group))
    subjects = group_dict['subjects']

####################
# Group Statistics #
####################
# Load the subject gats
group_gat = list()
group_reg = list()
for subject in subjects:
    subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
    fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                    + '_gat', 'npy')
    fname_reg = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                     + '_reg-ave', 'fif')
    group_gat.append(np.load(fname_gat))
    reg = mne.read_evoked(fname_reg)
    # transpose for the stats func
    group_reg.append(reg.data.T)

# Parameters
threshold = 1.96
p_accept = 0.05
chance = .5
n_subjects = len(config.subjects)
connectivity, ch_names = read_ch_connectivity('KIT-208')

#############################
# run a spatio-temporal REG #
#############################
group_reg = np.array(group_reg)
group_dict['reg_stats'] = stc_1samp_test(group_reg, n_permutations=1000,
                                         threshold=threshold, tail=0,
                                         connectivity=connectivity,
                                         seed=random_state)

#########################
# run a GAT clustering  #
#########################
# remove chance from the gats
group_gat = np.array(group_gat) - chance
n_chan = raw.info['nchan']
_, clusters, p_values, _ = stc_1samp_test(group_gat, n_permutations=1000,
                                          threshold=threshold, tail=0,
                                          seed=random_state, out_type='mask')
p_values_ = np.ones_like(group_gat[0]).T
for cluster, pval in zip(clusters, p_values):
    p_values_[cluster.T] = pval
group_dict['gat_sig'] = p_values_ < p_accept

pickle.dump(group_dict, open(fname_group, 'w'))
