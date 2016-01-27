import pickle
import os.path as op
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import mne
from mne.decoding import GeneralizationAcrossTime
from mne.stats import (linear_regression_raw,
                       spatio_temporal_cluster_1samp_test as stc_1samp_test,
                       spatio_temporal_cluster_test as stc_test)
from mne.channels import read_ch_connectivity
# import h5io

import config

# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
analysis = 'word-nonword_sensor_analysis'
random_state = 42
decim = 4
# decoding parameters
tmin, tmax = -.2, 1
# smoothing window
length = decim * 1e-3
step = decim * 1e-3
event_id = config.event_id
reject = config.reject


# setup group
group_template = op.join(config.results_dir, 'group',
                         'group_OLDT_%s_filt_%s_%s.%s')
fname_group_rerf = group_template % (filt, analysis, 'rerf', 'mne')
fname_group_gat = group_template % (filt, analysis, 'gat', 'mne')

group_gat = dict()
group_rerf = dict()
group_rerf_diff = list()

for subject in config.subjects:
    print config.banner % subject
    # define filenames
    subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
    fname_proj = subject_template % (exp, '_calm_' + filt + '_filt-proj', 'fif')
    fname_raw = subject_template % (exp, '_calm_' + filt + '_filt-raw', 'fif')
    fname_evts = subject_template % (exp, '-eve', 'txt')

    # loading events and raw
    evts = mne.read_events(fname_evts)
    raw = mne.io.read_raw_fif(fname_raw, preload=True, verbose=False)
    c_names = ['word', 'nonword']

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
    rerf_diff = mne.evoked.combine_evoked([rerf[c_names[0]], rerf[c_names[1]]],
                                          weights=[1, -1])
    # take the magnitude of the difference so that the t-val is interpretable
    group_rerf_diff.append(np.abs(rerf_diff.data.T))
    group_rerf[subject] = rerf

    # create epochs for gat
    epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax,
                        baseline=None, decim=decim, reject=reject,
                        preload=True, verbose=False)
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
                                   train_times=train_times)
    gat.fit(epochs, y=y)
    gat.score(epochs, y=y)
    group_gat[subject] = np.array(gat.scores_)

# define a layout
layout = mne.find_layout(epochs.info)
# additional properties
group_gat['layout'] = group_rerf['layout'] = layout
group_gat['times'] = group_gat['times'] = epochs.times
group_gat['sfreq'] = group_gat['sfreq'] = epochs.info['sfreq']

# temp hack
gat.scores_ = np.array([group_gat[subject] for subject
                        in config.subjects]).mean(axis=0)
group_gat['group'] = gat

##############
# Statistics #
##############

##############################
# run a spatio-temporal RERF #
##############################
group_rerf_diff = np.array(group_rerf_diff)
n_subjects = len(config.subjects)
n_chan = raw.info['nchan']
connectivity, ch_names = read_ch_connectivity('KIT-208')

threshold = 1.96
p_accept = 0.05
group_rerf['stats'] = stc_1samp_test(group_rerf_diff, n_permutations=1000,
                                     threshold=threshold, tail=0,
                                     connectivity=connectivity,
                                     seed=random_state)

# h5io.write_hdf5(fname_group_rerf, group_rerf)
pickle.dump(group_rerf, open(fname_group_rerf, 'w'))


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
