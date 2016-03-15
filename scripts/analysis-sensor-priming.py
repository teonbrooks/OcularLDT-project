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
from mne.stats import (linear_regression_raw,
                       spatio_temporal_cluster_1samp_test as stc_1samp_test)
from mne.channels import read_ch_connectivity

import config


# parameters
path = config.drive
filt = config.filt
redo = False
img = config.img
exp = 'OLDT'
clf_name = 'logit'
analysis = 'priming_%s_sensor_analysis' % clf_name
clf = make_pipeline(StandardScaler(), LogisticRegression())
random_state = 42
decim = 2
# decoding parameters
tmin, tmax = -.2, 1
# baseline
bmin, bmax = -.2, -.1
# smoothing window
length = decim * 1e-3
step = decim * 1e-3
event_id = config.event_id
reject = config.reject
c_names = ['word/target/primed', 'word/target/unprimed']

# setup group
group_template = op.join(path, 'group', 'group_%s_%s_filt_%s.%s')
fname_group = group_template % (exp, filt, analysis + '_dict', 'mne')

group_dict = dict()
group_rerf = list()
subjects = config.subjects

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
                                         + '_rerf', 'mne')


        # loading events and raw
        evts = mne.read_events(fname_evts)
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
        group_rerf.append(rerf)
        pickle.dump(rerf, open(fname_rerf, 'w'))

        # create epochs for gat
        epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax,
                            baseline=(bmin, bmax), reject=reject, decim=decim,
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
        gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=-1,
                                       train_times=train_times, clf=clf)
        gat.fit(epochs, y=y)
        gat.score(epochs, y=y)
        np.save(fname_gat, np.array(gat.scores_))

    ##################
    # Auxiliary info #
    ##################
    # define a layout
    layout = mne.find_layout(epochs.info)
    # additional properties
    group_dict['layout'] = layout
    group_dict['times'] = epochs.times
    group_dict['sfreq'] = epochs.info['sfreq']
    group_dict['subjects'] = subjects

else:
    group_dict = pickle.load(open(fname_group))
    subjects = group_dict['subjects']

##############
# Statistics #
##############
# Load the subject gats
group_gat = list()
group_rerf = list()
for subject in subjects:
    subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
    fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                    + '_gat', 'npy')
    fname_rerf = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                     + '_rerf', 'mne')
    group_gat.append(np.load(fname_gat))
    rerf = pickle.load(open(fname_rerf))
    # compute the diff
    rerf_diff = mne.evoked.combine_evoked([rerf[c_names[0]], rerf[c_names[1]]],
                                          weights=[1, -1])
    # transpose for the stats func
    group_rerf.append(rerf_diff.data.T)

# Parameters
threshold = 1.96
p_accept = 0.05
chance = .5
n_subjects = len(subjects)
connectivity, ch_names = read_ch_connectivity('KIT-208')

##############################
# run a spatio-temporal RERF #
##############################
group_rerf = np.array(group_rerf)
group_dict['rerf_stats'] = stc_1samp_test(group_rerf, n_permutations=1000,
                                          threshold=threshold, tail=0,
                                          connectivity=connectivity,
                                          seed=random_state)

#########################
# run a GAT clustering  #
#########################
# remove chance from the gats
group_gat_diff = np.array(group_gat) - chance
_, clusters, p_values, _ = stc_1samp_test(group_gat_diff, n_permutations=1000,
                                          threshold=threshold, tail=0,
                                          seed=random_state, out_type='mask')
p_values_ = np.ones_like(gat.scores_).T
for cluster, pval in zip(clusters, p_values):
    p_values_[cluster.T] = pval
group_dict['gat_sig'] = p_values_ < p_accept

pickle.dump(group_dict, open(fname_group, 'w'))
