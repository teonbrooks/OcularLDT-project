import os.path as op
import numpy as np
import scipy as sp

import mne
from mne.channels import read_ch_connectivity
from mne.stats import spatio_temporal_cluster_1samp_test as stc_1samp_test

def group_stats(subjects, path, exp, filt, analysis, c_names, seed=42,
                threshold=1.96, p_accept=0.05, chance=.5, n_perm=1000):
    # Load the subject gats
    group_gat = list()
    group_reg = list()
    group_dict = dict()
    for subject in subjects:
        subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
        fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_gat', 'npy')
        fname_reg = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                         + '_reg-ave', 'fif')
        group_gat.append(np.load(fname_gat))
        reg = mne.read_evokeds(fname_reg)
        if isinstance(c_names, list):
            reg = mne.evoked.combine_evoked([reg[c_names[0]], reg[c_names[1]]],
                                            weights=[1, -1])
        elif isinstance(c_names, str):
            reg = reg[0]
        # transpose for the stats func
        group_reg.append(reg.data.T)

    n_subjects = len(subjects)
    connectivity, ch_names = read_ch_connectivity('KIT-208')

    #############################
    # run a spatio-temporal REG #
    #############################
    group_reg = np.array(group_reg)
    group_dict['reg_stats'] = stc_1samp_test(group_reg, n_permutations=n_perm,
                                             threshold=threshold, tail=0,
                                             connectivity=connectivity,
                                             seed=seed)

    #########################
    # run a GAT clustering  #
    #########################
    # remove chance from the gats
    group_gat = np.array(group_gat) - chance
    n_chan = len(ch_names)
    _, clusters, p_values, _ = stc_1samp_test(group_gat, n_permutations=n_perm,
                                              threshold=threshold, tail=0,
                                              seed=seed, out_type='mask')
    p_values_ = np.ones_like(group_gat[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    group_dict['gat_sig'] = p_values_ < p_accept

    return group_dict
