import os.path as op
import numpy as np
import scipy as sp

import mne
from mne.channels import read_ch_connectivity
from mne.stats import (spatio_temporal_cluster_1samp_test as stc_1samp_test,
                       permutation_cluster_1samp_test as pc_1samp_test)

def group_stats(subjects, path, exp, filt, analysis, c_names, seed=42,
                threshold=1.96, p_accept=0.05, chance=.5, n_perm=10000,
                reg_type='rerf'):
    # Load the subject gats
    group_gat = list()
    group_td = list()
    group_reg = list()
    group_dict = dict()
    group_dev = list()
    for subject in subjects:
        subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
        fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                        + '_gat', 'npy')
        fname_reg = subject_template % (exp, '_calm_' + filt + '_filt_' + analysis
                                         + '_%s-ave' % reg_type, 'fif')
        gat = np.load(fname_gat)
        group_gat.append(gat)
        group_td.append(np.diag(gat))

        reg = mne.read_evokeds(fname_reg)
        if isinstance(c_names, list):
            evokeds = list()
            for r in reg:
                if r.comment == c_names[0]:
                    evokeds.insert(0, r)
                elif r.comment == c_names[1]:
                    evokeds.append(r)
            assert len(evokeds) == 2
            reg = mne.evoked.combine_evoked([evokeds[0], evokeds[1]],
                                             weights=[1, -1])
        elif isinstance(c_names, str):
            reg = reg[0]
        # transpose for the stats func
        group_reg.append(reg.data.T)

    n_subjects = len(subjects)
    connectivity, ch_names = read_ch_connectivity('KIT-208')
    ##################
    # Auxiliary info #
    ##################
    # define a layout
    layout = mne.find_layout(reg.info)
    group_dict['layout'] = layout
    group_dict['times'] = reg.times
    group_dict['sfreq'] = reg.info['sfreq']
    group_dict['subjects'] = subjects

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
    _, clusters, p_values, _ = stc_1samp_test(group_gat, n_permutations=n_perm,
                                              threshold=threshold, tail=0,
                                              seed=seed, out_type='mask')
    p_values_ = np.ones_like(group_gat[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    group_dict['gat_sig'] = p_values_ < p_accept

    ########################
    # run a TD clustering  #
    ########################
    # remove chance from the gats
    group_td = np.array(group_td) - chance
    group_dict['td_stats'] = pc_1samp_test(group_td, n_permutations=n_perm,
                                           threshold=threshold, tail=0,
                                           seed=seed)

    #########################
    # run a GAT clustering  #
    #########################
    # determining deviation from diag
    group_diag = np.array([np.diag(gat)[:, np.newaxis] for gat in group_gat])
    group_dev = group_gat - group_diag

    _, clusters, p_values, _ = stc_1samp_test(group_dev, n_permutations=n_perm,
                                              threshold=threshold, tail=0,
                                              seed=seed, out_type='mask')
    p_values_ = np.ones_like(group_dev[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    group_dict['gat_dev_sig'] = p_values_ < p_accept


    return group_dict
