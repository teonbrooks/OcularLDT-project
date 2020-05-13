import pickle
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.report import Report
from mne.viz.topomap import plot_topomap
from jr.plot import pretty_gat, pretty_decod, pretty_slices

import config


def group_plot(subjects, path, results_dir, exp, filt, clf_name, analysis,
               c_names, title, threshold, p_accept, chance, img='png',
               clim=None, reg_type='rerf'):
    # setup group
    group_template = op.join('%s', 'group', 'group_%s_%s_filt_%s.%s')
    fname_group = group_template % (path, exp, filt, analysis + '_dict', 'mne')

    group_dict = pickle.load(open(fname_group))
    group_gat = dict()
    group_rerf = dict()
    group_cov = dict()
    group_weights = dict()

    for subject in subjects:
        subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
        fname_gat = subject_template % (exp, '_calm_' + filt + '_filt_'
                                        + analysis + '_gat', 'npy')
        fname_rerf = subject_template % (exp, '_calm_' + filt + '_filt_'
                                         + analysis + '_%s-ave' % reg_type,
                                         'fif')
        fname_cov = subject_template % (exp, '_calm_' + filt + '_filt_'
                                        + analysis + '_data-cov', 'fif')
        fname_weights = subject_template % (exp, '_calm_' + filt + '_filt_'
                                            + analysis + '_gat_weights', 'npy')
        group_gat[subject] = np.load(fname_gat)
        group_rerf[subject] = mne.read_evokeds(fname_rerf)
        group_cov[subject] = mne.read_cov(fname_cov).data
        group_weights[subject] = np.load(fname_weights)

    group_rep = Report()

    # add'l info
    sfreq, times = group_dict['sfreq'], group_dict['times']

    # unpacking significant time bounds for Time Decoding
    T_obs, clusters, p_values, _ = group_dict['td_stats']
    good_cluster_inds = np.where(p_values < p_accept)[0]

    td_sig = np.zeros(len(times))
    idx_time_sig = list()
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster infomation, get unique indices
        td_sig[clusters[clu_idx]] = 1
        if not isinstance(clu_idx, list):
            clu_idx = [clu_idx]
        for idx in clu_idx:
            interval = clusters[idx][0].indices(len(times))
            if interval[1] == len(times):
                idx = [interval[0], interval[1] - 1]
            else:
                idx = [interval[0], interval[1]]
            idx_time_sig.append(idx)

    ################
    # Group Evoked #
    ################
    if reg_type == 'rerf':
        group_c0 = list()
        group_c1 = list()
        for subject in subjects:
            for r in group_rerf[subject]:
                if r.comment == c_names[0]:
                    group_c0.append(r)
                elif r.comment == c_names[1]:
                    group_c1.append(r)

        group_c0 = mne.grand_average(group_c0)
        group_c1 = mne.grand_average(group_c1)
        grand_averages = [group_c0, group_c1]

        rerf_diff = mne.evoked.combine_evoked([group_c0, group_c1], weights=[1, -1])
        fig = rerf_diff.plot()
        group_rep.add_figs_to_section(fig, 'Grand Average Difference', 'Group Plots')
    else:
        grand_averages = [group_rerf[subject][0] for subject in subjects]
        rerf_diff = mne.evoked.combine_evoked(grand_averages)
        fig = rerf_diff.plot()
        group_rep.add_figs_to_section(fig, 'Grand Average', 'Group Plots')

    ##############
    # Group RERF #
    ##############
    # this block of code for the Group RERF is borrowed from MNE example for
    # spatiotemporal clustering and modified for analyses
    # https://martinos.org/mne/stable/auto_tutorials/plot_spatio_temporal_cluster_stats_sensor.html

    T_obs, clusters, p_values, _ = group_dict['reg_stats']
    good_cluster_inds = np.where(p_values < p_accept)[0]


    # configure variables for visualization
    colors = 'r', 'steelblue'
    linestyles = '-', '-'
    if reg_type == 'reg':
        colors = (colors[-1],)
        linestyles = (linestyles[-1],)

    # get sensor positions via layout
    pos = group_dict['layout'].pos

    captions = list()
    figs = list()
    comments = list()
    # loop over significant clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster infomation, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # since time before baseline is confounded with eye-movements
        # look at time after saccade-related onset
        if times[time_inds.min()] < 0:
            continue

        # get topography for T stat
        t_map = rerf_diff.data[:, time_inds].mean(axis=1)

        # get signals at significant sensors
        signals = [evoked.data[ch_inds, ...].mean(axis=0) for evoked in
                   grand_averages]
        sig_times = times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(16, 3))
        suptitle = 'Group rERF Cluster #{0}'.format(i_clu + 1)
        fig.suptitle(suptitle, fontsize=14)

        # plot average test statistic and mark significant sensors
        image, _ = mne.viz.plot_topomap(t_map, pos, mask=mask, axes=ax_topo,
                                        vmin=np.min, vmax=np.max)

        # advanced matplotlib for showing image with figure and colorbar
        # in one plot
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel('Averaged Field-map ({:0.1f} - {:0.1f} ms)'.format(
            *sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        for signal, name, ls, color in zip(signals, c_names, \
            linestyles, colors):
            ax_signals.plot(times, signal, label=name,
                            linestyle=ls, color=color)

        # add information
        ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
        ax_signals.set_xlim([times[0], times[-1]])
        ax_signals.set_xlabel('time [ms]')
        ax_signals.set_ylabel('evoked magnetic fields [fT]')

        # plot significant time range
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                 color='orange', alpha=0.3)
        ax_signals.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax_signals.set_ylim(ymin, ymax)

        comment = ["Significant Time Region: %.3f to %.3f s"
                   % (sig_times[0], sig_times[-1])]

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        figs.append(fig)
        captions.append(suptitle)
        comments.extend(comment)

    group_rep.add_figs_to_section(figs, captions, 'Group Plots', comments=comments)

    #############
    # Group GAT #
    #############
    sig = group_dict['gat_sig']

    # individual gat plots
    fig, axes = plt.subplots(5, 4, figsize=(20,10))
    axes[-1, -1].axis('off')
    axes = [ax for axis in axes for ax in axis]
    for subject, ax in zip(subjects, axes):
        pretty_gat(scores=group_gat[subject], chance=chance, sfreq=sfreq,
                   ax=ax, times=times)
        ax.set_title(subject)
    fig.tight_layout()
    group_rep.add_figs_to_section(fig, 'Individual GATs', 'Individual Plots',
                                  image_format=img)

    # group gat plot
    gg = np.mean(group_gat.values(), axis=0)
    ax = pretty_gat(scores=gg, chance=chance, sfreq=sfreq, times=times,
                    clim=clim)
    fig = ax.get_figure()
    ax.set_title('Group GAT scores on Processing ' + title)
    group_rep.add_figs_to_section(fig, 'Group GAT', 'Group Plots',
                                  image_format=img)

    # group gat corrected
    sig = group_dict['gat_sig']
    gg = np.mean(group_gat.values(), axis=0)
    sig_gg = np.ma.masked_where(sig == False, gg)

    ax = pretty_gat(scores=gg, chance=chance, sfreq=sfreq, times=times, sig=sig,
                    clim=clim, alpha=.5)
    ax.set_title('Group GAT corrected scores on Processing ' + title)
    fig = ax.get_figure()
    group_rep.add_figs_to_section(fig, 'Group GAT corrected', 'Group Plots',
                                  image_format=img)

    # group gat dev plot
    gat_dev = group_dict['gat_dev_sig']
    # tois = np.arange(7) * .1
    # time_idx = [rerf_diff.time_as_index(toi) for toi in tois]
    ax = plt.imshow(gat_dev, origin='lower', extent=[-.2, 1, -.2, 1])
    for val in [idx for interval in idx_time_sig for idx in interval]:
        plt.axvline(times[val], color='c')
        plt.axhline(times[val], color='c')
    fig = ax.get_figure()

    fig.suptitle('Group GAT dev on Processing ' + title)
    comments = list()
    for imin, imax in idx_time_sig:
        comments.append(('Time Bounds: (%d - %d ms), from Time Decoding.'
                         % (times[imin] * 1e3, times[imax] * 1e3)))
    comments.append(('Stats (in red) from GAT - diagonal after correction. '
                     'Blue means generalizing, while red means '
                     'different classifier.'))
    comments = ' '.join(comments)
    group_rep.add_figs_to_section(fig, 'Group GAT dev', 'Group Plots',
                                  comments=comments, image_format=img)

    ################
    # Group Slices #
    ################

    # group gat slices
    tois = np.arange(0, 3) * 1e-1
    gg = group_gat.values()
    fig, axes = plt.subplots(len(tois), 1, figsize=(5,10))
    ax = pretty_slices(gg, chance=chance, times=times, sfreq=sfreq,
                       tois=tois, axes=axes)
    fig.suptitle('Group GAT slices on Processing ' + title)

    # comment = ['0ms: %s' %]
    group_rep.add_figs_to_section(fig, 'Group Slices: 0-200ms', 'Group Plots',
                                  image_format=img)

    # group gat slices
    tois = np.arange(3, 7) * 1e-1
    gg = group_gat.values()
    fig, axes = plt.subplots(len(tois), 1, figsize=(5,10))
    ax = pretty_slices(gg, chance=chance, times=times, sfreq=sfreq,
                       tois=tois, axes=axes)
    fig.suptitle('Group GAT slices on Processing ' + title)
    group_rep.add_figs_to_section(fig, 'Group Slices: 300-600ms', 'Group Plots',
                                  image_format=img)

    # group gat slices
    tois = np.arange(0, 10) * 1e-1
    gg = group_gat.values()
    fig, axes = plt.subplots(10, 1, figsize=(5,20))
    ax = pretty_slices(gg, chance=chance, times=times, sfreq=sfreq,
                       tois=tois, axes=axes)
    fig.suptitle('Group GAT slices on Processing ' + title)
    group_rep.add_figs_to_section(fig, 'Group Slices: First 1s', 'Extra',
                                  image_format=img)

    ##################
    # Group Patterns #
    ##################
    # individual patterns
    tois = np.arange(0, 20) * .05
    idxs = np.array([rerf_diff.time_as_index(toi) for toi in tois]).ravel()
    group_patterns = list()
    for idx, toi in zip(idxs, tois):
        toi_pattern = list()
        for subject in subjects:
            gat = group_gat[subject]
            cov = group_cov[subject]
            # create pattern from data covariance matrix and the coefs/weights
            # idx = np.argmax(np.diag(gat))
            weights = group_weights[subject]
            weights = np.mean(weights[:,idx,:], axis=0)
            pattern = cov.dot(weights)
            toi_pattern.append(pattern)
        group_patterns.append(np.array(toi_pattern))

    # downsample the ind plots to select time points of interest
    tois_ind = [.1, .2, .3, .4, .5]
    idxs_ind = np.array([rerf_diff.time_as_index(toi) for toi in tois_ind]).ravel()
    for idx, toi, toi_pattern in zip(idxs_ind, tois_ind, group_patterns):
        fig, axes = plt.subplots(5, 4, figsize=(20,10))
        axes[-1, -1].axis('off')
        axes = [ax for axis in axes for ax in axis]
        for subject, pattern, ax in zip(subjects, toi_pattern, axes):
            plot_topomap(pattern, pos=pos, axes=ax)
            ax.set_title(subject)
        fig.tight_layout()
        group_rep.add_figs_to_section(fig, 'Individual Patterns at t=%s' %toi,
                                      'Individual Plots', image_format=img)

    group_patterns = np.array(group_patterns).mean(axis=1)
    fig, axes = plt.subplots(5, 4, figsize=(10,10))
    axes = [ax for axis in axes for ax in axis]
    for pattern, toi, ax in zip(group_patterns, tois, axes):
        title_pattern = 't=%s' % toi
        plot_topomap(pattern, pos=pos, axes=ax)
        ax.set_title(title_pattern)
    fig.tight_layout()
    group_rep.add_figs_to_section(fig, 'Group Patterns',
                                  'Group Plots', image_format=img)

    #######################
    # Group Time Decoding #
    #######################
    group_diags = np.array([np.diag(group_gat[subject])
                           for subject in subjects])

    # individual td plots
    fig, axes = plt.subplots(5, 4, figsize=(20, 10))
    axes[-1, -1].axis('off')
    axes = [ax for axis in axes for ax in axis]
    for subject, diag, ax in zip(subjects, group_diags, axes):
        pretty_decod(scores=diag, chance=chance, sfreq=sfreq, ax=ax, times=times)
        ax.set_title(subject)
    fig.tight_layout()
    group_rep.add_figs_to_section(fig, 'Individual TDs', 'Individual Plots',
                                  image_format=img)

    #########
    # Extra #
    #########
    comments = list()
    for imin, imax in idx_time_sig:
        comments.append("Significant Time Region: %.3f to %.3f s"
                        % (times[imin], times[imax]))
    if not comments:
        comments = None
    else:
        comments = '<br>'.join(comments)
    ax = pretty_decod(group_diags, chance=chance, sfreq=sfreq, times=times,
                      fill=True, sig=td_sig, alpha=.3, fill_color='orange')
    fig = ax.get_figure()
    ax.set_title('Group TD scores on Processing ' + title)
    group_rep.add_figs_to_section(fig, 'Group Time Decoding', 'Extra',
                                  image_format=img, comments=comments)

    # patterns for sig region
    group_patterns = list()
    for imin, imax in idx_time_sig:
        toi_pattern = list()
        for subject in subjects:
            gat = group_gat[subject]
            cov = group_cov[subject]
            # create pattern from data covariance matrix and the coefs/weights
            weights = group_weights[subject]
            weights = np.mean(weights[:,imin:imax,:], axis=(0,1))
            pattern = cov.dot(weights)
            toi_pattern.append(pattern)
        toi_pattern = np.array(toi_pattern).mean(axis=0)
        group_patterns.append(toi_pattern)

    if group_patterns:
        fig, axes = plt.subplots(1, len(group_patterns), figsize=(10,10))
        if not isinstance(axes, (np.ndarray, np.generic, list)):
            axes = [axes]
        for pattern, (imin, imax), ax in zip(group_patterns, idx_time_sig, axes):
            tmin, tmax = times[imin], times[imax]
            title_pattern = 'Averaged Weights ({:0.1f} - {:0.1f} ms)'.format(tmin, tmax)
            plot_topomap(pattern, pos=pos, axes=ax)
            ax.set_title(title_pattern)
        fig.tight_layout()
        group_rep.add_figs_to_section(fig, 'Group Patterns Sig Time Region',
                                      'Extra', image_format=img)

    #################################
    # Group Pattern + Time Decoding #
    #################################
    # initialize figure
    for imin, imax in idx_time_sig:
        fig, ax_topo = plt.subplots(1, 1, figsize=(16, 3))
        caption = 'Group Time Decoding'
        fig.suptitle(caption, fontsize=14)

        # patterns for sig region
        group_patterns = list()

        for subject in subjects:
            gat = group_gat[subject]
            cov = group_cov[subject]
            # create pattern from data covariance matrix and the coefs/weights
            weights = group_weights[subject]
            weights = np.mean(weights[:,imin:imax,:], axis=(0,1))
            pattern = cov.dot(weights)
            group_patterns.append(pattern)
        group_patterns = np.array(group_patterns).mean(axis=0)

        # 1. topomap
        tmin, tmax = times[imin], times[imax]
        title_pattern = 'Averaged Weights ({:0.1f} - {:0.1f} ms)'.format(tmin, tmax)
        image, _ = plot_topomap(group_patterns, pos=pos, axes=ax_topo)
        ax_topo.set_xlabel(title_pattern)

        # 2. colorbar
        divider = make_axes_locatable(ax_topo)
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)

        # 3. time decoding
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)

        comments = ("Significant Time Region: %.3f to %.3f s"
                    % (times[imin], times[imax]))
        ax = pretty_decod(group_diags, chance=chance, sfreq=sfreq, times=times,
                          ax=ax_signals, sig=td_sig, alpha=.3,
                          fill=True, fill_color='orange')

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        figs.append(fig)
        captions.append(title)

        group_rep.add_figs_to_section(fig, caption, comments=comments,
                                      section='Group Plots')

    return group_rep
