import pickle
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.report import Report
from jr.plot import pretty_gat, pretty_decod

import config


# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
analysis = 'priming_sensor_analysis'
results_dir = config.results_dir
threshold = 1.96
p_accept = 0.05
c_names = ['word/target/primed', 'word/target/unprimed']

# setup group
group_template = op.join('%s', 'group', 'group_%s_%s_filt_%s.%s')
fname_group_rep = group_template % (results_dir, exp, filt, analysis, 'html')
fname_group_rerf = group_template % (path, exp, filt, analysis + '_rerf', 'mne')
fname_group_gat = group_template % (path, exp, filt, analysis + '_gat', 'mne')

subjects = config.subjects
group_gat = pickle.load(open(fname_group_gat))
group_rerf = pickle.load(open(fname_group_rerf))
group_rep = Report()

# add'l info
sfreq, times = group_gat['sfreq'], group_gat['times']

#####################
#Individual Reports #
#####################
for subject in subjects:
    rep = Report()
    subject_template = op.join(results_dir, subject, subject + '_%s_%s.%s')
    fname_rep = subject_template % (exp, analysis, 'html')
    # plot individual difference waves
    rerf = group_rerf[subject]
    rerf_diff = mne.evoked.combine_evoked([rerf[c_names[0]], rerf[c_names[1]]],
                                          weights=[1, -1])
    p = rerf_diff.plot(show=False)
    rep.add_figs_to_section(p, 'Difference Butterfly',
                            'Evoked Difference Comparison',
                            image_format=img)

    # plot individual gat matrix
    scores = group_gat[subject]
    ax = pretty_gat(scores=scores, chance=.5, sfreq=sfreq, times=times)
    fig = ax.get_figure()
    # fig = gat.plot(title='GAT Decoding Score on Processing Word vs. Nonword')
    rep.add_figs_to_section(fig, 'GAT Decoding Score on Word vs. Nonword',
                          'Decoding', image_format=img)
    scores = np.diag(scores)
    ax = pretty_decod(scores, chance=.5, sfreq=sfreq, times=times)
    fig = ax.get_figure()
    # fig = gat.plot_diagonal(title='Time Decoding on Processing Word '
    #                         'vs. Nonword')
    rep.add_figs_to_section(fig, 'Time Decoding Score on Processing Word '
                            'vs. Nonword', image_format=img)
    rep.save(fname_rep, open_browser=False, overwrite=True)

################
# Group Evoked #
################
group_c0 = mne.grand_average([group_rerf[subject][c_names[0]] for subject
                                in subjects])
group_c1 = mne.grand_average([group_rerf[subject][c_names[1]] for subject
                                   in subjects])

grand_averages = [group_c0, group_c1]

rerf_diff = mne.evoked.combine_evoked([grand_averages[0], grand_averages[1]],
                                      weights=[1, -1])
fig = rerf_diff.plot()
group_rep.add_figs_to_section(fig, 'Grand Average Difference', 'Evoked')

##############
# Group RERF #
##############
T_obs, clusters, p_values, _ = group_rerf['stats']
good_cluster_inds = np.where(p_values < p_accept)[0]


# configure variables for visualization
colors = 'r', 'steelblue'
linestyles = '-', '-'

# get sensor positions via layout
pos = group_rerf['layout'].pos

captions = list()
figs = list()
# loop over significant clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster infomation, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for T stat
    t_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at significant sensors
    signals = [evoked.data[ch_inds, ...].mean(axis=0) for evoked in
               grand_averages]
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((t_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(16, 3))
    title = 'Cluster #{0}'.format(i_clu + 1)
    fig.suptitle(title, fontsize=14)

    # plot average test statistic and mark significant sensors
    image, _ = mne.viz.plot_topomap(t_map, pos, mask=mask, axis=ax_topo,
                                    vmin=np.min, vmax=np.max)

    # advanced matplotlib for showing image with figure and colorbar
    # in one plot
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel('Averaged T-map ({:0.1f} - {:0.1f} ms)'.format(
        *sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    for signal, name, ls, color in zip(signals, c_names, linestyles, colors):
        ax_signals.plot(times, signal, label=name, linestyle=ls, color=color)

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

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    figs.append(fig)
    captions.append(title)

group_rep.add_figs_to_section(figs, captions, 'Spatio-temporal tests')

#############
# Group GAT #
#############
gat = group_gat['group']
group_scores = np.array([group_gat[subject] for subject in subjects])

# individual gat plots
# dim = np.ceil(np.sqrt(len(group_scores))).astype(int)
# fig, axes = plt.subplots(dim, dim, figsize=(20,10))
fig, axes = plt.subplots(5, 4, figsize=(20,10))
axes = [ax for axis in axes for ax in axis]
for subject, ax in zip(subjects, axes):
    pretty_gat(scores=group_gat[subject], chance=.5, sfreq=sfreq,
               ax=ax, times=times)
    # gat.scores_ = np.array(score)
    # gat.plot(ax=ax, show=False, xlabel=False, ylabel=False)
fig.tight_layout()
group_rep.add_figs_to_section(fig, 'Individual GATs', 'GAT',
                              image_format=img)

# group gat plot
T_obs, clusters, p_values, _ = group_gat['stats']
sig = p_values < .05
ax = pretty_gat(gat.scores_, chance=.5, sfreq=200, times=times)#, sig=sig)
fig = ax.get_figure()
# fig = gat.plot(title='Group GAT Decoding Score on Processing Word vs. Nonword')
group_rep.add_figs_to_section(fig, 'Group GAT', 'GAT', image_format=img)


#######################
# Group Time Decoding #
#######################
group_diags = np.array([np.diag(scores) for scores in group_scores])

# individual gat plots
# dim = np.ceil(np.sqrt(len(group_scores))).astype(int)
# fig, axes = plt.subplots(dim, dim, figsize=(20,10))
fig, axes = plt.subplots(5, 4, figsize=(20, 10))
axes = [ax for axis in axes for ax in axis]
for diag, ax in zip(group_diags, axes):
    pretty_decod(scores=diag, chance=.5, sfreq=sfreq, ax=ax, times=times)
    # gat.scores_ = np.array(score)
    # gat.plot(ax=ax, show=False, xlabel=False, ylabel=False)
fig.tight_layout()
group_rep.add_figs_to_section(fig, 'Individual TDs', 'Time Decoding',
                              image_format=img)

# group time decoding
group_diags = np.array([np.diag(scores) for scores in group_scores])
ax = pretty_decod(group_diags, chance=.5, sfreq=sfreq, times=times)
fig = ax.get_figure()
# fig = gat.plot_diagonal(title='Time Decoding on Processing Word vs. Nonword',
#                         chance=.5)
group_rep.add_figs_to_section(fig, 'Group Time Decoding', 'Time Decoding',
                              image_format=img)

group_rep.save(fname_group_rep, open_browser=False, overwrite=True)
