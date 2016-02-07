import sys
import os
import os.path as op
import warnings
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

import mne
from mne.report import Report
from mne.decoding import GeneralizationAcrossTime
from mne.stats import (linear_regression, linear_regression_raw,
                       spatio_temporal_cluster_test as stc_test,
                       spatio_temporal_cluster_1samp_test as stc_1samp_test)
from mne.channels import read_ch_connectivity
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, ShuffleSplit

import config

# parameters
path = config.drive
filt = config.filt
img = config.img
exp = 'OLDT'
analysis = 'priming_sensor_analysis'
random_state = 42
decim = 5
# decoding parameters
tmin, tmax = -.2, 1
# smoothing window
# length = 5. * decim * 1e-3
# step = 5. * decim * 1e-3
length = decim * 1e-3
step = decim * 1e-3
event_id = {'word/target/unprimed': 2, 'word/target/primed': 6,}
reject = config.reject


# setup group
fname_group = op.join(config.results_dir, 'group', 'group_OLDT_%s_filt_%s.html'
                      % (filt, analysis))
group_rep = Report()
group_scores = list()
# group priming t-vals
group_reg = list()

for subject in config.subjects:
    print config.banner % subject
    # define filenames
    fname_rep = op.join(config.results_dir, subject,
                        '%s_%s_%s.html' % (subject, exp, analysis))
    fname_proj = op.join(path, subject, 'mne', '%s_%s_calm_%s_filt-proj.fif'
                         % (subject, exp, filt))
    fname_raw = op.join(path, subject, 'mne',
                        '{}_{}_calm_{}_filt-raw.fif'.format(subject, exp, filt))
    fname_evts = op.join(path, subject, 'mne',
                         subject + '_{}-eve.txt'.format(exp))
    rep = Report()

    # loading events and raw
    evts = mne.read_events(fname_evts)
    raw = mne.io.read_raw_fif(fname_raw, preload=True, verbose=False)
    # add/apply proj
    proj = mne.read_proj(fname_proj)
    raw.add_proj(proj)
    raw.apply_proj()
    # select only meg channels
    raw.pick_types(meg=True)

    # create epochs for gat
    epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax, baseline=None,
                        decim=decim, reject=reject, preload=True, verbose=False)
    freqs = np.arange(1, 20, 2)
    cycles = freqs / 2.
    tfr = mne.time_frequency.tfr_morlet(epochs, freqs, cycles)
    asdf
    # # # currently disabled because of the HED
    # # epochs.equalize_event_counts(['unprimed', 'primed'], copy=False)
    # # plotting grand average
    # p = epochs.average().plot(show=False)
    # comment = ("This is a grand average over all the target epochs after "
    #            "equalizing the numbers in the priming condition.<br>"
    #            'unprimed: %d, and primed: %d, out of 96 possible events.'
    #            % (len(epochs['unprimed']), len(epochs['primed'])))
    # rep.add_figs_to_section(p, '%s: Grand Average on Target' % subject,
    #                       'Summary', image_format=img, comments=comment)

    # Convert the labels of the data to binary descriptors
    lbl = LabelEncoder()
    y = lbl.fit_transform(epochs.events[:,-1])

    print 'get ready for decoding ;)'
    train_times = {'start': tmin,
                   'stop': tmax,
                   'length': length,
                   'step': step
                   }

    # Generalization Across Time
    # default GAT: LogisticRegression with KFold (n=5)
    gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                                   train_times=train_times)
    gat.fit(epochs, y=y)
    group_scores.append(gat.score(epochs, y=y))
    fig = gat.plot(title='GAT Decoding Score on Semantic Priming: '
                   'Unprimed vs. Primed')
    rep.add_figs_to_section(fig, 'GAT Decoding Score on Priming',
                          'Decoding', image_format=img)
    fig = gat.plot_diagonal(title='Time Decoding Score on Semantic Priming: '
                            'Unprimed vs. Primed')
    rep.add_figs_to_section(fig, 'Time Decoding Score on Priming',
                          'Decoding', image_format=img)

    rep.save(fname_rep, open_browser=False, overwrite=True)

# temp hack
group_gat = gat
group_gat.scores_ = np.mean(group_scores, axis=0)

fig = gat.plot(title='GAT Decoding Score on Semantic Priming: '
               'Unprimed vs. Primed')
group_rep.add_figs_to_section(fig, 'GAT Decoding Score on Priming',
                              'Decoding', image_format=img)
fig = gat.plot_diagonal(title='Time Decoding Score on Semantic Priming: '
                        'Unprimed vs. Primed')
group_rep.add_figs_to_section(fig, 'Time Decoding Score on Priming',
                              'Decoding', image_format=img)
group_rep.save(fname_group, open_browser=False, overwrite=True)


###########################################
# run a spatio-temporal linear regression #
###########################################
group_reg = np.array(group_reg)
connectivity, ch_names = read_ch_connectivity('KIT-208')

threshold = 1.96
p_accept = 0.05
cluster_stats = stc_1samp_test(group_reg, n_permutations=10000,
                               threshold=threshold, tail=0,
                               connectivity=connectivity)
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

#################
# Visualization #
#################
# configure variables for visualization
condition_names = ['primed', 'unprimed']
times = epochs.times * 1e3
colors = 'r', 'steelblue'
linestyles = '-', '-'


# get sensor positions via layout
pos = mne.find_layout(epochs.info).pos

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
    signals = [primed.data[ch_inds, ...].mean(axis=0),
               unprimed.data[ch_inds, ...].mean(axis=0)]
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((t_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(16, 3))
    title = 'Cluster #{0}'.format(i_clu + 1)
    fig.suptitle(title, fontsize=14)

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(t_map, pos, mask=mask, axis=ax_topo,
                            cmap='Reds', vmin=np.min, vmax=np.max)

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
    for signal, name, ls, color in zip(signals, condition_names, linestyles, colors):
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
group_rep.save('/Users/teon/Desktop/test-st.html')
