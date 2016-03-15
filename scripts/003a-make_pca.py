import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import mne
from mne.report import Report
import config


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'
filt = config.filt
redo = True
reject = config.reject
baseline = (-.2, -.1)
tmin, tmax = -.5, 1
ylim = dict(mag=[-300, 300])
event_id = {'word/prime/unprimed': 1,
            'word/prime/primed': 5,
            'nonword/prime': 9,
           }

fname_rep_group = op.join(config.results_dir, 'group',
                          'group_%s_%s_filt_pca-report.html' % (exp, filt))
rep_group = Report()
for subject in config.subjects:
    print config.banner % subject

    # define filenames
    path = op.join(drive, subject, 'mne')
    fname_raw = op.join(path, subject + '_%s_calm_%s_filt-raw.fif'
                        % (exp, filt))
    fname_evts = op.join(path, subject + '_{}-eve.txt'.format(exp))
    fname_proj = op.join(path, subject + '_%s_calm_%s_filt-proj.fif'
                         % (exp, filt))

    if not op.exists(fname_proj) or redo:
        # pca input is from fixation cross to three hashes
        # no language involved
        raw = mne.io.read_raw_fif(fname_raw)
        events = mne.read_events(fname_evts)
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1,
                            baseline=baseline, reject=reject, verbose=False)

        # compute the SSP
        evoked = epochs.average()
        ev_proj = evoked.crop(-.1, .03, copy=True)
        projs = mne.compute_proj_evoked(ev_proj, n_mag=3)

        # apply projector individually
        evokeds = [evoked.copy().add_proj(proj).apply_proj() for proj in projs]

        # 1. plot before and after summary
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(1, 2)
        axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]
        # plot evoked
        evoked.plot(titles={'mag': 'Before: Original Evoked'}, show=False,
                    axes=axes[0], ylim=ylim)
        # remove all
        evoked_proj = evoked.copy().add_proj(projs).apply_proj()
        evoked_proj.plot(titles={'mag': 'After: Evoked - All PCs'}, show=False,
                         axes=axes[1], ylim=ylim)
        rep_group.add_figs_to_section(fig, '%s: Before and After PCA: Evoked'
                                      % subject, 'Before and After All',
                                      image_format=img)

        # 2. plot PCA topos
        p = mne.viz.plot_projs_topomap(projs, layout, show=False)
        rep_group.add_figs_to_section(p, '%s: PCA topos' % subject,
                                      'Topos', image_format=img)

        # 3. plot evoked - each proj
        for ii, ev in enumerate(evokeds):
            exp_var = ev.info['projs'][0]['explained_var'] * 100
            title = 'PC %d: %2.2f%% Explained Variance' % (ii, exp_var)
            tab = 'PC %d' % ii
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]
            e = ev.plot(titles={'mag': title}, ylim=ylim,
                        show=False, axes=axes[0])
            p = mne.viz.plot_projs_topomap(ev.info['projs'], layout,
                                           show=False, axes=axes[1])
            rep_group.add_figs_to_section(fig, '%s: Evoked w/o PC %d'
                                          % (subject, ii), tab,
                                          image_format=img)

        # save projs
        mne.write_proj(fname_proj, projs)
        # cleanup
        del epochs
rep_group.save(fname_rep_group, overwrite=True, open_browser=False)
