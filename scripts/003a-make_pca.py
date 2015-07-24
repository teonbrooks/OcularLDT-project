import mne
import os.path as op
from mne.report import Report
import config
import matplotlib.pyplot as plt
from matplotlib import gridspec


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'
filt = config.filt
redo = config.redo
reject = config.reject
baseline = (-.2, -.1)
tmin, tmax = -.2, .6
ylim = dict(mag=[-200, 200])

fname_rep_group = op.join(config.results_dir, 'group',
                          'group_%s_%s_filt_pca-report.html' % (exp, filt))
rep_group = Report()
for subject in config.subjects:
    print config.banner % subject

    # define filenames
    path = op.join(drive, subject, 'mne')
    fname_rep = op.join(config.results_dir, subject,
                        subject + '_%s_%s_filt_pca-report.html'
                        % (exp, filt))
    fname_epo = op.join(path, subject + '_%s_xca_calm_%s_filt-epo.fif'
                        % (exp, filt))
    fname_proj = op.join(path, subject + '_%s_calm_%s_filt-proj.fif'
                         % (exp, filt))

    if not op.exists(fname_proj) or redo:
        rep = Report()
        # pca input is from fixation cross to three hashes
        # no language involved
        epochs = mne.read_epochs(fname_epo, baseline=baseline)['prime']
        epochs.pick_types(meg=True, exclude='bads')
        epochs.drop_bad_epochs(reject=reject)

        # compute the SSP
        evoked = epochs.average()
        ev_proj = epochs.crop(-.1, .03, copy=True).average()
        projs = mne.compute_proj_evoked(ev_proj, n_mag=3)

        # apply projector individually
        evokeds = [evoked.copy().add_proj(proj).apply_proj() for proj in projs]

        # plot PCA topos
        p = mne.viz.plot_projs_topomap(projs, layout, show=False)
        rep.add_figs_to_section(p, 'PCA topographies', 'Summary',
                                image_format=img)
        rep_group.add_figs_to_section(p, '%s: PCA topographies' % subject,
                                      subject, image_format=img)

        # plot evoked - each proj
        for i, ev in enumerate(evokeds):
            pca = 'PC %d' % i
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]
            e = ev.plot(titles={'mag': 'PC %d' % i}, ylim=ylim,
                        show=False, axes=axes[0])
            p = mne.viz.plot_projs_topomap(ev.info['projs'], layout,
                                           show=False, axes=axes[1])
            rep.add_figs_to_section(fig, 'Evoked without PC %d' %i,
                                    pca, image_format=img)

        # plot before and after summary
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
        rep.add_figs_to_section(fig, 'Before and After PCA: Evoked Response '
                                'to Prime Word', 'Summary', image_format=img)
        rep_group.add_figs_to_section(fig, '%s: Before and After PCA: Evoked '
                                      'Response to Prime Word' % subject,
                                      subject, image_format=img)

        rep.save(fname_rep, overwrite=True, open_browser=False)

        # save projs
        mne.write_proj(fname_proj, projs)
        # cleanup
        del epochs
rep_group.save(fname_rep_group, overwrite=True, open_browser=False)
