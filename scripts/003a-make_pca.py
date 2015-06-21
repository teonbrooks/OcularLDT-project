import mne
import os.path as op
from mne.report import Report
import config


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'
filt = config.filt


for subject in config.subjects:
    print config.banner % subject

    # define filenames
    path = op.join(drive, subject, 'mne')
    fname_rep = op.join(config.results_dir, subject,
                        subject + '_%s_%s_filt_pca-report.html'
                        % (exp, filt))
    fname_epo = op.join(path, subject + '_%s_xca_calm_%s_filt-epo.fif'
                        % (exp, filt))
    fname_proj = op.join(path, subject + '_%s_xca_calm_%s_filt-proj.fif'
                         % (exp, filt))

    if not op.exists(proj_fname):
        r = Report()
        epochs = mne.read_epochs(fname_epo)
        epochs.pick_types(meg=True, exclude='bads')

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Original Evoked'}, show=False)
        r.add_figs_to_section(p, 'Original Evoked Response to Prime Word',
                              'Summary', image_format=img)

        # compute the SSP
        epochs.crop(-.1, .03, copy=False)
        ev_proj = epochs.average()
        projs = mne.compute_proj_evoked(ev_proj, n_mag=2)

        # apply projector step-wise
        evokeds = list()
        for proj in projs:
            ev = evoked.copy()
            ev.add_proj(proj, remove_existing=True)
            ev.apply_proj()
            evokeds.append(ev)

        # plot PCA topos
        p = mne.viz.plot_projs_topomap(projs, layout, show=False)
        r.add_figs_to_section(p, 'PCA topographies', 'Summary',
                              image_format=img)

        # plot evoked - each proj
        for i, ev in enumerate(evokeds):
            pca = 'PCA %d' % i
            e = ev.plot(titles={'mag': 'PCA %d' % i}, show=False)
            p = mne.viz.plot_projs_topomap(ev.info['projs'], layout,
                                           show=False)
            r.add_figs_to_section([e, p], ['Evoked without PCA %d' %i,
                                           'PCA topography %d' %i],
                                  pca, image_format=img)
        r.save(fname_rep, overwrite=True, open_browser=False)

        # save projs
        mne.write_proj(fname_proj, projs)
