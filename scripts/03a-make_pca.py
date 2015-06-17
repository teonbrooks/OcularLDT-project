import mne
import os.path as op
from mne.report import Report
import config


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'


for subject in config.subjects:
    print subject
    path = op.join(drive, subject, 'mne')
    ep_fname = op.join(path, '%s_%s_ica_calm_filt-epo.fif'
                       % (subject, exp))
    proj_fname = op.join(path, op.basename(ep_fname)[:10] + '-proj.fif')

    if not op.exists(proj_fname):
        r = Report()
        report_fname = op.join(op.expanduser('~'), 'Dropbox', 'academic', 
                               'Experiments', 'E-MEG', 'output', 'results',
                               subject, '%s_%s_filt_pca-report.html' % (subject, exp))

        epochs = mne.read_epochs(ep_fname)
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

        r.save(report_fname, overwrite=True, open_browser=False)

        # save projs
        mne.write_proj(proj_fname, projs)
