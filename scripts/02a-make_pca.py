import mne
import os.path as op
from mne.report import Report
import config


try:
    file = __file__
except NameError:
    file = '/Applications/packages/E-MEG/scripts/make_pca.py'
layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = 'local'


for subject in config.subjects:
    print subject
    exp = 'OLDT'
    path = op.join(config.drives[drive], subject, 'mne')

    r = Report()
    report_path = op.join(op.dirname(file), '..', 'output', 'results',
                          subject, '%s_%s_filt_pca-report.html' % (subject, exp))

    ep_fname = op.join(path, '%s_%s_ica_calm_filt-epo.fif'
                       % (subject, exp))
    epochs = mne.read_epochs(ep_fname)
    epochs.pick_types(meg=True, exclude='bads')

    proj_file = op.join(path, op.basename(ep_fname)[:10] + '-proj.fif')
    event_file = op.join(path, '%s_%s-eve.txt' % (subject, exp))

    evts = mne.read_events(event_file)

    # plot evoked
    evoked = epochs.average()
    p = evoked.plot(titles={'mag': 'Original Evoked'}, show=False)
    r.add_figs_to_section(p, 'Original Evoked Response to Prime Word',
                          'Summary', image_format=img)
    p = epochs.plot_drop_log(show=False)
    r.add_figs_to_section(p, 'Drop log of the events', 'Summary',
                          image_format=img, scale=1)

    # plot covariance and whitened evoked
    ep_proj = epochs.crop(-.1, .03, copy=True)
    ev_proj = ep_proj.average()
    cov = mne.compute_covariance(epochs, tmin=-.2, tmax=-.1, method='auto',
                                 verbose=False)
    p = cov.plot(epochs.info, show_svd=0, show=False)[0]
    comments = ('The covariance matrix is computed on the -200:-100 ms '
                'baseline. -100:0 ms is confounded with the eye-mvt.')
    r.add_figs_to_section(p, 'Covariance Matrix', 'Summary',
                          image_format=img, comments=comments)
    p = evoked.plot_white(cov, show=False)
    r.add_figs_to_section(p, 'Whitened Evoked to Prime Word', 'Summary',
                          image_format=img)

    # compute the SSP
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

        # plot new cov and whitened evoked
        cov = mne.compute_covariance(epochs, projs=projs[i],
                                     tmin=-.2, tmax=-.1, method='auto',
                                     verbose=False)
        p = cov.plot(ev.info, show_svd=False, show=False)[0]
        r.add_figs_to_section(p, 'Covariance Matrix After PCA Removal',
                              pca, image_format=img)
        p = ev.plot_white(cov, show=False)
        r.add_figs_to_section(p, 'Whitened Evoked after PCA Removal',
                              pca, image_format=img)

    r.save(report_path, overwrite=True, open_browser=False)

    # add proj to raw
    mne.write_proj(proj_file, projs)
