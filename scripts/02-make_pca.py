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

for subject in subjects:
    path = op.join(config.data_dir, subject, 'mne')
    for exp in config.subjects[subject]:
        r = Report()
        report_path = op.join(op.dirname(file), '..', 'output', 'results',
                              subject, 'meg', '%s_%s_pca-report.html'
                              % (subject, exp))
        # ecg, eog = 0, 1
        raw_file = op.join(path, '%s_%s_calm_lp40-raw.fif' % (subject, exp))
        proj_file = raw_file[:-18] + '-proj.fif'
        event_file = op.join(path, '%s_%s-eve.txt' % (subject, exp))

        evts = mne.read_events(event_file)
        raw = mne.io.Raw(raw_file)
        raw.info['bads'] = ['MEG 130', 'MEG 072', 'MEG 087', 'MEG 121']

        # plot evoked
        epochs = mne.Epochs(raw, evts, {'prime': 5}, tmin=-.2, tmax=.5,
                            decim=4, baseline=(None,0), reject=config.reject,
                            verbose=False, preload=True)
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Original Evoked'}, show=False)
        r.add_figs_to_section(p, 'Original Evoked Response to Prime Word',
                              'Summary', image_format=img)
        p = epochs.plot_drop_log(show=False)
        r.add_figs_to_section(p, 'Drop log of the events', 'Summary',
                              image_format=img, scale=1)

        # plot covariance and whitened evoked
        ep_proj = epochs.crop(-.1, 0, copy=True)
        # ep_proj = ep_proj[::10]
        cov = mne.compute_covariance(ep_proj, tmin=-.2, tmax=0, method='auto',
                                     verbose=False)
        p = cov.plot(epochs.info, show_svd=0, show=False)[0]
        r.add_figs_to_section(p, 'Covariance Matrix', 'Summary',
                              image_format=img)
        p = evoked.plot_white(cov, show=False)
        r.add_figs_to_section(p, 'Whitened Evoked to Prime Word', 'Summary',
                              image_format=img)

        # compute the SSP
        projs = mne.compute_proj_epochs(ep_proj, n_mag=10)

        # apply projector step-wise and get the std
        evokeds = list()
        baselines = list()
        for proj in projs:
            ev = evoked.copy()
            ev.add_proj(proj, remove_existing=True)
            ev.apply_proj()
            evokeds.append(ev)

            # baseline = evoked.copy()
            # baseline.add_proj(proj, remove_existing=True)
            # baseline.apply_proj()
            # baselines.append(baseline.data.std())

        # plot PCA topos
        p = mne.viz.plot_projs_topomap(projs, layout, show=False)
        r.add_figs_to_section(p, 'PCA topographies', 'Summary',
                              image_format=img)

        # plot evoked - each proj
        for i, ev in enumerate(evokeds):
            e = ev.plot(titles={'mag': 'PCA %d' % i}, show=False)
            p = mne.viz.plot_projs_topomap(ev.info['projs'], layout,
                                           show=False)
            r.add_figs_to_section([e, p], ['Evoked without PCA %d' %i,
                                           'PCA topography %d' %i],
                                  'PCA %d' % i, image_format=img)

        showed_interim = False
        # if not 'ecg' in locals():
        #     showed_interim = True
        #     r.save(report_path, overwrite=True)
        #     ecg = int(raw_input('ecg: '))
        if not 'eog' in locals():
            if showed_interim == False:
                r.save(report_path, overwrite=True)
            eog = raw_input('eog: ')
            if isinstance(eog, str):
                eog = int(eog)
                # plot evoked - bad projs
                bad_projs = [projs[eog]]
                bad_projs_idx = eog
                evoked.add_proj(bad_projs, remove_existing=True)

                p = evoked.plot(titles={'mag': 'PCA %d' % bad_projs_idx},
                                proj=True, show=False)
                r.add_figs_to_section(p, 'Evoked - EOG-related PC', 'Summary',
                                      image_format=img)

                # plot new cov and whitened evoked
                cov = mne.compute_covariance(ep_proj, projs=bad_projs,
                                             tmin=-.2, tmax=0, method='auto',
                                             verbose=False)
                p = cov.plot(raw.info, show_svd=False, show=False)[0]
                r.add_figs_to_section(p, 'Covariance Matrix After PCA Removal',
                                      'Summary', image_format=img)
                p = evoked.plot_white(cov, show=False)
                r.add_figs_to_section(p, 'Whitened Evoked after PCA removal',
                                      'Summary', image_format=img)
                r.save(report_path, overwrite=True, open_browser=False)

                # add proj to raw
                mne.write_proj(proj_file, bad_projs)
            elif eog == 'None':
                r.save(report_path, overwrite=True, open_browser=False)
            else:
                raise TypeError('Response must be int or None, not %s'
                                % type(eog))
            del eog
