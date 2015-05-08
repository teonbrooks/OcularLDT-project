from mpl_toolkits.axes_grid1 import make_axes_locatable
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

for subject in config.subjects:
    print subject
    path = op.join(config.data_dir, subject, 'mne')
    for exp in config.subjects[subject]:
        print exp
        r = Report()
        report_path = op.join(op.dirname(file), '..', 'output', 'results',
                              subject, 'meg', '%s_%s_ica-report.html'
                              % (subject, exp))
        # ecg, eog = 0, 1
        raw_file = op.join(path, '%s_%s_calm_lp40-raw.fif' % (subject, exp))
        proj_file = raw_file[:-18] + '-proj.fif'
        event_file = op.join(path, '%s_%s-eve.txt' % (subject, exp))

        evts = mne.read_events(event_file)
        raw = mne.io.Raw(raw_file)
        raw.info['bads'] = config.bads[subject]

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
        
        # fastica is used to fix the state
        ica = mne.preprocessing.ICA(.9, random_state=42, method='fastica')
        ica.fit(raw, decim=4)

        # # remove ecg
        # title = 'Sources related to %s artifacts (red)'
        # picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
        #                        stim=False, exclude='bads')
        # ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)
        # ica.fit(raw, picks=picks, decim=decim, reject=dict(mag=3e-12))
        # ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        # p = ica.plot_sources(raw, show_picks, exclude=ecg_inds, title=title % 'ecg',
        #                      show=False)
        # report.add_figs_to_section(p, 'Sources related to ECG artifact', 'ECG',
        #                            image_format=img)
        # p = ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True,
        #                         show=False)
        # report.add_figs_to_section(p, 'Topos of Sources related to ECG artifact',
        #                            'ECG', image_format=img)
        # ecg_inds = ecg_inds[:n_max_ecg]
        # ica.exclude += ecg_inds

        # plot ICs
        p = ica.plot_components(show=False)
        caption = ['Independent Components']
        comments = ['']
        section = 'Summary'
        if len(p) > 1:
            caption = caption * len(p)
            comments = comments * len(p)
        r.add_figs_to_section(p, caption, section, comments=comments,
                              image_format=img)
        p = ica.plot_sources(raw, start=10000, stop=20000, show=False)
        r.add_figs_to_section(p, 'ICs over 10s', 'Summary', image_format=img)

        # plot 
        evokeds = list()
        for i in range(ica.n_components_):
            ica.exclude = [i]
            evokeds.append(ica.apply(evoked, copy=True))

        for i, ev in enumerate(evokeds):
            p = ica.plot_components(i, show=False)
            # fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
            # divider = make_axes_locatable(ax_topo)
            # ax_topo = divider.append_axes('right', size='10%', pad=1.2)
            # ax_topo.axes = p.axes
            # ax_signals = divider.append_axes('right', size='300%', pad=1.2)
            e = ev.plot(titles={'mag': 'IC %d' % i}, show=False)  # axes=ax_topo

            r.add_figs_to_section([e, p], ['Evoked without IC %d' %i, 'IC %d' %i],
                                  'IC %d' %i, image_format=img)

        # ica.exclude = [1,2]
        # ica.save(ica_file)
        # ev = ica.apply(evoked, copy=True)
        # p = ev.plot(titles={'mag': 'IC %d and %d' % tuple(ica.exclude)}, show=False);
        # report.add_figs_to_section(p, 'Evoked-ICA over all events', 'Summary',
        #                            image_format=img)
        #
        # del epochs
        # ica.apply(raw)
        # epochs = mne.Epochs(raw, evts, None, tmin=-.1, tmax=.5, decim=4,
        #                     baseline=(None,0), reject={'mag': 3e-12}, verbose=False)
        # evoked = epochs.average()
        # cov = mne.compute_covariance(epochs[::10], tmin=-.1, tmax=0, method='auto',
        #                              verbose=False)
        # p = cov.plot(raw.info, show=False)[0]
        # report.add_figs_to_section(p, 'Covariance Matrix After ICA', 'Summary',
        #                            image_format=img)
        #
        # p = evoked.plot_white(cov, show=False)
        # report.add_figs_to_section(p, 'Whitened Evoked after ICA over all events',
        #                            'Summary', image_format=img)

        r.save(report_path, overwrite=True)
