import mne
import os.path as op
from mne.report import Report
import config


img = config.img
drive = config.drive
exp = 'OLDT'
filt = config.filt
redo = config.redo
reject = {'mag': 3e-12}
baseline = (-.2, -.1)
tmin, tmax = -.2, 1
event_id = config.event_id

fname_rep = op.join(config.results_dir, 'group',
                    'group_%s_%s_filt_ica-report.html' % (exp, filt))
rep = Report()


for subject in config.subjects:
    print config.banner % subject
    if redo:
        # define filenames
        path = op.join(drive)


        # define filenames
        subject_template = op.join(path, subject, 'mne', subject + '_%s%s.%s')
        fname_raw = subject_template % (exp, '_calm_' + filt + '_filt-raw', 'fif')
        fname_evts = subject_template % (exp, '-eve', 'txt')
        evts = mne.read_events(fname_evts)

        # load from raw
        raw = mne.io.read_raw_fif(fname_raw, preload=False, verbose=False)

        # target eye-movements
        epochs = mne.Epochs(raw, evts, event_id, tmin=tmin, tmax=tmax,
                            reject=reject, preload=False, verbose=False)
        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Original Evoked'}, show=False)
        rep.add_figs_to_section(p, 'Original Evoked Response to Saccade',
                                'Summary', image_format=img)
        # epochs.crop(-.1, .03, copy=False)

        # compute the ICA
        ica = mne.preprocessing.ICA(.9, random_state=42, method='fastica')
        ica.fit(raw, decim=4)

        ics = ica.get_sources(epochs.copy().load_data())
        picks=range(len(ics.info['chs']))
        p = ica.plot_sources(evoked)
        rep.add_figs_to_section(p, 'IC Evoked Response to Saccade',
                                'Summary', image_format=img)
        # p = ica.plot_components()
        rep.add_figs_to_section(p, 'IC Topos', 'Summary', image_format=img)

        rep.save(fname_rep, overwrite=True)
        asdf


        ics.crop(-.03, .03, copy=False)
        ics_mean = ics.average()
        ics_std_error = ics.standard_error()
        ics_wald = ics_mean / ics_std_error
        ics_

        import numpy as np
        freqs = np.arange(4, 30, 2)
        cycles = freqs / 3.
        tfr = mne.time_frequency.tfr_morlet(ics, freqs, cycles)
        # # plot ICs
        # p = ica.plot_components(show=False)
        # caption = ['Independent Components']
        # comments = ['']
        # section = 'Summary'
        # if len(p) > 1:
        #     caption = caption * len(p)
        #     comments = comments * len(p)
        # rep.add_figs_to_section(p, caption, section, comments=comments,
        #                       image_format=img)
        #
        # # plot
        # evokeds = list()
        # for i in range(ica.n_components_):
        #     ica.exclude = [i]
        #     evokeds.append(ica.apply(evoked, copy=True))
        #
        # for i, ev in enumerate(evokeds):
        #     p = ica.plot_components(i, show=False)
        #     # fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
        #     # divider = make_axes_locatable(ax_topo)
        #     # ax_topo = dividerep.append_axes('right', size='10%', pad=1.2)
        #     # ax_topo.axes = p.axes
        #     # ax_signals = dividerep.append_axes('right', size='300%', pad=1.2)
        #     e = ev.plot(titles={'mag': 'IC %d' % i}, show=False)  # axes=ax_topo
        #
        #     rep.add_figs_to_section([e, p], ['Evoked without IC %d' %i, 'IC %d' %i],
        #                           'IC %d' %i, image_format=img)
        #
        # rep.save(fname_rep, overwrite=True, open_browser=False)
        #
        # # save ica
        # ica.save(fname_ica)
