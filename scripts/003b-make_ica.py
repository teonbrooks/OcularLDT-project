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
tmin, tmax = -.2, .03


for subject in config.subjects:
    print config.banner % subject

    # define filenames
    path = op.join(drive)
    fname_rep = op.join(config.results_dir, subject,
                        subject + '_%s_%s_filt_ica-report.html'
                        % (exp, filt))

    if redo:
        rep = Report()

        # load from raw
        exps = config.subjects[subject]
        raw = config.kit2fiff(subject=subject, exp=exps[0],
                              path=path, preload=False)
        raw2 = config.kit2fiff(subject=subject, exp=exps[2],
                              path=path, dig=False, preload=False)
        mne.concatenate_raws([raw, raw2])
        raw.info['bads'] = config.bads[subject]
        raw.preload_data()
        if filt == 'fft':
            raw.filter(.1, 40, method=filt, l_trans_bandwidth=.05)
        else:
            raw.filter(1, 40, method=filt)

        # target eye-movements
        evts = mne.find_stim_steps(raw, merge=-2)
        epochs = mne.Epochs(raw, evts, None, tmin=tmin, tmax=tmax,
                            baseline=baseline, reject=reject,
                            preload=True, verbose=False)
        epochs.pick_types(meg=True, exclude='bads')

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Original Evoked'}, show=False)
        rep.add_figs_to_section(p, 'Original Evoked Response to Saccade',
                                'Summary', image_format=img)
        epochs.crop(-.1, .03, copy=False)

        # compute the ICA
        ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
        ica.fit(epochs, decim=4)

        ics = ica.get_sources(epochs)
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
