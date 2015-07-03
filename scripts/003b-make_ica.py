import mne
import os.path as op
from mne.report import Report
import config


img = config.img
drive = config.drive
exp = 'OLDT'
filt = config.filt
redo = config.redo


for subject in config.subjects:
    print config.banner % subject

    # define filenames
    path = op.join(drive, subject, 'mne')
    fname_rep = op.join(config.results_dir, subject,
                        subject + '_%s_%s_filt_ica-report.html'
                        % (exp, filt))
    fname_epo = op.join(path, subject + '_%s_xca_calm_%s_filt-epo.fif'
                        % (exp, filt))
    fname_ica = op.join(path, subject + '_%s_calm_%s_filt-ica.fif'
                        % (exp, filt))

    if not op.exists(fname_ica) or redo:
        rep = Report()
        epochs = mne.read_epochs(fname_epo)
        epochs.pick_types(meg=True, exclude='bads')

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Original Evoked'}, show=False)
        rep.add_figs_to_section(p, 'Original Evoked Response to Prime Word',
                                'Summary', image_format=img)

        # compute the ICA
        # epochs.crop(-.1, .03, copy=False)
        epochs.pick_types(meg=True, eeg=False, eog=False,
                          stim=False, exclude='bads')
        ica = mne.preprocessing.ICA(.9, random_state=42, method='infomax')
        ica.fit(epochs)

        # plot ICs
        p = ica.plot_components(show=False)
        caption = ['Independent Components']
        comments = ['']
        section = 'Summary'
        if len(p) > 1:
            caption = caption * len(p)
            comments = comments * len(p)
        rep.add_figs_to_section(p, caption, section, comments=comments,
                              image_format=img)

        # plot 
        evokeds = list()
        for i in range(ica.n_components_):
            ica.exclude = [i]
            evokeds.append(ica.apply(evoked, copy=True))

        for i, ev in enumerate(evokeds):
            p = ica.plot_components(i, show=False)
            # fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
            # divider = make_axes_locatable(ax_topo)
            # ax_topo = dividerep.append_axes('right', size='10%', pad=1.2)
            # ax_topo.axes = p.axes
            # ax_signals = dividerep.append_axes('right', size='300%', pad=1.2)
            e = ev.plot(titles={'mag': 'IC %d' % i}, show=False)  # axes=ax_topo

            rep.add_figs_to_section([e, p], ['Evoked without IC %d' %i, 'IC %d' %i],
                                  'IC %d' %i, image_format=img)

        rep.save(fname_rep, overwrite=True, open_browser=False)

        # save ica
        ica.save(fname_ica)
