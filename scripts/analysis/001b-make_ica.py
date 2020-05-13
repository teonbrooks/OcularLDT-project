import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.report import Report
from mne_bids import read_raw_bids
from mne_bids.utils import get_entity_vals


layout = mne.channels.read_layout('KIT-AD.lout')
img_ext = 'png'
task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'Experiments', task)

redo = True
reject = dict(mag=3e-12)
baseline = (-.2, -.1)
tmin, tmax = -.5, 1
ylim = dict(mag=[-300, 300])
banner = ('#' * 9 + '\n# %s #\n' + '#' * 9)
evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']
subjects_list = get_entity_vals(bids_root, entity_key='sub')

fname_rep_group = op.join('..', 'output',
                          'group', f'group_{task}_ica-report.html')
rep_group = Report()

for subject in subjects_list[:1]:
    print(banner % subject)

    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_part-01_meg.fif")
    fname_ica = op.join(path, f"sub-{subject}_task-{task}_ica.fif")

    if not op.exists(fname_ica) or redo:
        # pca input is from fixation cross to three hashes
        # no language involved
        try:
            raw = read_raw_bids(fname_raw, bids_root=bids_root)
        except FileNotFoundError:
            raw = read_raw_bids(fname_mp_raw, bids_root=bids_root)
        events, event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in event_id.items()
                    if key in evts_labels}
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=.1,
                            baseline=baseline, reject=reject, verbose=False)

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Original Evoked'},
                        window_title=subject, show=False)
        rep_group.add_figs_to_section(p, f'{subject}: Evoked Response peri-saccade',
                                      'Summary Evokeds', image_format=img_ext)

        ica_tmin, ica_tmax = -.1, .1
        min_cycles = 1 / (ica_tmax - ica_tmin)
        epochs.load_data().crop(ica_tmin, ica_tmax)

        # compute the ICA
        ica = mne.preprocessing.ICA(.9, random_state=42, method='fastica')
        ica.fit(epochs)

        # transform epochs to ICs
        epochs_ica = ica.get_sources(epochs)

        # compute the inter-trial coherence
        itc = mne.time_frequency.tfr_array_morlet(epochs_ica.get_data(),
                                                  epochs_ica.info['sfreq'],
                                                  np.arange(min_cycles, 30),
                                                  n_cycles=.1,
                                                  output='itc')


        # plot ICs
        picks=range(ica.n_components_)
        p = ica.plot_sources(evoked)
        rep_group.add_figs_to_section(p, f'{subject}: Reconstructed IC sources peri-saccade',
                                      'Summary Time-locked ICs', image_format=img_ext)
        p = ica.plot_components(picks)
        rep_group.add_figs_to_section(p, f'{subject}: IC Topos',
                                      'Summary IC Topos',
                                      image_format=img_ext)

        for i in range(ica.n_components_):
            p = ica.plot_sources(evoked, picks=[i], show=False)
            e = ica.plot_components(picks=[i], title={'mag': 'IC %d' % i},
                                    show=False)
            ic_name = 'ICA{:03d}'.format(i)
            rep_group.add_figs_to_section([e, p], [f'{subject}: {ic_name} Time Course',
                                                   f'{subject}: {ic_name}'],
                                          subject, image_format=img_ext)

        rep_group.save(fname_rep_group, overwrite=True, open_browser=False)
        #
        # # save ica
        # ica.save(fname_ica)
