import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.report import Report

from mne_bids import read_raw_bids
from mne_bids.read import _handle_events_reading
from mne_bids.utils import get_entity_vals


layout = mne.channels.read_layout('KIT-AD.lout')
img_ext = 'png'
task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'teon-backup', 'Experiments', task)

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

for subject in subjects_list:
    print(banner % subject)

    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    events_fname = op.join(path, f"sub-{subject}_task-{task}_events.tsv")
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_split-01_meg.fif")
    fname_ica = op.join(path, f"sub-{subject}_task-{task}_ica.fif")

    if not op.exists(fname_ica) or redo:
        # pca input is from fixation cross to three hashes
        # no language involved
        try:
            raw = mne.io.read_raw_fif(fname_raw)
        except FileNotFoundError:
            raw = mne.io.read_raw_fif(fname_mp_raw)
        # TODO: replace with proper solution
        raw = _handle_events_reading(events_fname, raw)
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
        # let's find the most coherent over this time course
        time_course = (-.1, .03)
        start, stop = epochs.time_as_index(time_course)
        # sum itc across time then sum across frequency
        itc_score = itc[start:stop].sum(axis=(1,2))
        # take the top three for comparison-sake
        ica_idx = (itc_score).argsort()[::-1][:3]

        p = ica.plot_scores(itc_score)
        rep_group.add_figs_to_section(p, f'{subject}: IC scores',
                                      'IC Scores', image_format=img_ext)

        # plot ICs
        picks=range(ica.n_components_)
        p = ica.plot_sources(evoked)
        rep_group.add_figs_to_section(p, f'{subject}: Reconstructed IC sources peri-saccade',
                                      'Summary Time-locked ICs', image_format=img_ext)
        p = ica.plot_components(picks)
        rep_group.add_figs_to_section(p, f'{subject}: IC Topos',
                                      'Summary IC Topos',
                                      image_format=img_ext)

        for ii, idx in enumerate(ica_idx):
            tab = f'IC {idx}'
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]

            # TODO: Currently hacked mne to support axes args
            # make a pull request to allow for axes there
            ica.plot_sources(evoked, picks=idx, fig=fig, axes=axes[0])
            ica.plot_components(idx, fig=fig, axes=axes[1])
            fig.tight_layout()
            caption = (f'{subject}: IC {idx}')

            rep_group.add_figs_to_section(fig, caption, f'IC Sum(ITC) R-{ii}')
        ica.exclude = ica_idx
        ica.save(fname_ica)
    plt.close('all')

    rep_group.save(fname_rep_group, overwrite=True, open_browser=False)
