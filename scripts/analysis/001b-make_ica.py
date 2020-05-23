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

evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']
subjects_list = get_entity_vals(bids_root, entity_key='sub')

fname_rep_group = op.join('..', '..', 'output',
                          'group', f'group_{task}_ica-report.html')
rep_group = Report()

for subject in subjects_list:
    print("#" * 9 + f"\n# {subject} #\n" + "#" * 9)

    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    events_fname = op.join(path, f"sub-{subject}_task-{task}_events.tsv")
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_split-01_meg.fif")
    fname_ica = op.join(path, f"sub-{subject}_task-{task}_ica.fif")


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

    epo_tmin, epo_tmax = -.1, .1
    reject = dict(mag=3e-12)
    epochs = mne.Epochs(raw, events, event_id, tmin=epo_tmin, tmax=epo_tmax,
                        reject=reject, verbose=False, preload=True)

    # plot evoked
    evoked = epochs.average()
    p = evoked.plot(titles={'mag': 'Original Evoked'},
                    window_title=subject, show=False)
    rep_group.add_figs_to_section(p, f'{subject}: Evoked Response peri-saccade',
                                    'Summary Evokeds', image_format=img_ext)

    if not op.exists(fname_ica) or redo:
        # compute the ICA
        # TODO: is there a good heuristic for why .9?
        ica = mne.preprocessing.ICA(.9, random_state=42, method='fastica')
        ica.fit(epochs)

    else:
        ica = mne.preprocessing.read_ica(fname_ica)

    # transform epochs to ICs
    epochs_ica = ica.get_sources(epochs)

    # compute the inter-trial coherence
    min_cycles = 1 / (epo_tmax - epo_tmin)
    itc = mne.time_frequency.tfr_array_morlet(epochs_ica.get_data(),
                                              epochs_ica.info['sfreq'],
                                              np.arange(min_cycles, 30),
                                              n_cycles=.1,
                                              output='itc')
    # let's find the most coherent over this time course
    # TODO: find a source for time duration of saccade.
    itc_tmin, itc_tmax = -.1, .03
    start, stop = epochs_ica.time_as_index((itc_tmin, itc_tmax))
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
    rep_group.add_figs_to_section(p, f'{subject}: IC t.s. peri-saccade',
                                    'Summary Time-locked ICs', image_format=img_ext)
    p = ica.plot_components(picks)
    rep_group.add_figs_to_section(p, f'{subject}: IC Topos',
                                    'Summary IC Topos',
                                    image_format=img_ext)

    for ii, idx in enumerate(ica_idx):
        fig = ica.plot_properties(epochs, picks=idx)
        caption = (f'{subject}: IC {idx}')
        rep_group.add_figs_to_section(fig, caption, f'IC Sum(ITC) Rank-{ii}')

    ica.exclude = ica_idx
    ica.save(fname_ica)
    plt.close('all')

    rep_group.save(fname_rep_group, overwrite=True, open_browser=False)
