"""
001-make_ica.py

This script is used for the computation of the ICs to be removed for
each subject in the experiment. One approach to this analysis is to remove
the independent component that is related to the saccadic eye movement
in this experiment.
"""

import os.path as op
from pathlib import Path
import numpy as np
import toml

import mne
from mne.report import Report

from mne_bids import read_raw_bids, BIDSPath
from mne_bids import get_entity_vals


redo = True
derivative = 'ica'
evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']

cfg = toml.load(open(Path('./config.toml'), 'rb'))
task = cfg['task']
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')

fname_rep_group = op.join('output', 'reports', f'group_{task}-report.h5')

## some versioning change in either mne or h5io cause my h5 object to break
with mne.open_report(fname_rep_group) as rep_group:
    rep_group.title = f"{task} Group Report"
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        # define filenames
        fname_ica = op.join(cfg['bids_root'], f"sub-{subject}", cfg['datatype'],
                            f"sub-{subject}_task-{task}_{derivative}.fif")

        # ica input is from fixation cross to three hashes
        # no language involved
        raw = read_raw_bids(bids_path).pick_types(meg=True).load_data()
        events, total_event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in total_event_id.items()
                    if key in evts_labels}

        epo_tmin, epo_tmax = -.1, .1
        reject = dict(mag=3e-12)
        epochs = mne.Epochs(raw, events, event_id, tmin=epo_tmin, tmax=epo_tmax,
                            baseline=None, reject=reject, verbose=False,
                            preload=True)

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

        # here, we're only remove the top offender
        ica.exclude = [ica_idx[0]]
        p = rep_group.add_ica(ica, title=f"{subject} {derivative}", inst=epochs,
                              picks=ica_idx, eog_evoked=epochs.average(),
                              eog_scores=itc_score, tags=(derivative,))

        # TODO: make a grid of the excluded ICAs across all subjects
        # p = ica.plot_components(ica.exclude)

        ica.save(fname_ica, overwrite=redo)

    rep_group.save(fname_rep_group % 'html', open_browser=False, overwrite=redo)
