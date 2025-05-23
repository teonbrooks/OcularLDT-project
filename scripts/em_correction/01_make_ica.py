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
import tomllib as toml

import mne
from mne_bids import read_raw_bids, BIDSPath
from mne_bids import get_entity_vals

import matplotlib.pyplot as plt


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml', 'rb'))

redo = False
task = cfg['task']
derivative = 'ica'
evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']

bids_root = root / 'data' / task
subjects_list = get_entity_vals(bids_root, entity_key='subject')
bids_path = BIDSPath(root=bids_root, session=None, task=task,
                     datatype=cfg['datatype'], check=False)

fname_rep_group = op.join(root, 'output', 'reports',
                          f'group_{task}-report' + '{}')
fname_group_img = op.join(root, 'output', 'reports', derivative,
                          f'group_{task}-ica_' + '{:02d}.svg')

## some versioning change in either mne or h5io cause my h5 object to break
# pick types -> pick
n_subjects = len(subjects_list)
row, col = int(np.ceil(np.sqrt(n_subjects))), int(np.floor(np.sqrt(n_subjects)))

# Make group fig of the top three ICs from this method
group_fig, group_ax = list(), list()
for ii in range(3):
    fig, ax = plt.subplots(row, col, layout='constrained')
    group_fig.append(fig)
    group_ax.append(ax)

with mne.open_report(fname_rep_group.format('.h5')) as rep_group:
    rep_group.title = f"{task} Group Report"
    for ii, subject in enumerate(subjects_list):
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        # define filenames
        fname_ica = bids_path.update(suffix=derivative).fpath

        # ica input is from fixation cross to three hashes
        # no language involved
        bids_path.update(suffix=None)
        raw = read_raw_bids(bids_path).pick(['meg'])
        events, total_event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in total_event_id.items()
                    if key in evts_labels}

        epo_tmin, epo_tmax = -.1, .1
        reject = dict(mag=3e-12)
        epochs = mne.Epochs(raw, events, event_id, tmin=epo_tmin, tmax=epo_tmax,
                            baseline=None, reject=reject, verbose=False)

        if not op.exists(fname_ica) or redo:
            # compute the ICA
            # TODO: is there a good heuristic for why .9?
            ica = mne.preprocessing.ICA(.9, random_state=42, method='fastica')
            ica.fit(epochs)

        else:
            ica = mne.preprocessing.read_ica(fname_ica)

        # transform epochs to ICs
        epochs_ica = ica.get_sources(epochs.load_data())

        # compute the inter-trial coherence
        min_cycles = 1 / (epo_tmax - epo_tmin)
        # zero_mean == False was the setting when this analysis was originally
        # conducted. The default value changes in MNE 1.8
        # wip, maybe migrate to using the method
        # _, itc = epochs_ica.compute_tfr(method='morlet',
        #                                 freqs=np.arange(min_cycles, 30),
        #                                 picks='misc',
        #                                 output='power',
        #                                 average=True,
        #                                 n_cycles=.1,
        #                                 zero_mean=False,
        #                                 return_itc=True)
        itc = mne.time_frequency.tfr_array_morlet(epochs_ica.get_data(),
                                                  epochs_ica.info['sfreq'],
                                                  np.arange(min_cycles, 30),
                                                  n_cycles=.1,
                                                  zero_mean=False,
                                                  output='itc')
        # let's find the most coherent over this time course
        # TODO: find a source for time duration of saccade.
        itc_tmin, itc_tmax = -.1, .03
        start, stop = epochs_ica.time_as_index((itc_tmin, itc_tmax))
        # sum itc across time then sum across frequency
        itc_score = itc[start:stop].sum(axis=(1,2))
        # wip using itc object
        # itc_score = itc.get_data(tmin=start, tmax=stop).sum(axis=(1,2))
        # take the top three for comparison-sake
        ica_idx = (itc_score).argsort()[::-1][:3].tolist()

        # here, we're only remove the top offender
        ica.exclude = [ica_idx[0]]
        # need to have a labels dict
        ica.labels_ = {'eog': ica_idx}
        p = rep_group.add_ica(ica, title=f"{subject} {derivative}", inst=epochs,
                              picks=ica_idx, eog_evoked=epochs.average(),
                              eog_scores=itc_score, tags=(derivative,))

        for kk, ica_i in enumerate(ica_idx):
            # TODO: make a grid of the excluded ICAs across all subjects
            jj = np.unravel_index(ii, (row,col))
            p = ica.plot_components(ica_i, axes=group_ax[kk][jj], show=False)
            # Remove the IC Number from group plot
            group_ax[kk][jj].set_title(subject)
        if redo:
            ica.save(fname_ica, overwrite=redo)

    rep_group.save(fname_rep_group.format('.html'), open_browser=False, overwrite=True)

for group in group_ax:
    [ax.set_visible(False) for axes in group for ax in axes if ax.has_data() == False]

for jj, group in enumerate(group_fig):
    group.suptitle('Group IC Topographies')
    group.savefig(fname_group_img.format(jj))