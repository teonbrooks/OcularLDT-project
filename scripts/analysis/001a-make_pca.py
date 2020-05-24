"""
001a-make_pca.py

This script is used for the computation of the SSP for
each subject in the experiment. One approach to this analysis is to project
out this SSP that is related to the saccadic eye movement in this experiment.
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import mne
from mne.report import Report

from mne_bids import read_raw_bids
from mne_bids.read import _handle_events_reading
from mne_bids.utils import get_entity_vals


task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'teon-backup', 'Experiments', task)
derivative = 'pca'

redo = True
reject = dict(mag=3e-12)
baseline = (-.2, -.1)
tmin, tmax = -.5, 1
ylim = dict(mag=[-200, 200])

evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']
subjects_list = get_entity_vals(bids_root, entity_key='sub')

fname_rep_group = op.join('..', '..', 'output', 'preprocessing',
                          f'group_{task}_{derivative}-report.html')
rep_group = Report()

for subject in subjects_list:
    print("#" * 9 + f"\n# {subject} #\n" + "#" * 9)

    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    events_fname = op.join(path, f"sub-{subject}_task-{task}_events.tsv")
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_split-01_meg.fif")
    fname_proj = op.join(path, f"sub-{subject}_task-{task}_proj.fif")

    if not op.exists(fname_proj) or redo:
        # pca input is from fixation cross to three hashes
        # no language involved
        # TODO: replace path with basename
        try:
            raw = mne.io.read_raw_fif(fname_raw)
        except FileNotFoundError:
            raw = mne.io.read_raw_fif(fname_mp_raw)
        # TODO: replace with proper solution
        raw = _handle_events_reading(events_fname, raw)
        events, event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in event_id.items()
                    if key in evts_labels}
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1,
                            baseline=baseline, reject=reject, verbose=False)

        # compute the SSP
        evoked = epochs.average()
        ev_proj = evoked.copy().crop(-.1, .03)
        projs = mne.compute_proj_evoked(ev_proj, n_mag=3)

        # apply projector individually
        evokeds = [evoked.copy().add_proj(proj).apply_proj() for proj in projs]

        # 1. plot before and after summary
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(1, 2)
        axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]
        # plot evoked
        evoked.crop(-.1,.1).plot(titles={'mag': 'Before: Original Evoked'},
                                 show=False,
                                 axes=axes[0], ylim=ylim)
        # remove all
        evoked_proj = evoked.copy().add_proj(projs).apply_proj()
        evoked_proj.plot(titles={'mag': 'After: Evoked - First 3 PCs'},
                         show=False, axes=axes[1], ylim=ylim)
        rep_group.add_figs_to_section(fig, '%s: Evoked, Before and After PCA'
                                      % subject, 'Evoked, Before and After')

        # 2. plot PCA topos
        p = mne.viz.plot_projs_topomap(projs, evoked.info, show=False,
                                       extrapolate='head')
        rep_group.add_figs_to_section(p, '%s: PCA topos' % subject,
                                      'Topos')

        # 3. plot evoked - each proj
        for ii, ev in enumerate(evokeds):
            exp_var = ev.info['projs'][0]['explained_var'] * 100
            title = 'PC %d: %2.2f%% Explained Variance' % (ii, exp_var)
            tab = 'PC %d' % ii
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]
            e = ev.plot(titles={'mag': title},
                        show=False, axes=axes[0])
            p = mne.viz.plot_projs_topomap(ev.info['projs'], ev.info,
                                           show=False, axes=axes[1],
                                           extrapolate='head')
            rep_group.add_figs_to_section(fig, '%s: Evoked w/o PC %d'
                                          % (subject, ii), tab)

        # save projs
        mne.write_proj(fname_proj, projs)
        # cleanup
        del epochs
    rep_group.save(fname_rep_group, open_browser=False, overwrite=True)
