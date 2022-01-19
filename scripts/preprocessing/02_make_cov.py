"""
002-make_cov.py

This script is used for the computation of the covariance matrix for
each subject in the experiment. The covariance matrix is used for the
inverse estimate when the sensor data is projected to source space.
"""
import os.path as op

import mne
from mne.report import Report

from mne_bids import read_raw_bids
from mne_bids.read import _handle_events_reading
from mne_bids.utils import get_entity_vals


layout = mne.channels.read_layout('KIT-AD.lout')
task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'teon-backup', 'Experiments', task)
derivative = 'cov'

redo = True

evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']
subjects_list = get_entity_vals(bids_root, entity_key='sub')

fname_rep_group = op.join('/', 'Users', 'tbrooks', 'codespace', 'OcularLDT-code',
                          'output', 'group',
                          f'group_{task}_{derivative}-report.html')
rep_group = Report()

for subject in subjects_list:
    print("#" * 9 + f"\n# {subject} #\n" + "#" * 9)
    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    events_fname = op.join(path, f"sub-{subject}_task-{task}_events.tsv")
    # TODO: replace path with basename
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_split-01_meg.fif")
    fname_cov = op.join(path, f"sub-{subject}_task-{task}_{derivative}.fif")

    if not op.exists(fname_cov) or redo:

        try:
            raw = mne.io.read_raw_fif(fname_raw)
        except FileNotFoundError:
            raw = mne.io.read_raw_fif(fname_mp_raw)
        # TODO: replace with proper solution
        raw = _handle_events_reading(events_fname, raw)
        events, event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in event_id.items()
                    if key in evts_labels}
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=.2,
                            baseline=(-.2, -.1), reject={'mag': 3e-12},
                            verbose=False)

        # # back to coding
        # proj = mne.read_proj(fname_proj)
        # epochs.add_proj(proj)
        # epochs.apply_proj()

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Evoked Response'}, show=False)
        rep_group.add_figs_to_section(p,
                                      (f"{subject}: Evoked Response " +
                                       "to Prime Word"),
                                      'Evoked')

        # plot covariance and whitened evoked
        epochs.load_data().crop(-.2, -.1)
        cov = mne.compute_covariance(epochs, method='auto', verbose=False)
        p = cov.plot(epochs.info, show_svd=0, show=False)[0]
        # comments = ('The covariance matrix is computed on the -200:-100 ms '
        #             'baseline. -100:0 ms is confounded with the eye-mvt.')
        rep_group.add_figs_to_section(p,
                                      f"{subject}: Covariance Matrix",
                                      'Covariance Matrix')
        p = evoked.plot_white(cov, show=False)
        rep_group.add_figs_to_section(p,
                                f"{subject}: Whitened Evoked to Prime Word",
                                'Whitened Evoked')

        # save covariance
        mne.write_cov(fname_cov, cov)

rep_group.save(fname_rep_group, overwrite=True, open_browser=False)


