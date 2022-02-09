"""
001_make_cov.py

This script is used for the computation of the covariance matrix for
each subject in the experiment. The covariance matrix is used for the
inverse estimate when the sensor data is projected to source space.
"""
import os.path as op
import json

import mne
from mne.report import Report

from mne_bids import get_entity_vals, read_raw_bids, BIDSPath


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
derivative = 'cov'

redo = False

evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']
subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])

fname_rep_group = op.join(cfg['project_path'], 'output', 'reports',
                          f'group_{task}-report.%s')

with mne.open_report(fname_rep_group % 'h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        # define filenames
        fname_cov = op.join(cfg['bids_root'], f"sub-{subject}",
                            cfg['datatype'],
                            f"sub-{subject}_task-{task}_{derivative}.fif")
        fname_ica = op.join(cfg['bids_root'], f"sub-{subject}",
                            cfg['datatype'],
                            f"sub-{subject}_task-{task}_ica.fif")


        raw = read_raw_bids(bids_path)
        events, event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in event_id.items()
                    if key in evts_labels}
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=.2,
                            baseline=None, reject={'mag': 3e-12},
                            verbose=False, preload=True)

        # apply ica to epochs
        ica = mne.preprocessing.read_ica(fname_ica)
        epochs = ica.apply(epochs)
        evoked = epochs.average()

        if not op.exists(fname_cov) or redo:
            # plot covariance and whitened evoked
            epochs.load_data().crop(-.2, -.1)

            cov = mne.compute_covariance(epochs, method='auto', verbose=False)
            # comments = ('The covariance matrix is computed on the -200:-100 ms '
            #             'baseline. -100:0 ms is confounded with the eye-mvt.')
            # save covariance
            mne.write_cov(fname_cov, cov)
        else:
            cov = mne.read_cov(fname_cov)

        rep_group.add_covariance(cov, info=epochs.info,
                                 title=f'{subject} {derivative}')

        rep_group.add_evokeds(evoked,  titles=f'{subject} evoked',
                              noise_cov=cov, tags=('evoked',))

    rep_group.save(fname_rep_group % 'html', overwrite=redo,
                   open_browser=False)
