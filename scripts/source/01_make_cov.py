"""
001_make_cov.py

This script is used for the computation of the covariance matrix for
each subject in the experiment. The covariance matrix is used for the
inverse estimate when the sensor data is projected to source space.
"""
from pathlib import Path
import tomllib as toml

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))

redo = False
task = cfg['task']
bids_root = root / 'data' / task
derivative = 'cov'


evts_labels = ['word/prime/unprimed', 'word/prime/primed', 'nonword/prime']
subjects_list = get_entity_vals(bids_root, entity_key='subject')
bids_path = BIDSPath(root=bids_root, session=None, task=task,
                     datatype=cfg['datatype'])

basename_rep_group = str(root / 'output' / 'reports', f'group_{task}-report')

with mne.open_report(basename_rep_group + '.h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        raw = read_raw_bids(bids_path)
        events, event_id = mne.events_from_annotations(raw)
        event_id = {key: value for key, value in event_id.items()
                    if key in evts_labels}
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=.2,
                            baseline=None, reject={'mag': 3e-12},
                            verbose=False, preload=True)

        # define filenames
        fname_cov = bids_path.update(suffix=derivative, extension='.fif',
                                     check=False).fpath
        fname_ica = bids_path.update(suffix='ica', extension='.fif',
                                     check=False).fpath

        # apply ica to epochs
        ica = mne.preprocessing.read_ica(fname_ica)
        epochs = ica.apply(epochs)
        evoked = epochs.average()

        if not fname_cov.exists() or redo:
            # plot covariance and whitened evoked
            epochs.load_data().crop(-.2, -.1)

            cov = mne.compute_covariance(epochs, method='auto', verbose=False)
            # comments = ('The covariance matrix is computed on the -200:-100 ms '
            #             'baseline. -100:0 ms is confounded with the eye-mvt.')
            # save covariance
            mne.write_cov(fname_cov, cov)
        else:
            cov = mne.read_cov(fname_cov)

        rep_group.add_covariance(cov, info=epochs.info, tags=(derivative,),
                                 title=f'{subject} {derivative}')

        rep_group.add_evokeds(evoked,  titles=f'{subject} evoked',
                              tags=('evoked',), noise_cov=cov)

    rep_group.save(basename_rep_group + '.html', overwrite=True,
                   open_browser=False)
