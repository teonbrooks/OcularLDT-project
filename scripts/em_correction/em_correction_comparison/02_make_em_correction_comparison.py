from pathlib import Path
import os.path as op
import tomllib as toml

import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

cfg = toml.load(open(Path('./config.toml'), 'rb'))
task = cfg['task']
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
tmin, tmax = -.2, .2

subjects = get_entity_vals(cfg['bids_root'], entity_key='subject')

for subject in subjects:
    bids_path.update(subject=subject)
    print(cfg['banner'] % subject)
    # define filenames
    subject_template = op.join(cfg['bids_root'], f'sub-{subject}',
                               'meg', f'sub-{subject}_task-{task}')
    fname_raw = f'{subject_template}_meg.fif'
    fname_ica = f'{subject_template}_ica.fif'
    fname_ga_uncorrected = f'{subject_template}_saccades_uncorrected_ave.fif'
    fname_ga_corrected = f'{subject_template}_saccades_corrected_ave.fif'

    # select only meg channels
    raw = read_raw_bids(bids_path).pick_types(meg=True).load_data()
    events, total_event_id = mne.events_from_annotations(raw)

    # compute grand averages for the uncorrected
    epochs_ga_uncorrected = mne.Epochs(raw, events, total_event_id,
                                       baseline=None, tmin=tmin, tmax=tmax)
    mne.write_evokeds(fname_ga_uncorrected,
                      epochs_ga_uncorrected.average())

    # now apply ICA
    ica = mne.preprocessing.read_ica(fname_ica)
    epochs_ga_corrected = ica.apply(epochs_ga_uncorrected.load_data())
    mne.write_evokeds(fname_ga_corrected,
                      epochs_ga_corrected.average())