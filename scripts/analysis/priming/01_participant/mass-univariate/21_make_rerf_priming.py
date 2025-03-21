import os.path as op
import json

import mne
from mne.stats import linear_regression_raw
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['project_name']
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
decim = 2
tmin, tmax = -.2, 1

c_names = ['word/target/unprimed', 'word/target/primed']
event_id = {c_name: cfg['event_id'][c_name] for c_name in c_names}

subjects = get_entity_vals(cfg['bids_root'], entity_key='subject')

for subject in subjects[:1]:
    bids_path.update(subject=subject)
    print(cfg['banner'] % subject)
    # define filenames
    subject_template = op.join(cfg['bids_root'], f'sub-{subject}',
                               'meg', f'sub-{subject}_task-{task}')
    fname_raw = f'{subject_template}_meg.fif'
    fname_ica = f'{subject_template}_ica.fif'
    fname_erf_uncorrected = f'{subject_template}_priming_uncorrected_ave.fif'
    fname_erf_corrected = f'{subject_template}_priming_corrected_ave.fif'

    # select only meg channels
    raw = read_raw_bids(bids_path).pick_types(meg=True).load_data()
    events, total_event_id = mne.events_from_annotations(raw)

    # compute an ERF for the priming conditions
    epochs_uncorrected = mne.Epochs(raw, events, event_id, baseline = None,
                                    tmin=tmin, tmax=tmax)
    epochs_uncorrected.equalize_event_counts(c_names)
    evokeds_uncorrected = epochs_uncorrected.average(by_event_type=True)
    mne.write_evokeds(fname_erf_uncorrected, evokeds_uncorrected)
    del evokeds_uncorrected

    # now apply ICA
    ica = mne.preprocessing.read_ica(fname_ica)
    epochs_corrected = ica.apply(epochs_uncorrected.load_data())
    evokeds_corrected = epochs_corrected.average(by_event_type=True)
    mne.write_evokeds(fname_erf_uncorrected, evokeds_corrected)
    del evokeds_corrected
