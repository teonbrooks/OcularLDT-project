import os.path as op
import json

import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['project_name']
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
tmin, tmax = -.2, .2

subjects = get_entity_vals(cfg['bids_root'], entity_key='subject')

evokeds_uncorrected = list()
evokeds_corrected = list()

for subject in subjects:
    bids_path.update(subject=subject)
    print(cfg['banner'] % subject)
    # define filenames
    subject_template = op.join(cfg['bids_root'], f'sub-{subject}',
                               'meg', f'sub-{subject}_task-{task}')
    fname_ga_uncorrected = f'{subject_template}_saccades_uncorrected_ave.fif'
    fname_ga_corrected = f'{subject_template}_saccades_corrected_ave.fif'

    evokeds_uncorrected.extend(mne.read_evokeds(fname_ga_uncorrected))
    evokeds_corrected.extend(mne.read_evokeds(fname_ga_corrected))
