from pathlib import Path
import os.path as op
import tomllib as toml

import shutil as sh

from mne import read_events
from mne.io import read_raw_fif
from mne_bids import (write_raw_bids, make_bids_basename,
                      make_bids_folders, make_dataset_description)

cfg = toml.load(open(Path('./cfg.toml'), 'rb'))


experiment = cfg.experiment
filt = cfg.filt
event_id = cfg.event_id
task = cfg.task
subjects = cfg.exp_list.keys()

input_path = cfg.drives['home']
output_path = cfg.bids_root

# meg.
for subject in subjects:
    print(cfg.banner % subject)

    fname_raw = op.join(input_path, subject, 'mne',
                        f'{subject}_{experiment}_calm_{filt}_filt-raw.fif')

    raw = read_raw_fif(fname_raw)
    events_data = read_events(op.join(input_path, subject, 'mne',
                              subject + '_%s-eve.txt' % experiment))
    bids_basename = make_bids_basename(subject=subject, task=task)
    write_raw_bids(raw, bids_basename, output_path,
                   event_id=event_id, events_data=events_data,
                   overwrite=True)

# eyetrack. note, the BEP for eyetracking isn't merged
# but this is likely the naming convention for it
for subject, experiments in cfg.exp_list.items():
    subname = f'sub-{subject}'
    for ii, exp in enumerate(experiments, 1):
        if exp != 'n/a':
            bids_basename = make_bids_basename(subject=subject,
                                            run='{:02d}'.format(ii),
                                            task=task)
            bids_eyetrack = op.join(output_path, subname,
                                    'eyetrack', bids_basename + '_eyetrack.edf')
            input_fname = op.join(input_path, subject,
                                'edf', f'{subject}_{exp}.edf')
            make_bids_folders(subject=subject, kind='eyetrack',
                            output_path=output_path)
            sh.copyfile(input_fname, bids_eyetrack)

# make a dataset description
make_dataset_description(path=output_path, data_license='CC0',
                         name=task,
                         authors=['Teon L Brooks', 'Laura Gwilliams',
                                  'Alexandre Gramfort', 'Alec Marantz'],
                         how_to_acknowledge='',
                         funding=['NSF DGE-1342536 (TB)',
                                  'Abu  Dhabi  Institute Grant G1001 (AM)'],
                         references_and_links='',
                         doi='')