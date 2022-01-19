import os.path as op
import shutil as sh

from mne import read_events
from mne.io import read_raw_fif
from mne_bids import (write_raw_bids, make_bids_basename,
                      make_bids_folders, make_dataset_description)

import config


experiment = config.experiment
filt = config.filt
event_id = config.event_id
project_name = config.project_name
subjects = config.exp_list.keys()

input_path = config.drives['home']
output_path = op.join(config.drives['home'], '..', '..', project_name)

# meg.
for subject in subjects:
    print(config.banner % subject)

    fname_raw = op.join(input_path, subject, 'mne',
                        '{subject}_{experiment}_calm_{filt}_filt-raw.fif')

    raw = read_raw_fif(fname_raw)
    events_data = read_events(op.join(input_path, subject, 'mne',
                              subject + '_%s-eve.txt' % experiment))
    bids_basename = make_bids_basename(subject=subject, task=project_name)
    write_raw_bids(raw, bids_basename, output_path,
                   event_id=event_id, events_data=events_data,
                   overwrite=True)

# eyetrack. note, the BEP for eyetracking isn't merged
# but this is likely the naming convention for it
for subject, experiments in config.exp_list.items():
    subname = 'sub-{}'.format(subject)
    for ii, exp in enumerate(experiments, 1):
        if exp != 'n/a':
            bids_basename = make_bids_basename(subject=subject,
                                            run='{:02d}'.format(ii),
                                            task=project_name)
            bids_eyetrack = op.join(output_path, subname,
                                    'eyetrack', bids_basename + '_eyetrack.edf')
            input_fname = op.join(input_path, subject,
                                'edf', '{}_{}.edf'.format(subject, exp))
            make_bids_folders(subject=subject, kind='eyetrack',
                            output_path=output_path)
            sh.copyfile(input_fname, bids_eyetrack)

# make a dataset description
make_dataset_description(path=output_path, data_license='CC-BY',
                         name=project_name,
                         authors=['Teon L Brooks', 'Laura Gwilliams',
                                  'Alexandre Gramfort', 'Alec Marantz'],
                         how_to_acknowledge='',
                         funding=['NSF DGE-1342536 (TB)',
                                  'Abu  Dhabi  Institute Grant G1001 (AM)'],
                         references_and_links='',
                         doi='')