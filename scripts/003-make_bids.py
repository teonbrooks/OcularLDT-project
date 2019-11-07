import os.path as op
import numpy as np
import shutil as sh

from mne import read_events
from mne.io import read_raw_fif
from mne_bids import (write_raw_bids, make_bids_basename,
                      make_bids_folders, make_dataset_description)

import config
import config_raw


filt_type = config.filt[:3]
reject = None
exp = config.exp
event_id = config.event_id
project_name = config.project_name

input_path = config.drives['home']
output_path = op.join(config.drives['home'], '..', '..', project_name)

# meg.
for subject in config_raw.subjects.keys():
    print(config.banner % subject)

    fname_raw = op.join(input_path, subject, 'mne',
                        subject + '_%s_calm_iir_hp0.51_lp40_filt-raw.fif' % exp)

    raw = read_raw_fif(fname_raw)
    events_data = read_events(op.join(input_path, subject, 'mne',
                              subject + '_%s-eve.txt' % exp))
    bids_basename = make_bids_basename(subject=subject, task=project_name)
    write_raw_bids(raw, bids_basename, output_path,
                   event_id=event_id, events_data=events_data,
                   overwrite=True)

# eyetrack. note, the BEP for eyetracking isn't merged
# but this is likely the naming convention for it
for subject, experiments in config_raw.subjects.items():
    exps = experiments[0], experiments[2]
    subname = 'sub-{}'.format(subject)
    for ii, exp in enumerate(exps, 1):
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