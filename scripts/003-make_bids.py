import os.path as op
import numpy as np

from mne import read_events
from mne.io import read_raw_fif
from mne_bids import write_raw_bids, make_bids_basename, make_dataset_description

import config
import config_raw


filt_type = config.filt[:3]
reject = None
exp = config.exp
event_id = config.event_id
project_name = config.project_name

input_path = config.drives['home']
output_path = op.join(config.drives['home'], '..', '..', project_name)

for subject in config_raw.subjects.keys():
    print(config.banner % subject)

    fname_raw = op.join(input_path, subject, 'mne',
                        subject + '_%s_calm_iir_hp0.51_lp40_filt-raw.fif' % exp)

    raw = read_raw_fif(fname_raw)
    events_data = read_events(op.join(input_path, subject, 'mne',
                              subject + '_%s-eve.txt' % exp))
    bids_basename = make_bids_basename(subject=subject, session='01',
                                       run='01', task=project_name)
    write_raw_bids(raw, bids_basename, output_path,
                   event_id=event_id, events_data=events_data,
                   overwrite=True)

make_dataset_description(path=output_path, data_license='CC-BY',
                         authors=['Teon L Brooks', 'Laura Gwilliams',
                                  'Alexandre Gramfort', 'Alec Marantz'],
                         how_to_acknowledge='',
                         funding=['NSF DGE-1342536 (TB)',
                                  'Abu  Dhabi  Institute Grant G1001 (AM)'],
                         references_and_links='',
                         doi='')