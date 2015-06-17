import mne
import numpy as np
import os.path as op
import config
import make_events


redo = True
path = config.drive
for subject in config.subjects:
    print subject
    exps = config.subjects[subject]
    evt_file = op.join(path, subject, 'mne', subject + '_OLDT-eve.txt')
    if not op.exists(evt_file) or redo:
        raw = config.kit2fiff(subject=subject, exp=exps[0],
                              path=path, preload=False)
        raw2 = config.kit2fiff(subject=subject, exp=exps[2],
                              path=path, preload=False)
        mne.concatenate_raws([raw, raw2])
        make_events.make_events(raw, subject, 'OLDT')
