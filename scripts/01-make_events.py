import mne
import numpy as np
import os.path as op
import config
from make_events import make_events

from config import subjects, data_dir

for subject in subjects:
    print subject
    for expt in subjects[subject]:
        make_events(data_dir, subject, expt)
