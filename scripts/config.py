import os.path as op
from glob import glob
import re
import itertools
import numpy as np


# directories
drives = {'local': op.join(op.expanduser('~'), 'Experiments', 'E-MEG', 'data'),
          'server': op.join('/Volumes', 'server', 'MORPHLAB', 'Teon',
                            'E-MEG', 'data'),
          'home': op.join('/Volumes', 'teon-backup', 'Experiments',
                          'E-MEG', 'data'),
          'office': op.join('/Volumes', 'backup', 'Experiments',
                            'E-MEG', 'data'),
          'google_drive': op.join(op.expanduser('~'), 'Google Drive',
                                  'E-MEG', 'data'),
          'dropbox': op.join(op.expanduser('~'), 'Dropbox', 'academic',
                             'Experiments', 'E-MEG', 'output'),
          'mne_bin': '/Applications/packages/mne-c/bin',
          'project': '/Applications/packages/E-MEG',
         }

# Experiments
experiments = ['OLDT', 'SENT']
exp = experiments[0]

# analysis parameters
redo = False
results_dir = op.join(drives['dropbox'], 'results')
reject = dict(mag=3e-12)
baseline = (-.2, -.1)
img = 'png'
# can't really use `hp0.03` since there filter settings varied, name not true
filts = {'no': 'iir_no',
         'hp0.03': 'iir_hp0.03_lp40',  # DO NOT USE!
         'hp0.1': 'iir_hp0.1_lp40',  # try this after thesis is completed
         'hp0.51': 'iir_hp0.51_lp40', 'hp1': 'iir_hp1_lp40'}
filt = filts['hp0.51']
banner = ('#' * 9 + '\n# %s #\n' + '#' * 9)
# determine which drive you're working from
drive = 'office'
event_id = {'word/prime/unprimed': 1,
            'word/target/unprimed': 2,
            'word/prime/primed': 5,
            'word/target/primed': 6}
if exp == 'OLDT':
    event_id.update({'nonword/prime': 9, 'nonword/target': 10,
                     # for the co-registration, there are no
                     # fixations in the evts
                     # 'fixation': 128
                    })
drive = drives[drive]
subjects = glob(op.join(drive, 'A*'))
for ii, subject in enumerate(subjects):
    subjects[ii] = re.findall('/(A[0-9]*)', subject)[0]
