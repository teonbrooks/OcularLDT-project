import os.path as op
from glob import glob
import re
import itertools
import numpy as np


# directories
drives = {'home': op.join('/Volumes', 'teon-backup', 'Experiments',
                          'OcularLDT'),
          'office': op.join('/Volumes', 'backup', 'Experiments',
                            'OcularLDT'),
          'project': '/Applications/codespace/OcularLDT-code',
         }

# Experiments
experiments = ['OLDT', 'SENT']
project_names = ['OcularLDT', 'OcularSimpleSentences']

index = 0
exp = experiments[index]
project_name = project_names[index]

# analysis parameters
redo = True
results_dir = op.join(drives['project'], 'output')
reject = dict(mag=3e-12)
baseline = (-.2, -.1)
img = 'png'
# can't really use `hp0.03` since their filter settings varied, name not true
filts = {'no': 'iir_no',
         'hp0.03': 'iir_hp0.03_lp40',  # DO NOT USE!
         'hp0.1': 'iir_hp0.1_lp40',  # try this after thesis is completed
         'hp0.51': 'iir_hp0.51_lp40', 'hp1': 'iir_hp1_lp40'}
filt = filts['hp0.51']
banner = ('#' * 9 + '\n# %s #\n' + '#' * 9)
# determine which drive you're working from
drive = 'home'
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
subjects = glob(op.join(drive, 'sub-A*'))
for ii, subject in enumerate(subjects):
    subjects[ii] = re.findall('/sub-(A[0-9]*)', subject)[0]
