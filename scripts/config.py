import os.path as op
from glob import glob
import itertools
import numpy as np
from collections import defaultdict


mne_bin = '/Applications/packages/mne-c/bin'
# drives
drives = {'local': op.join(op.expanduser('~'), 'Experiments', 'E-MEG', 'data'),
          'server': op.join('/Volumes', 'server', 'MORPHLAB', 'Teon',
                            'E-MEG', 'data'),
          'home': op.join('/Volumes', 'teon-backup', 'Experiments',
                          'E-MEG', 'data'),
          'office': op.join('/Volumes', 'GLYPH-1 TB', 'Experiments',
                            'E-MEG', 'data'),
          'google_drive': op.join(op.expanduser('~'), 'Google Drive',
                                  'E-MEG', 'data'),
          'dropbox': op.join(op.expanduser('~'), 'Dropbox', 'academic',
                             'Experiments', 'E-MEG', 'output'),
         }
# Parameters:
redo = True
results_dir = '/Users/teon/Dropbox/academic/Experiments/E-MEG/output/results/'
reject = dict(mag=3e-12)
img = 'png'
filt = 'iir'
banner = ('#' * 9 + '\n# %s #\n' + '#' * 9)
# running from NY
drive = 'home'


# Bad Channels
bads = defaultdict(lambda: ['MEG 130'])
bads['A0129'] += ['MEG 057', 'MEG 082', 'MEG 087', 'MEG 115',
                  'MEG 157', 'MEG 179', 'MEG 181', 'MEG 198',
                  'MEG 092', 'MEG 041', 'MEG 102']  # weird drift
bads['A0148'] += ['MEG 004', 'MEG 011', 'MEG 019', 'MEG 021',
                  'MEG 030', 'MEG 032', 'MEG 046', 'MEG 057',
                  'MEG 072', 'MEG 084', 'MEG 088', 'MEG 109',
                  'MEG 114', 'MEG 119', 'MEG 122', 'MEG 125',
                  'MEG 133', 'MEG 147', 'MEG 171', 'MEG 177',
                  'MEG 179', 'MEG 198', 'MEG 207']
bads['A0023'] += ['MEG 059']
bads['A0023'] += ['MEG 049', 'MEG 054']
bads['A0136'] += ['MEG 057', 'MEG 128', 'MEG 179']
bads['A0130'] += ['MEG 020', 'MEG 024', 'MEG 025', 'MEG 031',
                  'MEG 034', 'MEG 046', 'MEG 057', 'MEG 072',
                  'MEG 078', 'MEG 080', 'MEG 081', 'MEG 082',
                  'MEG 087', 'MEG 094', 'MEG 111', 'MEG 115',
                  'MEG 128', 'MEG 130', 'MEG 134', 'MEG 146',
                  'MEG 149', 'MEG 163', 'MEG 177', 'MEG 179',
                  'MEG 181', 'MEG 186', 'MEG 195', 'MEG 195']
bads['A0161'] += ['MEG 005', 'MEG 009', 'MEG 010', 'MEG 012',
                  'MEG 020', 'MEG 024', 'MEG 029', 'MEG 032',
                  'MEG 035', 'MEG 038', 'MEG 051', 'MEG 064',
                  'MEG 068', 'MEG 075', 'MEG 078', 'MEG 080',
                  'MEG 081', 'MEG 082', 'MEG 084', 'MEG 087',
                  'MEG 089', 'MEG 095', 'MEG 097', 'MEG 104',
                  'MEG 108', 'MEG 111', 'MEG 116', 'MEG 136',
                  'MEG 145', 'MEG 146', 'MEG 149', 'MEG 164',
                  'MEG 165', 'MEG 166', 'MEG 177', 'MEG 187',
                  'MEG 194', 'MEG 198', 'MEG 199', 'MEG 202',
                  'MEG 203', 'MEG 207']



epochs = {'A0148': 144, 'A0129': 95, 'A0161': 3}
# A0161 first run drop

# arrange the OLDT in the presentation order
subjects = {'A0023': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0078': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0085': ['OLDT2', 'SENT2', 'OLDT1'],
            # 'A0100': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0106': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0110': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0123': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0125': ['OLDT1', 'SENT2', 'OLDT2'],
            'A0127': ['OLDT2', 'SENT1', 'OLDT1'],
            'A0129': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0130': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0134': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0136': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0148': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0150': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0155': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0159': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0161': ['OLDT1', 'SENT1', 'OLDT2'],
            # 'A0163': ['OLDT2', 'SENT2', 'OLDT1'],
            # 'A0164': ['OLDT2', 'SENT2', 'OLDT1'],
            }

for subject, _ in subjects.items():
    if op.exists(op.join(drives[drive], subject)):
        continue
    else:
        del subjects[subject]

drive = drives[drive]

def kit2fiff(subject, exp, path, dig=True, preload=False):
    from mne.io import read_raw_kit
    kit = op.join(path, subject, 'kit', 
                  subject + '*' + exp + '*' + 'calm.con')
    mrk_pre = op.join(path, subject, 'kit', 
                      subject + '*mrk*' + 'pre_' + exp + '*.mrk')
    mrk_post = op.join(path, subject, 'kit', 
                       subject + '*mrk*' + 'post_' + exp + '*.mrk')
    elp = op.join(path, subject, 'kit', subject + '*p.txt')
    hsp = op.join(path, subject, 'kit', subject + '*h.txt')

    mrk_pre = glob(mrk_pre)[0]
    mrk_post = glob(mrk_post)[0]
    elp = glob(elp)[0]
    hsp = glob(hsp)[0]
    kit = glob(kit)[0]

    if dig:
        raw = read_raw_kit(input_fname=kit, mrk=[mrk_pre, mrk_post],
                           elp=elp, hsp=hsp, stim='>', slope='+',
                           preload=preload, verbose=False)
    else:
        raw = read_raw_kit(input_fname=kit, stim='>', slope='+',
                           preload=preload, verbose=False)

    return raw
