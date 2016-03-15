import os.path as op
from glob import glob
import itertools
import numpy as np


# arrange the OLDT in the presentation order
# this is for the 001-make_raw.py
subjects = {'A0023': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0078': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0085': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0100': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0106': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0110': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0123': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0125': ['OLDT1', 'SENT2', 'OLDT2'],
            'A0127': ['OLDT2', 'SENT1', 'OLDT1'],
            'A0129': ['OLDT1', 'SENT1', 'OLDT2'],
            # 'A0130': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0134': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0136': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0148': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0150': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0155': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0159': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0161': ['n/a', 'SENT1', 'OLDT2'],
            'A0161': ['OLDT1', 'SENT1', 'OLDT2'],
            'A0163': ['OLDT2', 'SENT2', 'OLDT1'],
            'A0164': ['OLDT2', 'SENT2', 'OLDT1'],
            }

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
