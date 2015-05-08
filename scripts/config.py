import itertools
import numpy as np

# Parameters:
ds_factor = 8

subjects = {'A0148': ['OLDT'],
            'A0149': ['OLDT'],
            'A0129': ['OLDT1', 'SENT1', 'OLDT2'],
            }

data_dir = '/Users/teon/Google Drive/E-MEG/data/'
results_dir = '/Users/teon/Dropbox/academic/Experiments/E-MEG/output/results/'

reject = dict(mag=3e-12)
img = 'png'

bads = {'A0148': ['MEG 035', 'MEG 130', 'MEG 095', 'MEG 182'],
        'A0149': ['MEG 067', 'MEG 078', 'MEG 095', 'MEG 102', 
                  'MEG 130', 'MEG 138', 'MEG 160', 'MEG 176',
                  'MEG 183', 'MEG 195'],
        'A0129': ['MEG 130'],
        }
