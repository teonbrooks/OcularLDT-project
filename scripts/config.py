import itertools
import numpy as np

# Parameters:
ds_factor = 8

subjects = {'A0148': ['OLDT'],
            'A0149': ['OLDT'],
            'A0129': ['OLDT1', 'SENT1', 'OLDT2'],
            }

data_dir = '/Users/teon/Google Drive/E-MEG/data/'
results_dir = '/Applications/packages/E-MEG/output/results/'

reject = dict(mag=3e-12)
img = 'png'
