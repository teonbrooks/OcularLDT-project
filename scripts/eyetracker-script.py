import os.path as op
import pyeparse as pp
from pyeparse import interest_areas as ias


path = '/Users/teon/Experiments/E-MEG/data'
raw = pp.read_raw(op.join(path, 'A0129', 'edf', 'A0129_OLDT1.edf'))
ia_regions = ias.read_ia(op.join(path, 'group', 'OLDT_ias.txt'))

ia = ias.InterestAreas(raw, ia_regions, None, 'fix')
