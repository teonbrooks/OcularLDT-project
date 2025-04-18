import os.path as op
from pathlib import Path
import tomllib as toml

from scripts._helper import recode_events
from _reading import AOIReport

cfg = toml.load(open(Path('./config.toml'), 'rb'))
# Define OLDT interest areas
fname_ia = op.join(cfg['project_path'], 'input', 'stimuli',
                   'OcularLDT_ias.txt')
ia_words = ['fixation', 'prime', 'target', 'post']

for subject, experiments in cfg['exp_list'].items():
    print(cfg['banner'] % subject)
    for ii, exp in enumerate(experiments, 1):
        if exp == 'n/a':
            continue
        # Define filenames
        basename = op.join(cfg['bids_root'], f"sub-{subject}", 'eyetrack',
                       f"sub-{subject}_task-{cfg['task']}_run-{ii:02d}")    
        fname_edf = basename + "_eyetrack.edf"
        fname_ds = basename + "_log.tsv"

        # extracting fixation times from the edf file.
        data = AOIReport(fname_edf, fname_ia, ia_words).data

        # we need to recode before we save to disk
        data['trigger'] = data['msg'].str.extract(r'TTL ([0-9]+)')
        data['trigger'] = data['trigger'].fillna(0).astype(int)
        data['trigger'] = recode_events(data['trigger'])

        data.to_csv(fname_ds, sep='\t', index=False)
