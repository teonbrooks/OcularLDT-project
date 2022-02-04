"""
002_make_fsaverage.py

This script is used to copy the fsaverage brain from freesurfer to this
project's data directory for the computation of the source space.
"""
import os.path as op
import json

import mne


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))

# first, copy over fsaverage from FreeSurfer

mne.coreg.create_default_subject(fs_home=cfg['freesurfer_root'], update=True,
                                subjects_dir=cfg['mri_root'])
ss = mne.setup_source_space(subject='fsaverage', spacing='ico4',
                            surface='white', subjects_dir=cfg['mri_root'])
fname_src = op.join(cfg['mri_root'], 'fsaverage', 'bem',
                    'fsaverage-ico-4-src.fif')
mne.write_source_spaces(fname_src, ss)