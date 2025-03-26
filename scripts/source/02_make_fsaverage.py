"""
002_make_fsaverage.py

This script is used to copy the fsaverage brain from freesurfer to this
project's data directory for the computation of the source space.
"""
from pathlib import Path
import tomllib as toml

import mne


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))

task = cfg['task']
mri_root = root / 'data' / task + '_MRI'


# first, copy over fsaverage from FreeSurfer

mne.coreg.create_default_subject(fs_home=cfg['freesurfer_root'], update=True,
                                subjects_dir=mri_root)
ss = mne.setup_source_space(subject='fsaverage', spacing='ico4',
                            surface='white', subjects_dir=mri_root)
fname_src = mri_root / 'fsaverage' / 'bem' / 'fsaverage-ico-4-src.fif'
mne.write_source_spaces(fname_src, ss)
