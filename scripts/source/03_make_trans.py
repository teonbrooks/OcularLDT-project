"""
002_make_trans.py

This script is used for the computation of the trans matrix for
each subject in the experiment. The trans matrix is used to transform
data from MEG sensor space to MRI space. This is necessary for source
estimation. In this study, we used the template brain, fsaverage, from
FreeSurfer and scale the head model using digitization data for
constraints. This is semi-automatic, we will use a uniform scaling mode
and will fit using ICP.

Note, the scaled MRI creation does not follow the BIDS standard as of now.
To prevent breaking the dataset, these MRIs are placed in a separate
directory, `OcularLDT_MRI`.
"""
from pathlib import Path
import tomllib as toml

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids


mne.viz.set_3d_backend('pyvistaqt')

parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))

redo = True
task = cfg['task']
derivative = 'trans'

bids_root = root / 'data' / task
subjects_list = get_entity_vals(bids_root, entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])

basename_rep_group = str(root / 'output' / 'reports', f'group_{task}-report')

# first, copy over fsaverage from FreeSurfer
with mne.open_report(basename_rep_group + '.h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        raw = read_raw_bids(bids_path)
        if not bids_path.fpath.exists() or redo:
            mne.gui.coregistration(inst=raw.filenames[0],
                                subjects_dir=cfg['mri_root'])

        # define filename
        # trans files are not supported in the spec so set check to false
        fname_trans = bids_path.update(suffix='trans', extension='.fif').fpath

        rep_group.add_trans(trans=bids_path.fpath, info=raw.info,
                            title=f"{subject} trans", tags=(derivative,),
                            subject=f"sub-{subject}",
                            subjects_dir=bids_root + '_MRI')

rep_group.save(basename_rep_group + '.html', overwrite=True,
               open_browser=False)
