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
import os.path as op
import json

import mne
from mne.report import Report

from mne_bids import read_raw_bids, BIDSPath
from mne_bids import get_entity_vals


mne.viz.set_3d_backend('pyvistaqt')
cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
derivative = 'trans'

redo = True

bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')

fname_rep_group = op.join(cfg['project_path'], 'output', 'reports',
                          f'group_{task}-report.%s')

# first, copy over fsaverage from FreeSurfer
with mne.open_report(fname_rep_group % 'h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        # define filename
        fname_trans = op.join(cfg['bids_root'], f"sub-{subject}", 'meg',
                              f"sub-{subject}_task-{task}_{derivative}.fif")

        raw = read_raw_bids(bids_path)
        if not op.exists(fname_trans) or redo:
            mne.gui.coregistration(inst=raw.filenames[0],
                                subjects_dir=cfg['mri_root'])

        rep_group.add_trans(trans=fname_trans, info=raw.info,
                            title=f"{subject} trans",
                            subject=f"sub-{subject}",
                            subjects_dir=cfg['mri_root'], tags=('trans',))

rep_group.save(fname_rep_group % 'html', overwrite=True, open_browser=False)
