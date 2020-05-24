"""
003-make_trans.py

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

import mne
from mne.report import Report
mne.viz.set_3d_backend('mayavi')

from mne_bids import read_raw_bids
from mne_bids.read import _handle_events_reading
from mne_bids.utils import get_entity_vals


layout = mne.channels.read_layout('KIT-AD.lout')
task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'teon-backup', 'Experiments', task)
fs_home = op.join('/', 'Applications', 'freesurfer', '7.1.0')
mri_subjects_dir = op.join('/', 'Volumes', 'teon-backup', 'Experiments',
                           task + '_MRI')
derivative = 'trans'

redo = False

subjects_list = get_entity_vals(bids_root, entity_key='sub')

fname_rep_group = op.join('/', 'Users', 'tbrooks', 'codespace',
                          f'{task}-code', 'output', 'preprocessing',
                          f'group_{task}_{derivative}-report.html')
rep_group = Report()

# first, copy over fsaverage from FreeSurfer
mne.coreg.create_default_subject(fs_home=fs_home, update=True,
                                 subjects_dir=mri_subjects_dir)


for subject in subjects_list:
    print("#" * 9 + f"\n# {subject} #\n" + "#" * 9)
    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_split-01_meg.fif")
    fname_trans = op.join(path, f"sub-{subject}_task-{task}_{derivative}.fif")


    try:
        raw = mne.io.read_raw_fif(fname_raw)
    except FileNotFoundError:
        raw = mne.io.read_raw_fif(fname_mp_raw)
        fname_raw = fname_mp_raw

    if not op.exists(fname_trans) or redo:
        mne.viz.set_3d_backend('mayavi')
        mne.gui.coregistration(inst=fname_raw, subjects_dir=mri_subjects_dir)

    mne.viz.set_3d_backend('pyvista')
    p = mne.viz.plot_alignment(raw.info, trans=fname_trans,
                                subject=f'sub-{subject}',
                                subjects_dir=mri_subjects_dir,
                                surfaces='head',
                                dig=True, eeg=[], meg='sensors',
                                coord_frame='meg')
    rep_group.add_figs_to_section(p, f"{subject}", 'Coreg')

    rep_group.save(fname_rep_group, overwrite=True, open_browser=False)


