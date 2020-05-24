import os.path as op

import mne
from mne.report import Report

from mne_bids import read_raw_bids
from mne_bids.read import _handle_events_reading
from mne_bids.utils import get_entity_vals

layout = mne.channels.read_layout('KIT-AD.lout')
task = 'OcularLDT'
bids_root = op.join('/', 'Volumes', 'teon-backup', 'Experiments', task)
fs_home = op.join('/', 'Applications', 'freesurfer', '7.1.0')
mri_subjects_dir = op.join('/', 'Volumes', 'teon-backup', 'Experiments',
                           task + '_MRI')
derivative = 'fwd'

redo = False

subjects_list = get_entity_vals(bids_root, entity_key='sub')

for subject in subjects_list:
    print("#" * 9 + f"\n# {subject} #\n" + "#" * 9)

    # define filenames
    path = op.join(bids_root, f"sub-{subject}", 'meg')
    fname_raw = op.join(path, f"sub-{subject}_task-{task}_meg.fif")
    fname_mp_raw = op.join(path, f"sub-{subject}_task-{task}_split-01_meg.fif")
    fname_fwd = op.join(path, f"sub-{subject}_task-{task}_{derivative}.fif")
    fname_trans = op.join(path, f"sub-{subject}_task-{task}_trans.fif")
    bem_sol = op.join(mri_subjects_dir, f"sub-{subject}", 'bem',
                      f'sub-{subject}-inner_skull-bem-sol.fif')
    fname_src = op.join(mri_subjects_dir, f"sub-{subject}", 'bem',
                        f'sub-{subject}-ico-4-src.fif')
    if not op.exists(fname_fwd) or redo:
        try:
            raw = mne.io.read_raw_fif(fname_raw)
        except FileNotFoundError:
            raw = mne.io.read_raw_fif(fname_mp_raw)
            fname_raw = fname_mp_raw

        info = mne.io.read_info(fname_raw)

        fwd = mne.make_forward_solution(info=info, trans=fname_trans,
                                        src=fname_src, bem=bem_sol,
                                        meg=True, eeg=False,
                                        mindist=0.0, ignore_ref=True)
        mne.write_forward_solution(fname=fname_fwd, fwd=fwd)
