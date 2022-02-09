import os.path as op
import json

import mne
from mne.report import Report
from mne_bids import get_entity_vals, BIDSPath


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
derivative = 'fwd'

redo = True

subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])

fname_rep_group = op.join(cfg['project_path'], 'output', 'reports',
                          f'group_{task}-report.%s')

with mne.open_report(fname_rep_group % 'h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject, suffix='meg')

        # define filenames
        path = op.join(cfg['bids_root'], f"sub-{subject}", 'meg')
        fname_fwd = op.join(path, f"sub-{subject}_task-{task}_{derivative}.fif")
        fname_trans = op.join(path, f"sub-{subject}_task-{task}_trans.fif")
        bem_sol = op.join(cfg['mri_root'], f"sub-{subject}", 'bem',
                        f'sub-{subject}-inner_skull-bem-sol.fif')
        fname_src = op.join(cfg['mri_root'], f"sub-{subject}", 'bem',
                            f'sub-{subject}-ico-4-src.fif')
        if not op.exists(fname_fwd) or redo:
            info = mne.io.read_info(bids_path.fpath)
            fwd = mne.make_forward_solution(info=info, trans=fname_trans,
                                            src=fname_src, bem=bem_sol,
                                            meg=True, eeg=False,
                                            mindist=0.0, ignore_ref=True)
            mne.write_forward_solution(fname=fname_fwd, fwd=fwd,
                                       overwrite=redo)

        rep_group.add_forward(fwd, title=f"{subject} fwd",
                              subject=f"sub-{subject}",
                              subjects_dir=cfg['mri_root'],
                              tags=('forward-solution', ))

rep_group.save(fname_rep_group % 'html', overwrite=redo, open_browser=False)
