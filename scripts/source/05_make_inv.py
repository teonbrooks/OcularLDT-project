import os.path as op
import json

import mne
from mne.report import Report
from mne_bids import get_entity_vals, BIDSPath


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['task']
derivative = 'inv'

redo = True

subjects_list = get_entity_vals(cfg['bids_root'], entity_key='subject')
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])

fname_rep_group = op.join(cfg['project_path'], 'output', 'reports',
                          f'group_{task}-report.%s')

with mne.open_report(fname_rep_group % 'h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject)

        # Define filenames
        path = op.join(cfg['bids_root'], f"sub-{subject}", 'meg')
        fname_meg = bids_path.update(suffix='meg').fpath
        fname_cov = bids_path.update(suffix='cov').fpath
        fname_fwd = bids_path.update(suffix='fwd').fpath
        fname_inv = bids_path.update(suffix='inv').fpath

        if not op.exists(fname_inv) or redo:
            info = mne.io.read_info(fname_meg)
            cov = mne.read_cov(fname_cov)
            fwd = mne.read_forward_solution(fname_fwd, surf_ori=True)
            # Use surface orientation
            mne.convert_forward_solution(fwd, surf_ori=True, copy=False)
            # Fix the orientation of the dipole, loose by default (0.2)
            inv_op = mne.minimum_norm.make_inverse_operator(info, fwd, cov)
            mne.minimum_norm.write_inverse_operator(fname_inv, inv_op)
        
        rep_group.add_inverse_operator(inverse_operator=inv_op,
                                       title=f"{subject} inv",
                                       tags=('inverse-operator'))

rep_group.save(fname_rep_group % 'html', overwrite=redo, open_browser=False)
