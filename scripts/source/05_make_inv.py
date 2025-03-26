from pathlib import Path
import tomllib as toml

import mne
from mne_bids import get_entity_vals, BIDSPath


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))

redo = True
task = cfg['task']
derivative = 'inv'

bids_root = root / 'data' / task
subjects_list = get_entity_vals(bids_root, entity_key='subject')
bids_path = BIDSPath(root=bids_root, session=None, task=task,
                     datatype=cfg['datatype'])

basename_rep_group = str(root / 'output' / 'reports', f'group_{task}-report')

with mne.open_report(basename_rep_group + '.h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject, extension='.fif')

        # Define filenames
        fname_meg = bids_path.update(suffix='meg').fpath
        fname_cov = bids_path.update(suffix='cov').fpath
        fname_fwd = bids_path.update(suffix='fwd').fpath
        fname_inv = bids_path.update(suffix='inv').fpath

        if not fname_inv.exists() or redo:
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

rep_group.save(basename_rep_group + '.html', overwrite=True,
               open_browser=False)
