from pathlib import Path
import tomllib as toml
import os.path as op

import mne
from mne_bids import BIDSPath, get_entity_vals


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml' , 'rb'))

task = cfg['task']
derivative = 'fwd'

redo = True

bids_root = root / 'data' / task
mri_root = bids_root + '_MRI'
subjects_list = get_entity_vals(bids_root, entity_key='subject')
bids_path = BIDSPath(root=bids_root, session=None, task=task,
                     datatype=cfg['datatype'])

basename_rep_group = str(root / 'output' / 'reports', f'group_{task}-report')

with mne.open_report(basename_rep_group + '.h5') as rep_group:
    for subject in subjects_list:
        print(cfg['banner'] % subject)
        bids_path.update(subject=subject, suffix='meg')

        # define filenames
        fname_fwd = bids_path.update(suffix='fwd', extension='.fif').fpath
        fname_trans = bids_path.update(suffix='trans', extension='.fif').fpath
        # TODO: update bem and src to use BIDSPath
        bem_sol = (mri_root / f"sub-{subject}" / 'bem' /
                   f'sub-{subject}-inner_skull-bem-sol.fif')
        fname_src = op.join(cfg['mri_root'], f"sub-{subject}", 'bem',
                            f'sub-{subject}-ico-4-src.fif')
        if not fname_fwd.exists() or redo:
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

rep_group.save(basename_rep_group + '.html', overwrite=True,
               open_browser=False)
