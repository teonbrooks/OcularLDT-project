import os.path as op
import shutil as sh
from glob import glob

import config


for subject, experiments in config.exp_list.items():
    print(config.banner % subject)

    for ii, exp in enumerate(experiments, 1):
        fname_pat = op.join(config.raw_root, subject, 'edf',
                            f"actual_TRIAL_DataSource_{exp}_*")
        bids_log = op.join(config.bids_root, f"sub-{subject}", 'eyetrack',
                           (f"sub-{subject}_task-{config.project_name}_" +
                            f"run-{ii:02d}_log.dat"))
        fname_log = glob(fname_pat)
        if len(fname_log) == 1:
            sh.copyfile(fname_log[0], bids_log)
