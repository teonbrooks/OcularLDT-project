"""
Writing the co-registration event file
--------------------------------------
The MEG and EM files must be aligned to have a proper decoding.
This arranges the triggers in events of interests: prime, target.
These events are consistent and identifiable in both data types.

They are selected and rearranged according to timestamps. This allows
for ease of matching later on.

"""

import os.path as op
import json

import pandas as pd


cfg = json.load(open(op.join('/', 'Users', 'teonbrooks', 'codespace',
                     'OcularLDT-project', 'scripts', 'config.json')))
task = cfg['project_name']
task = cfg['task']
datatype = 'eyetrack'
derivative = 'log'

for subject, experiments in cfg['exp_list'].items():
    print(cfg['banner'] % subject)
    basename = op.join(cfg['bids_root'], f"sub-{subject}", datatype)
    fname_coreg = op.join(basename, f"sub-{subject}_task-{task}" + 
                          f"_coreg-{datatype}_log.tsv")
    subject_ds = list()
    for ii, exp in enumerate(experiments, 1):
        if exp == 'n/a':
            continue

        fname_template = op.join(basename,
                                f"sub-{subject}_task-{task}_run-{ii:02d}")
        fname_et = fname_template + f"_{derivative}.tsv"

        df = pd.read_csv(fname_et, sep='\t')
        # drop practice trials
        df = df[df['trial_no'] >= 10]
        # correct trial no
        # for combining the different runs, since each run is 240 trials
        # add 240 avoid trial_no collision
        df['trial_no'] -= 10
        if ii > 1:
            df['trial_no'] += 240
        grouping = ['trial_no', 'fix_pos']
        # find the first boundary
        idx =  df['metric'] == 'boundary'
        df_boundary = df[idx].groupby(grouping, as_index=False)['stime']  \
                             .first()
        # find triggers if available
        idx =  df['metric'] == 'ttl'
        df_trig = df[idx].groupby(grouping, as_index=False)['trigger'].first()

        # we need trialno, fix_pos, ttl, trigger, and first_fix
        idx = df['is_gaze'] == True
        df_ffd_stime = df[idx].groupby(grouping, as_index=False)['stime']  \
                              .first()
        df_fix = df[idx].groupby(grouping, as_index=False)['dur']  \
                        .agg([('ffd', 'first'), ('gzd', 'sum')])
        df_dm = df_ffd_stime.merge(df_fix, how='right', on=grouping)
        # for when triggers are missing
        if df_trig.size:
            df_dm = df_dm.merge(df_trig, how='right', on=grouping)
        df_dm = df_dm.merge(df_boundary, how='right', on=grouping,
                            suffixes=('_fix', '_boundary'))
        reordered = ['trial_no', 'fix_pos', 'trigger', 'ffd', 'gzd',
                     'stime_boundary', 'stime_fix']
        if 'trigger' not in df_dm.columns:
            reordered.pop(reordered.index('trigger'))
        else:
             df_dm['trigger'] = df_dm['trigger'].fillna(0).astype(int)
        df_dm = df_dm[reordered]

        subject_ds.append(df_dm)
    subject_ds = pd.concat(subject_ds)
    subject_ds['stime_boundary'] = subject_ds['stime_boundary'].round(3)
    subject_ds['stime_fix'] = subject_ds['stime_fix'].round(3)
    subject_ds.to_csv(fname_coreg, sep='\t', index=False)
