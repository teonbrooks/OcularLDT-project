import os.path as op
import json

import numpy as np
from pandas import concat, DataFrame, read_csv
from _reading import Reading

from _recode_events import _recode_events


cfg = json.load(open(op.join('/', 'Users', 'tbrooks', 'codespace',
                             'OcularLDT-project', 'scripts', 'config.json')))

group_ds = list()
fname_stim = op.join(cfg['project_path'], 'input', 'stims',
                     f"{exp_fname}_stimuli_properties.csv")

# Define OLDT interest areas
fname_ia = op.join(cfg['project_path'], 'input', 'stimuli', 'OLDT_ias.txt')
ia_words = ['fixation', 'prime', 'target', 'post']


for subject, experiments in cfg['exp_list'].items():
    print(cfg['banner'] % subject)
    # Define output
    fname_ds = op.join(cfg['project_path'], 'derivatives', f"sub-{subject}",
                       f"{subject}_OLDT_ffd.txt")

    subject_ds = list()
    n_trials = 0
    for ii, exp in enumerate(experiments, 1):
        if exp == 'n/a':
            continue
        # Define filenames
        basename = op.join(cfg['project_path'], f"sub-{subject}", 'eyetrack',
                           f"sub-{subject}_task-{cfg['task']}_run-{ii:2d}")
        fname_log = basename + "_log.dat"
        fname_edf = basename + "_eyetrack.edf"

        # extracting lexical properties
        data = np.loadtxt(fname_log, dtype=str, delimiter='\t')
        data_orig_len = data.shape[0]
        # remove the practice
        idx = data[:, 0] != '0'
        n_practice = sum(~idx)
        data = data[idx]

        # extracting fixation times from the edf file.
        ias = Reading(fname_raw, fname_ia, ia_words)
        assert data_orig_len == ias.shape[0]

        # trial properties
        trials = np.arange(data.shape[0])
        semantics = np.asarray(data[:, 3] == '1', int)
        sem_dict = dict(zip(trials, semantics))

        iters = [('prime', 1), ('target', 2), ('post', 3)]
        for ia, ii in iters:
            times = ias.get_gaze_duration(ia=ii, first_fix=True)
            times['trial'] -= n_practice
            # finally drop practice from em data
            times = times[times['trial'] >= 0]

            # coding semantic priming
            semantics = [sem_dict[x] for x in times['trial']]
            # extracting triggering info from datasource file.
            # prime trigger is index 8, target trigger is index 10
            triggers = list()
            strings = list()
            for _, time in times.iterrows():
                trial = time['trial'].astype(int)
                if ia == 'prime':
                    triggers.append(data[trial, 8])
                    strings.append(data[trial, 7].strip('"'))
                elif ia == 'aux':
                    triggers.append(data[trial, 6])
                    strings.append(data[trial, 5].strip('"'))
                elif ia == 'target':
                    triggers.append(data[trial, 10])
                    strings.append(data[trial, 9].strip('"'))
                elif ia == 'post':
                    triggers.append(data[trial, 12])
                    strings.append(data[trial, 11].strip('"'))

            # coding trigger events
            evts = np.zeros((len(triggers), 3))
            triggers_old = evts[:, 2] = triggers
            evts = _recode_events(exp, evts)[0]

            triggers = evts[:, -1]

            # dummy label
            ia_label = [ia] * len(times)
            subject_label = [subject] * len(times)
            block = np.ones(len(times), int) * ii

            strings = [string.lower() for string in strings]
            columns = list(times.columns)
            columns.extend(['priming', 'string', 'ia', 'trigger',
                            'trigger_old', 'block', 'subject'])
            ds = map(DataFrame, [semantics, strings, ia_label, triggers,
                                 triggers_old, block, subject_label])
            ds.insert(0, times.reset_index(drop=True))
            ds = concat(ds, axis=1)
            ds.columns = columns
            # for concatenation to work properly
            ds.trial += n_trials
            subject_ds.append(ds)
        n_trials = len(data)

    subject_ds = concat(subject_ds)
    subject_ds.to_csv(fname_ds)
