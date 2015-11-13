from glob import glob
import os.path as op
import numpy as np
from pandas import concat, DataFrame
import config
from pyeparse import reading


path = config.drive
redo = config.redo

group_ds = list()
fname_group = op.join(path, 'group', 'group_OLDT_fixation_times.txt')

# Define OLDT interest areas
fname_ia = op.join(path, 'group', 'OLDT_IAs.txt')
ia_words = ['fixation', 'prime', 'target', 'post']

for subject in config.subjects:
    print config.banner % subject
    # Define output
    fname_ds = op.join(path, subject, 'edf',
                       subject + '_OLDT_fixation_times.txt')
    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    if 'n/a' in exps:
        exps.pop(exps.index('n/a'))
    subject_ds = list()
    n_trials = 0
    for exp in exps:
        # Define filenames
        fname_trial = glob(op.join(path, subject, 'edf',
                           '*_{}_*BLOCKTRIAL.dat'.format(exp)))[0]
        fname_raw = op.join(path, subject, 'edf',
                            '{}_{}.edf'.format(subject, exp))

        # extracting lexical properties
        data = np.loadtxt(fname_trial, dtype=str, delimiter='\t')

        # extracting fixation times from the edf file.
        ias = reading.Reading(fname_raw, fname_ia, ia_words)

        # trial properties
        trials = np.arange(data.shape[0])
        semantics = data[:, 3] == '1'
        semantics = dict(zip(trials, semantics))
        # words_idx = data[:, 4].astype(int) > 2
        # words_idy = data[:, 4].astype(int) == 0
        # words = words_idx + words_idy
        words = dict(zip(trials, data[:, 4].astype(int)))

        for ia, ii in [('prime', 1), ('target', 2), ('post', 3)]:
            times = ias.get_gaze_duration(ia=ii)

            # coding semantic priming
            sem_dict = [semantics[x] for x in times['trial']]
            # defining word vs. nonword
            word_dict = [words[x] != ii for x in times['trial']]
            # extracting triggering info from datasource file.
            # prime trigger is index 8, target trigger is index 10
            triggers = list()
            for _, time in times.iterrows():
                trial = time['trial'].astype(int)
                if ia == 'prime':
                    triggers.append(data[trial, 8])
                elif ia == 'target':
                    triggers.append(data[trial, 10])
                else:
                    triggers.append(data[trial, 12])

            # dummy label
            ia_label = [ia] * len(times)
            subject_label = [subject] * len(times)

            columns = list(times.columns)
            columns.extend(['priming', 'word', 'ia', 'trigger', 'subject'])
            ds = map(DataFrame, [sem_dict, word_dict, ia_label,
                                 triggers, subject_label])
            ds.insert(0, times)
            ds = concat(ds, axis=1)
            ds.columns = columns
            # for concatenation to work properly
            ds.trial += n_trials
            subject_ds.append(ds)
        n_trials = len(data)

    subject_ds = concat(subject_ds)
    subject_ds.to_csv(fname_ds)
    group_ds.append(subject_ds)

group_ds = concat(group_ds)
group_ds.to_csv(fname_group)
