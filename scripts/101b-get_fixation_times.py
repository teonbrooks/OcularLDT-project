from glob import glob
import os.path as op
import numpy as np
from pandas import concat, DataFrame, read_csv
import config
import config_raw
from pyeparse import reading
from _recode_events import _recode_events


path = config.drive
redo = config.redo
exp_fname = config.exp

group_ds = list()
fname_group = op.join(path, 'group', 'group_%s_fixation_times.txt' % exp_fname)
fname_stim = op.join(config.drives['project'], 'input', 'stims',
                     '%s_stimuli_properties.csv' % exp_fname)

# Define OLDT interest areas
fname_ia = op.join(path, 'group', '%s_IAs.txt' % exp_fname)
ia_words = ['fixation', 'prime', 'target', 'post']
# lexical properties
# header: "Occurences","Word","Length","Freq_HAL","Log_Freq_HAL","BG_Mean"
lex_props = read_csv(fname_stim, dtype=str, delimiter=',')

# type-cast
lex_props['BG_Mean'] = map(float, lex_props['BG_Mean'])
lex_props['Log_Freq_HAL'] = map(float, lex_props['Log_Freq_HAL'])
# make a useable dict
lex_bg = dict(zip(lex_props['Word'], lex_props['BG_Mean']))
lex_freq = dict(zip(lex_props['Word'], lex_props['Log_Freq_HAL']))

for subject, experiments in config_raw.subjects.items():
    print config.banner % subject
    # Define output
    fname_ds = op.join(path, subject, 'edf',
                       subject + '_%s_fixation_times.txt' % exp_fname)
    exps = [experiments[0], experiments[2]]
    if 'n/a' in exps:
        exps.pop(exps.index('n/a'))
    subject_ds = list()
    n_trials = 0
    for ii, exp in enumerate(exps):
        # Define filenames
        fname_trial = glob(op.join(path, subject, 'edf',
                           '*_{}_*BLOCKTRIAL.dat'.format(exp)))[0]
        fname_raw = op.join(path, subject, 'edf',
                            '{}_{}.edf'.format(subject, exp))

        # extracting lexical properties
        data = np.loadtxt(fname_trial, dtype=str, delimiter='\t')
        data_orig_len = data.shape[0]
        # remove the practice
        idx = data[:, 0] != '0'
        n_practice = sum(~idx)
        data = data[idx]

        # extracting fixation times from the edf file.
        ias = reading.Reading(fname_raw, fname_ia, ia_words)
        assert data_orig_len == ias.shape[0]

        # trial properties
        trials = np.arange(data.shape[0])
        semantics = np.asarray(data[:, 3] == '1', int)
        sem_dict = dict(zip(trials, semantics))
        word_dict = dict(zip(trials, data[:, 4].astype(int)))

        for ia, ii in [('prime', 1), ('target', 2), ('post', 3)]:
            times = ias.get_gaze_duration(ia=ii, first_fix=True)
            times['trial'] -= n_practice
            # finally drop practice from em data
            times = times[times['trial'] >= 0]

            # coding semantic priming
            semantics = [sem_dict[x] for x in times['trial']]
            # defining word vs. nonword
            words = [int(word_dict[x] != ii) for x in times['trial']]
            # extracting triggering info from datasource file.
            # prime trigger is index 8, target trigger is index 10
            triggers = list()
            strings = list()
            for _, time in times.iterrows():
                trial = time['trial'].astype(int)
                if ia == 'prime':
                    triggers.append(data[trial, 8])
                    strings.append(data[trial, 7].strip('"'))
                elif ia == 'target':
                    triggers.append(data[trial, 10])
                    strings.append(data[trial, 9].strip('"'))
                else:
                    triggers.append(data[trial, 12])
                    strings.append(data[trial, 11].strip('"'))

            bg_means = [lex_bg[string] if string in lex_bg else np.nan
                        for string in strings]
            log_freq = [lex_freq[string] if string in lex_freq else np.nan
                        for string in strings]

            # coding trigger events
            evts = np.zeros((len(triggers), 3))
            triggers_old = evts[:, 2] = triggers
            evts, fix_idx, primes_idx, targets_idx, \
                semantic_idx, nonwords_idx = _recode_events(exp, evts)
            triggers = evts[:, -1]

            # dummy label
            ia_label = [ia] * len(times)
            subject_label = [subject] * len(times)
            block = np.ones(len(times), int) * ii

            columns = list(times.columns)
            columns.extend(['priming', 'word', 'string', 'bg_mean', 'log_freq',
                            'ia', 'trigger', 'trigger_old', 'block', 'subject'])
            ds = map(DataFrame, [semantics, words, strings, bg_means, log_freq,
                                 ia_label, triggers, triggers_old,
                                 block, subject_label])
            ds.insert(0, times.reset_index(drop=True))
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
