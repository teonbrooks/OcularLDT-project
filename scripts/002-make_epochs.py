import mne
import numpy as np
import os.path as op
import config


drive = config.drive
filt = config.filt
reject = None
exp = 'OLDT'
redo = config.redo
baseline = None


event_id = {'word/prime/unprimed': 1,
            'word/target/unprimed': 2,
            'word/prime/primed': 5,
            'word/target/primed': 6,
            'nonword/prime': 9,
            'nonword/target': 10,
            'fixation': 128}


for subject in config.subjects:
    print config.banner % subject

    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    path = config.drive
    fname_evt = op.join(path, subject, 'mne', subject + '_OLDT-eve.txt')
    fname_coreg_evt = op.join(path, subject, 'mne',
                              subject + '_OLDT_coreg-eve.txt')
    # complete epochs list
    fname_epo = op.join(path, subject, 'mne',
                        subject + '_%s_calm_%s_filt-epo.fif' % (exp, filt))

    if redo:
        evts = mne.read_events(fname_evt)
        coreg_evts = mne.read_events(fname_coreg_evt)
        if 'n/a' in exps:
            exps.pop(exps.index('n/a'))
            raw = config.kit2fiff(subject=subject, exp=exps[0],
                                  path=path, preload=False)
        else:
            raw = config.kit2fiff(subject=subject, exp=exps[0],
                                  path=path, preload=False)
            raw2 = config.kit2fiff(subject=subject, exp=exps[1],
                                  path=path, dig=False, preload=False)
            mne.concatenate_raws([raw, raw2])
        raw.info['bads'] = config.bads[subject]
        raw.load_data()
        if filt == 'fft':
            raw.filter(.1, 40, method=filt, l_trans_bandwidth=.05)
        else:
            raw.filter(1, 40, method=filt)

        # because of file size
        # priming epochs
        epochs = mne.Epochs(raw, evts, event_id, tmin=-.2, tmax=.6,
                            baseline=baseline, reject=reject, verbose=False)
        epochs.save(fname_epo)
        del raw, epochs
