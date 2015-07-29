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

for subject in config.subjects:
    print config.banner % subject

    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    path = config.drive
    fname_raw = op.join(path, subject, 'mne',
                        subject + '_%s_calm_%s_filt-raw.fif' % (exp, filt))

    if redo:
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
        raw.preload_data()
        if filt == 'fft':
            raw.filter(.1, 40, method=filt, l_trans_bandwidth=.05)
        else:
            raw.filter(1, 40, method=filt)

        raw.save(fname_raw)
