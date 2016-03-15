import mne
import numpy as np
import os.path as op
import config
import config_raw


drive = config.drive
filt_type = config.filt[:3]
reject = None
exp = 'OLDT'
redo = config.redo
baseline = None
subjects, experiments = zip(*config_raw.subjects.items())

for subject, experiments in config_raw.subjects.items():
    print config.banner % subject

    exps = [experiments[0], experiments[2]]
    path = config.drive
    fname_raw = op.join(path, subject, 'mne',
                        subject + '_%s_' % exp + 'calm_%s_filt-raw.fif')

    raw = config_raw.kit2fiff(subject=subject, exp=exps[0],
                          path=path, preload=False)
    raw2 = config_raw.kit2fiff(subject=subject, exp=exps[1],
                          path=path, dig=False, preload=False)
    raws = list()
    # we need to go experiment by experiment since some of the channels where
    # saturated
    for ri, exp in zip([raw, raw2], exps):
        ri.load_data()
        ri.plot(block=True, duration=5, n_channels=10,
                highpass=None, lowpass=40)
        print "%s: Bad Chs: %s" % (exp, ri.info['bads'])
        ri.interpolate_bads()
        raws.append(ri)
    raw = mne.concatenate_raws(raws)
    del raws
    # zeroth filtering option
    highpass, lowpass = (.03, 200)
    filt = filt_type + '_hp%s_lp%s' % (highpass, lowpass)
    raw.save(fname_raw % filt, overwrite=redo)
    # first filtering option
    highpass, lowpass = (.51, 40)
    filt = filt_type + '_hp%s_lp%s' % (highpass, lowpass)
    raw.filter(highpass, lowpass, method=filt_type)
    raw.save(fname_raw % filt, overwrite=redo)
    # second filtering option
    highpass, lowpass = (1, 40)
    filt = filt_type + '_hp%s_lp%s' % (highpass, lowpass)
    raw.filter(highpass, lowpass, method=filt_type)
    raw.save(fname_raw % filt, overwrite=redo)
    del raw
