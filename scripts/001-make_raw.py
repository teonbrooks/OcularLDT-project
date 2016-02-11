import mne
import numpy as np
import os.path as op
import config


drive = config.drive
filt_type = config.filt[:3]
reject = None
exp = 'OLDT'
redo = config.redo
baseline = None

for subject in config.subjects:
    print config.banner % subject

    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    path = config.drive
    fname_raw = op.join(path, subject, 'mne',
                        subject + '_%s_' % exp + 'calm_%s_filt-raw.fif')

    raw = config.kit2fiff(subject=subject, exp=exps[0],
                          path=path, preload=False)
    raw2 = config.kit2fiff(subject=subject, exp=exps[1],
                          path=path, dig=False, preload=False)
    raws = list()
    # we need to go experiment by experiment since some of the channels where
    # saturated
    for ri, exp in zip([raw, raw2], exps):
        ri.load_data()
        ri.plot(block=True, duration=5, n_channels=10,
                highpass=None, lowpass=40)
        print "%s: Bad Chs: %s" % (exp, raw.info['bads'])
    #     ri.interpolate_bads()
    #     raws.append(ri)
    # raw = mne.concatenate_raws(raws)[0]
    # # zeroth filtering option
    # highpass, lowpass = (None, 40)
    # filt = 'iir_hp%s_lp%s' % (highpass, lowpass)
    # raw.save(fname_raw % filt, overwrite=redo)
    # raw_filt = raw.copy()
    # # first filtering option
    # highpass, lowpass = (1, 40)
    # filt = 'iir_hp%s_lp%s' % (highpass, lowpass)
    # raw_filt.filter(highpass, lowpass, method=filt_type)
    # raw_filt.save(fname_raw % filt, overwrite=redo)
    # del raw_filt
    # # second filtering option
    # highpass, lowpass = (.51, 40)
    # filt = 'iir_hp%s_lp%s' % (highpass, lowpass)
    # raw.filter(highpass, lowpass, method=filt_type)
    # raw.save(fname_raw % filt, overwrite=redo)
    del raw, raws
