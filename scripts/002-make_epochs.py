import mne
import numpy as np
import os.path as op
import config


drive = config.drive
filt = config.filt
reject = None
exp = 'OLDT'
redo = config.redo
baseline = config.baseline

event_id = {'nonword': 1,
            'word': 2,
            'unprimed': 3,
            'primed': 4,
            'prime': 5,
            'target': 6,
            'alignment': 50,
            'fixation': 99}
# priming = {'unprimed': 3,
#            'primed': 4}
# xca = {'prime': 5,
#        'target': 6}


for subject in config.subjects:
    print config.banner % subject

    exps = [config.subjects[subject][0], config.subjects[subject][2]]
    path = config.drive
    fname_evt = op.join(path, subject, 'mne', subject + '_OLDT-eve.txt')
    fname_coreg_evt = op.join(path, subject, 'mne',
                              subject + '_OLDT_coreg-eve.txt')
    fname_epo = op.join(path, subject, 'mne',
                        subject + '_%s_calm_%s_filt-epo.fif' % (exp, filt))
    # fname_epo_priming = op.join(path, subject, 'mne',
    #                             subject + '_%s_priming_calm_%s_filt-epo.fif'
    #                             % (exp, filt))
    # fname_epo_xca = op.join(path, subject, 'mne',
    #                         subject + '_%s_xca_calm_%s_filt-epo.fif'
    #                         % (exp, filt))
    # fname_epo_coreg = op.join(path, subject, 'mne',
    #                           subject + '_%s_coreg_calm_%s_filt-epo.fif'
    #                           % (exp, filt))

    if not op.exists(fname_epo_coreg) or redo:
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
        raw.preload_data()
        if filt == 'fft':
            raw.filter(.1, 40, method=filt, l_trans_bandwidth=.05)
        else:
            raw.filter(1, 40, method=filt)

        # because of file size
        # priming epochs
        epochs = mne.Epochs(raw, evts, event_id, tmin=-.2, tmax=.6,
                            baseline=baseline, reject=reject)
        epochs.save(fname_epo)
        # # xca: prime and target regions
        # epochs = mne.Epochs(raw, evts, xca, tmin=-.2, tmax=.6,
        #                     baseline=baseline, reject=reject)
        # epochs.save(fname_epo_xca)
        # # coreg
        # epochs = mne.Epochs(raw, coreg_evts, None, tmin=-.2, tmax=.6,
        #                     baseline=baseline, reject=reject)
        # epochs.save(fname_epo_coreg)
