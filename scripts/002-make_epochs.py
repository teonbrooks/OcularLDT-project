import mne
import numpy as np
import os.path as op
import config


redo = True
drive = config.drive
reject = None
baseline = (None, -.1)
event_id = {'nonword': 1,
            'word': 2,
            'unprimed': 3,
            'primed': 4,
            'prime': 5,
            'target': 6,
            'alignment': 50,
            'fixation': 99}

priming = {'unprimed': 3,
           'primed': 4}
ica = {'prime': 5}
target = {'target': 6}

for subject in config.subjects:
    print subject
    exps = config.subjects[subject]
    path = config.drive
    evt_fname = op.join(path, subject, 'mne', subject + '_OLDT-eve.txt')
    coreg_evt_fname = op.join(path, subject, 'mne',
                              subject + '_OLDT_coreg-eve.txt')
    epo_priming_fname = op.join(path, subject, 'mne',
                                subject + '_OLDT_priming_calm_filt-epo.fif')
    epo_xca_fname = op.join(path, subject, 'mne',
                            subject + '_OLDT_xca_calm_filt-epo.fif')
    epo_target_fname = op.join(path, subject, 'mne',
                               subject + '_OLDT_target_calm_filt-epo.fif')
    epo_coreg_fname = op.join(path, subject, 'mne',
                              subject + '_OLDT_coreg_calm_filt-epo.fif')

    if not op.exists(epo_priming_fname) or redo:
        evts = mne.read_events(evt_fname)
        coreg_evts = mne.read_events(coreg_evt_fname)
        raw = config.kit2fiff(subject=subject, exp=exps[0],
                              path=path, preload=False)
        raw2 = config.kit2fiff(subject=subject, exp=exps[2],
                              path=path, preload=False)
        mne.concatenate_raws([raw, raw2])
        raw.info['bads'] = config.bads[subject]
        raw.preload_data()
        raw.filter(.1, 40, method='fft', l_trans_bandwidth=.05)

        # priming
        epochs_priming = mne.Epochs(raw, evts, priming, tmin=-.2, tmax=.6,
                                    baseline=baseline, reject=reject)
        epochs_priming.save(epo_priming_fname)
        # PCA/ICA
        epochs_xca = mne.Epochs(raw, evts, ica, tmin=-.2, tmax=.6,
                                baseline=baseline, reject=reject)
        epochs_xca.save(epo_xca_fname)
        # target region
        epochs_target = mne.Epochs(raw, evts, target, tmin=-.2, tmax=.6,
                                   baseline=baseline, reject=reject)
        epochs_target.save(epo_target_fname)
        # coreg
        epochs_coreg = mne.Epochs(raw, coreg_evts, None, tmin=-.2, tmax=.6,
                                  baseline=baseline, reject=reject)
        epochs_coreg.save(epo_coreg_fname)
