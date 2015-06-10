import mne
import numpy as np
import os.path as op
import config


drive = 'home'
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
    path = config.drives[drive]
    evt_file = op.join(path, subject, 'mne', subject + '_OLDT-eve.txt')
    epo_priming_file = op.join(path, subject, 'mne',
                               subject + '_OLDT_priming_calm_filt-epo.fif')
    epo_ica_file = op.join(path, subject, 'mne',
                           subject + '_OLDT_ica_calm_filt-epo.fif')
    epo_target_file = op.join(path, subject, 'mne',
                              subject + '_OLDT_target_calm_filt-epo.fif')

    if not op.exists(epo_priming_file):
        evts = mne.read_events(evt_file)
        raw = config.kit2fiff(subject=subject, exp=exps[0],
                              path=config.drives[drive], preload=False)
        raw2 = config.kit2fiff(subject=subject, exp=exps[2],
                              path=config.drives[drive], preload=False)
        mne.concatenate_raws([raw, raw2])
        raw.info['bads'] = config.bads[subject]
        raw.preload_data()
        raw.filter(.1, 40, method='fft', l_trans_bandwidth=.05)

        epochs_priming = mne.Epochs(raw, evts, priming, tmin=-.2, tmax=.6,
                                    baseline=baseline, reject=reject)
        epochs_priming.save(epo_priming_file)
        epochs_ica = mne.Epochs(raw, evts, ica, tmin=-.2, tmax=.6,
                                baseline=baseline, reject=reject)
        epochs_ica.save(epo_ica_file)
        epochs_target = mne.Epochs(raw, evts, target, tmin=-.2, tmax=.6,
                                   baseline=baseline, reject=reject)
        epochs_target.save(epo_target_file)
