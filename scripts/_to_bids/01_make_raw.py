import mne
import os.path as op
import config


redo = config.redo
baseline = None

for subject, experiments in config.exp_list.items():
    print(config.banner % subject)

    path = config.raw_root
    fname_raw = op.join(config.bids_root, subject, 'mne',
                        f'{subject}_OLDT_calm_%s_filt-raw.fif')

    raw_fifs = list()
    for exp in experiments:
        raw = config.kit2fiff(subject=subject, exp=exp,
                              path=path, preload=False)
        raw_fifs.append(raw)
    raws = list()
    # we need to go experiment by experiment since some of the channels where
    # saturated

    for ri, exp in zip(raw_fifs, experiments):
        ri.load_data()
        ri.plot(block=True, duration=5, n_channels=10,
                highpass=None, lowpass=40)
        print(f"{exp}: Bad Chs: {ri.info['bads']}")
        ri.interpolate_bads()
        raws.append(ri)
    raw = mne.concatenate_raws(raws)
    del raw_fifs, raws

    # bandpass filtering
    highpass, lowpass = (.51, 40)
    filt_type = 'iir'
    filt = f'{filt_type}_hp{highpass}_lp{lowpass}'
    raw.filter(highpass, lowpass, method=filt_type)
    raw.save(fname_raw % filt, overwrite=redo)
