from pathlib import Path
import tomllib as toml

import mne
from mne.stats import linear_regression_raw
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


parents = list(Path(__file__).resolve().parents)
root = [path for path in parents if str(path).endswith('OcularLDT-project')][0]
cfg = toml.load(open(root / 'config.toml', 'rb'))

task = cfg['task']
bids_path = BIDSPath(root=cfg['bids_root'], session=None, task=task,
                     datatype=cfg['datatype'])
decim = 2
tmin, tmax = -.2, 1

c_names = ['word/target/unprimed', 'word/target/primed']
event_id = {c_name: cfg['event_id'][c_name] for c_name in c_names}

bids_root = root / 'data' / 'OcularLDT'
subjects = get_entity_vals(bids_root, entity_key='subject')

for subject in subjects[:1]:
    bids_path.update(subject=subject)
    print(cfg['banner'] % subject)
    # define filenames
    fname_raw = bids_path.update(suffix='meg').fpath
    fname_ica = bids_path.update(suffix='ica').fpath
    fname_erf_uncorrected = bids_path.update(suffix='priming_uncorrected_ave').fpath
    fname_erf_corrected = bids_path.update(suffix='priming_corrected_ave').fpath

    # select only meg channels
    raw = read_raw_bids(bids_path).pick_types(meg=True).load_data()
    events, total_event_id = mne.events_from_annotations(raw)

    # compute an ERF for the priming conditions
    epochs_uncorrected = mne.Epochs(raw, events, event_id, baseline = None,
                                    tmin=tmin, tmax=tmax)
    epochs_uncorrected.equalize_event_counts(c_names)
    evokeds_uncorrected = epochs_uncorrected.average(by_event_type=True)
    mne.write_evokeds(fname_erf_uncorrected, evokeds_uncorrected)
    del evokeds_uncorrected

    # now apply ICA
    ica = mne.preprocessing.read_ica(fname_ica)
    epochs_corrected = ica.apply(epochs_uncorrected.load_data())
    evokeds_corrected = epochs_corrected.average(by_event_type=True)
    mne.write_evokeds(fname_erf_uncorrected, evokeds_corrected)
    del evokeds_corrected
