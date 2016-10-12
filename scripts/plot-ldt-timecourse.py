import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked

data_path = sample.data_path()

raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg-proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(ave_fname)

label_names = ['Aud-lh', 'Aud-rh']
labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
          for ln in label_names]

baseline = 200
ldt = 1000
sfreq = 1000
epoch = ldt + baseline
times = (np.arange(epoch).astype(float) - baseline) / sfreq

def data_fun(times):
    epoch = len(times)
    baseline = 200
    cycles = 5
    sfreq = 1000


    sim_times = np.arange(epoch - baseline)
    y = np.sin(2*np.pi*cycles*sim_times/1000)
    # effect is half a cycle
    sim_effect = y[:sfreq/(cycles*2)]

    tc = .5 * np.ones(epoch) + np.random.rand(epoch) * .01
    # FFD roughly 70-170 ms
    beg, end = 70 + baseline, 170 + baseline
    amp = .04
    tc[beg:end] += sim_effect * amp

    # Semantic Priming roughly 400-500 ms
    beg, end = 400 + baseline, 500 + baseline
    amp = .08
    tc[beg:end] += sim_effect * amp

    return tc
    
stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                          random_state=42, labels=labels, data_fun=data_fun)

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=180)[1]
snr = 6.  # dB
evoked = simulate_evoked(fwd, stc, info, cov, snr, iir_filter=iir_filter)
