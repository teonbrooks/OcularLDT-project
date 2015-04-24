import numpy as np
import scipy as sp
import mne
from matplotlib import pyplot as plt
import os.path as op



plot_path = op.join(op.dirname(__file__), '..', 'output', 'sim_results')
mu, sigma = 400, 50
times = np.linspace(-200, 2000, 1.2e4)
sim = 5 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 / (2 * sigma ** 2))
N400 = sim[::10]
times = np.arange(-200, 1000)
N = 1001

Fs = 1000.
nyq = Fs/2
Fp = .1
trans = .05
Fstop = Fp - trans
gain = [0, 0, 1, 1]
freq = np.array([0, Fstop, Fp, nyq])/(nyq)
order = 4   # butterworth

# FIR
taps = sp.signal.firwin2(N, freq, gain)
fir = sp.signal.filtfilt(taps, 1, N400, padtype=None)

#IIR - butterworth
sos = sp.signal.butter(order, Fp/nyq, 'highpass', False, 'sos')
iir = sp.signal.sosfilt(sos, N400)
# filt = mne.filter._filter(N400, Fs, freq, gain)

sos = sp.signal.butter(order, Fp/nyq, 'highpass', False, 'sos')
iir = sp.signal.sosfilt(sos, N400)

plt.plot(times, N400)
plt.plot(times, fir)
plt.plot(times, iir)
plt.legend(['raw', 'FIR', 'IIR'])
plt.title('Simulated Data vs. Different High-Pass Filters')
plt.tight_layout()
plt.savefig(op.join(plot_path, 'sim_filter_comparison.svg'))
plt.close('all')

#IIR - butterworth
iirs = list()
Fps = [.01, .03, 0.05, 0.07, .1]
for Fp in Fps: 
    sos = sp.signal.butter(order, Fp/nyq, 'highpass', False, 'sos')
    iirs.append(sp.signal.sosfilt(sos, N400))

plt.plot(times, N400)
for iir in iirs:
    plt.plot(times, iir)
plt.legend(['raw'] + Fps)
plt.title('Simulated Data vs. Different IIR Filters')
plt.tight_layout()
plt.savefig(op.join(plot_path, 'sim_filter_iir_comparison.svg'))
plt.close('all')