import os.path as op
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft

import mne
from mne.report import Report


r = Report()
try:
    file = __file__
except NameError:
    file = '/Applications/packages/E-MEG/output/results/simulation'
plot_path = op.join(op.dirname(file), '..', 'output', 'results',
                    'simulation')
mu, sigma = 400, 50
times = np.linspace(-200, 1200, 1.4e4)
sim = 5 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 / (2 * sigma ** 2))
N400 = sim[::10]
times = np.arange(-200, 1200)
N = 1401

Fs = 1000.
nyq = Fs/2
Fp = 1
trans = .05
Fstop = Fp - trans
gain = [0, 0, 1, 1]
freq = np.array([0, Fstop, Fp, nyq])/nyq
order = 4   # butterworth

# FIR
coef = signal.firwin2(N, freq, gain)
# Make zero-phase filter function
fft_coef = np.abs(fft(coef)).ravel()
N400_ext = np.r_[N400, N400[-1]]
# convolve and return to original length
fir = np.real(ifft(fft_coef * fft(N400_ext))).ravel()[:-1]

# # FIR using MNE-python
# freq = np.array([0, Fstop, Fp, nyq])
# fir_mne = mne.filter._filter(N400, Fs, freq, gain)


#IIR - butterworth
sos = signal.butter(order, Fp/nyq, 'highpass', False, 'sos')
iir = signal.sosfilt(sos, N400)
iir = signal.sosfilt(sos, iir[::-1])[::-1]


# SG Filter
window_length = (int(np.round(Fs / Fp)) // 2) * 2 + 1
sg = signal.savgol_filter(N400, axis=0, polyorder=5,
                          window_length=window_length)
sg = N400 - sg

# Plot Filters against each other
plt.close('all')
fig = plt.figure()
plt.plot(times, N400)
plt.plot(times, fir)
plt.plot(times, iir)
plt.plot(times, sg)
plt.legend(['raw', 'FIR', 'IIR', 'SG'])
plt.title('Simulated Data vs. Different High-Pass Filters')
plt.tight_layout()
r.add_figs_to_section(fig, 'Filter Comparison', 'Summary')

# Compute frequency response to noise
ii = 1000 
nfft = 1e5

psd_noise = list()
psd_fir = list()
psd_iir = list()
psd_sg = list()
for i in range(ii):
    noise = np.random.normal(0, 1, N)
    # Raw noise
    psd_noise.append(signal.welch(noise, nfft=nfft, fs=Fs, nperseg=nfft))
    # FIR with noise signal
    fir = np.real(ifft(fft_coef * fft(noise))).ravel()[:-1]
    psd_fir.append(signal.welch(fir, fs=Fs, nfft=nfft))
    # IIR with noise signal
    iir = signal.sosfilt(sos, noise)
    iir = signal.sosfilt(sos, iir[::-1])[::-1]
    psd_iir.append(signal.welch(iir, fs=Fs, nfft=nfft))
    # SG with noise signal
    sg = signal.savgol_filter(noise, axis=0, polyorder=5,
                              window_length=window_length)
    sg = noise - sg
    psd_sg.append(signal.welch(sg, fs=Fs, nfft=nfft))

psd_noise = np.array(psd_noise).mean(axis=0)
psd_fir = np.array(psd_fir).mean(axis=0)
psd_iir = np.array(psd_iir).mean(axis=0)
psd_sg = np.array(psd_sg).mean(axis=0)

fig = plt.figure()
plt.semilogy(psd_noise[0][:100], psd_noise[1][:100])
plt.semilogy(psd_fir[0][:100], psd_fir[1][:100])
plt.semilogy(psd_iir[0][:100], psd_iir[1][:100])
plt.semilogy(psd_sg[0][:100], psd_sg[1][:100])
plt.legend(['noise', 'FIR', 'IIR', 'SG'])
r.add_figs_to_section(fig, 'Filter Frequency response to Gaussian Noise: 0-1Hz',
                      'Summary')

fig = plt.figure()
plt.semilogy(psd_noise[0][:1000], psd_noise[1][:1000])
plt.semilogy(psd_fir[0][:1000], psd_fir[1][:1000])
plt.semilogy(psd_iir[0][:1000], psd_iir[1][:1000])
plt.semilogy(psd_sg[0][:1000], psd_sg[1][:1000])
plt.legend(['noise', 'FIR', 'IIR', 'SG'])
r.add_figs_to_section(fig, 'Filter Frequency response to Gaussian Noise: 0-10Hz',
                      'Summary')

# fig = plt.figure()
# plt.semilogy(*psd_noise)
# plt.semilogy(*psd_fir)
# plt.semilogy(*psd_iir)
# plt.semilogy(*psd_sg)
# plt.legend(['noise', 'FIR', 'IIR', 'SG'])
# r.add_figs_to_section(fig, 'Full Filter Frequency response to Gaussian Noise',
#                       'Summary')


# Compute different FIR filters for comparison
trans = [.001, .002, .003, .004, .005]
Fps = [.01, .03, 0.05, 0.07, .1]
for t in trans:
    plt.close('all')
    fig = plt.figure()
    firs = list()
    for Fp in Fps:
        Fstop = Fp - t
        gain = [0, 0, 1, 1]
        freq = np.array([0, Fstop, Fp, nyq])/nyq
        coef = signal.firwin2(N, freq, gain)
        # Make zero-phase filter function
        fft_coef = np.abs(fft(coef)).ravel()
        N400_ext = np.r_[N400, N400[-1]]
        # convolve and return to original length
        fir = np.real(ifft(fft_coef * fft(N400_ext))).ravel()[:-1]
        firs.append(fir)

    plt.plot(times, N400)
    for fir in firs:
        plt.plot(times, fir)
    plt.legend(['raw'] + Fps)
    plt.title('Simulated Data vs. Different FIR Filters: Trans %s' % t)
    plt.tight_layout()
    r.add_figs_to_section(fig, 'FIR Comparison: Trans=%s' % t, 'FIR')


# Compute different IIR - butterworth filters for comparison
order = [1, 2, 3, 4]
Fps = [.01, .03, 0.05, 0.07, .1]
for o in order:
    plt.close('all')
    fig = plt.figure()    
    iirs = list()
    for Fp in Fps: 
        sos = signal.butter(o, Fp/nyq, 'highpass', False, 'sos')
        iir = signal.sosfilt(sos, N400)
        iir = signal.sosfilt(sos, iir[::-1])[::-1]
        iirs.append(iir)

    plt.plot(times, N400)
    for iir in iirs:
        plt.plot(times, iir)
    plt.legend(['raw'] + Fps)
    plt.title('Simulated Data vs. Different IIR Filters: Order %d' % o)
    plt.tight_layout()
    r.add_figs_to_section(fig, 'IIR Comparison: Order=%s' % o, 'IIR')
    plt.close('all')

r.save(op.join(plot_path, 'Filter_Simulation_report.html'), overwrite=True)
