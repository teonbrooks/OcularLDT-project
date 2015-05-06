import os.path as op
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft

import mne
from mne.report import Report


r = Report()
plot_path = op.join(op.dirname(__file__), '..', 'output', 'results',
                    'simulation')
mu, sigma = 400, 50
times = np.linspace(-200, 1200, 1.4e4)
sim = 5 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 / (2 * sigma ** 2))
N400 = sim[::10]
times = np.arange(-200, 1200)
N = 1401

Fs = 1000.
nyq = Fs/2
Fp = .9
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

# FIR using MNE-python
# freq = np.array([0, Fstop, Fp, nyq])
# fir = mne.filter._filter(N400, Fs, freq, gain)


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


#FIR
# filter attentuation
# Compute minimum attenuation at stop frequency
filt_freq, filt_resp = signal.freqz(coef.ravel(), worN=np.pi * freq)
# filt_resp = np.abs(filt_resp)  # use amplitude response
# filt_resp /= np.max(filt_resp)
# filt_resp[np.where(gain == 1)] = 0
# idx = np.argmax(filt_resp)
# att_db = -20 * np.log10(filt_resp[idx])
# att_freq = freq[idx]

# Plot frequency response
fig = plt.figure()
plt.title('Digital filter frequency response')
ax1 = fig.add_subplot(111)

plt.plot(filt_freq, 20 * np.log10(abs(filt_resp)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')

r.add_figs_to_section(fig, 'FIR Filter Frequency response', 'FIR')

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


#Compute different IIR - butterworth filters for comparison
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