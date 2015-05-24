import mne
import os.path as op
from mne.report import Report
from mne.preprocessing import create_ecg_epochs
import numpy as np

# eog = 1
report = Report()
subject = 'A0129'
exp = 'OLDT2'
decim = 4
n_max_ecg = 3
reject = {'mag': 3e-12}
# report_path = op.join(op.dirname(__file__), '..', 'output', 'results',
#                       subject, 'meg', '%s_%s_ica-report.html' % (subject, exp))
# report_folder = '/Applications/packages/E-MEG/output/results/%s/meg/' % subject
report_folder = '/Users/teon/Google Drive/E-MEG/results/'
report_path = op.join(report_folder, '%s_%s_remove_artifact-report.html'
                      % (subject, exp))
layout = mne.channels.read_layout('/Applications/packages/mne-python/mne/'
                                  'channels/data/layouts/KIT-AD.lout')


path = '/Users/teon/Google Drive/E-MEG/data/%s/mne/' % subject
# path = '/Volumes/GLYPH-1 TB/Experiments/E-MEG/data/%s/mne' % subject
# path = '/Volumes/teon-backup/Experiments/E-MEG/data/%s/mne' % subject
raw_file = op.join(path, '%s_%s_calm_lp40-raw.fif' % (subject, 'OLDT1'))
raw_file2 = op.join(path, '%s_%s_calm_lp40-raw.fif' % (subject, 'OLDT2'))
ica_file = op.join(path, raw_file[:-8] + 'ica.fif')

cleaned_raw = op.join(path, '%s_%s_cleaned-raw.fif' % (subject, 'OLDT'))
event_file = op.join(path, '%s_%s-eve.txt' % (subject, exp))

evts = mne.read_events(event_file)
raw = mne.io.Raw([raw_file, raw_file2], preload=True)
raw.info['bads'] = ['MEG 130']

# Summary plots
epochs = mne.Epochs(raw, evts, None, tmin=-.1, tmax=.5, decim=decim,
                    baseline=(None,0), reject=reject, verbose=False,
                    preload=True)
evoked = epochs.average()
p = evoked.plot(show=False)
report.add_figs_to_section(p, 'Evoked over all events', 'Summary',
                           image_format='png')

cov = mne.compute_covariance(epochs[::10], tmin=-.1, tmax=0, method='auto',
                             verbose=False)
p = cov.plot(epochs.info, show=False)[0]
report.add_figs_to_section(p, 'Covariance Matrix', 'Summary', image_format='png')

p = evoked.plot_white(cov, show=False)
report.add_figs_to_section(p, 'Whitened Evoked over all events', 'Summary',
                           image_format='png')

# plot PCA topos
ep_proj = epochs.crop(-.1, 0, copy=True)
ep_proj = ep_proj[::10]
projs = mne.compute_proj_epochs(ep_proj, n_mag=10)
p = mne.viz.plot_projs_topomap(projs, layout, show=False)
report.add_figs_to_section(p, 'PCA topographies', 'Summary', image_format='png')

# plot evoked - each proj
evokeds = list()
for proj in projs:
    ev = evoked.copy()
    ev.add_proj(proj, remove_existing=True)
    ev.apply_proj()
    evokeds.append(ev)
for i, ev in enumerate(evokeds):
    e = ev.plot(titles={'mag': 'PCA %d' % i}, show=False)
    p = mne.viz.plot_projs_topomap(ev.info['projs'], layout, show=False)
    report.add_figs_to_section([e, p], ['Evoked without PCA %d' %i,
                                   'PCA topography %d' %i],
                          'PCA %d' % i, image_format='png')

report.save(report_path, overwrite=True)
if not 'eog' in locals():
    eog = int(raw_input('eog: '))

# plot evoked - bad projs
bad_projs = [projs[eog]]
bad_projs_idx = eog
evoked.add_proj(bad_projs, remove_existing=True)

p = evoked.plot(titles={'mag': 'PCA %d applied' % bad_projs_idx},
                proj=True, show=False)
raw.add_proj(bad_projs, remove_existing=True)
raw.apply_proj()
report.add_figs_to_section(p, 'Evoked - PCA over all events', 'Summary',
                      image_format='png')

# # ICA
# if op.exists(ica_file):
#     ica = mne.preprocessing.read_ica(ica_file)
# else:
#     # fastica is used to fix the state
#     ica = mne.preprocessing.ICA(.9, random_state=42, method='fastica')
#
# # remove ecg
# title = 'Sources related to %s artifacts (red)'
# picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
#                        stim=False, exclude='bads')
# ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)
# ica.fit(raw, picks=picks, decim=decim, reject=reject)
# ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
# show_picks = np.abs(scores).argsort()[::-1][:5]
# p = ica.plot_sources(raw, show_picks, exclude=ecg_inds, title=title % 'ecg',
#                      show=False)
# report.add_figs_to_section(p, 'Sources related to ECG artifact', 'ECG',
#                            image_format='png')
# p = ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True,
#                         show=False)
# report.add_figs_to_section(p, 'Topos of Sources related to ECG artifact',
#                            'ECG', image_format='png')
# ecg_inds = ecg_inds[:n_max_ecg]
# ica.exclude += ecg_inds
#
# # estimate average artifact
# ecg_evoked = ecg_epochs.average()
# p = ica.plot_sources(ecg_evoked, exclude=ecg_inds)  # plot ECG sources + selection
# report.add_figs_to_section(p, 'Sources before and after ECG artifact', 'ECG',
#                            image_format='png')
# p = ica.plot_overlay(ecg_evoked, exclude=ecg_inds)  # plot ECG cleaning
# report.add_figs_to_section(p, 'Signal before and after ECG artifact', 'ECG',
#                            image_format='png')
# ica.save(ica_file)
# ica.apply(raw)


report.save(report_path)
print cleaned_raw
raw.save(cleaned_raw)












# p = ica.plot_components(show=False)
# report.add_figs_to_section(p, 'Independent Components', 'Summary',
#                            image_format='png')
#
# ics = ica.get_sources(raw)
# # just picked an arbitrary time range
# p = ics.plot(evts, 10, start=100, show=False, scalings={'misc': 4})
# report.add_figs_to_section(p, 'ICs over time', 'Summary', image_format='png')
#
#
# evokeds = list()
# for i in range(ica.n_components_):
#     ica.exclude = [i]
#     evokeds.append(ica.apply(evoked, copy=True))
#
# for i, ev in enumerate(evokeds):
#     e = ev.plot(titles={'mag': 'IC %d' % i}, show=False);
#     p = ica.plot_components(i, show=False);
#     report.add_figs_to_section([e, p], ['Evoked without IC %d' %i, 'IC %d' %i],
#                                'IC %d' %i, image_format='png')
#
# ica.exclude = [1,2]
# ica.save(ica_file)
# ev = ica.apply(evoked, copy=True)
# p = ev.plot(titles={'mag': 'IC %d and %d' % tuple(ica.exclude)}, show=False);
# report.add_figs_to_section(p, 'Evoked-ICA over all events', 'Summary',
#                            image_format='png')
#
# del epochs
# ica.apply(raw)
# epochs = mne.Epochs(raw, evts, None, tmin=-.1, tmax=.5, decim=4,
#                     baseline=(None,0), reject={'mag': 3e-12}, verbose=False)
# evoked = epochs.average()
# cov = mne.compute_covariance(epochs[::10], tmin=-.1, tmax=0, method='auto',
#                              verbose=False)
# p = cov.plot(raw.info, show=False)[0]
# report.add_figs_to_section(p, 'Covariance Matrix After ICA', 'Summary',
#                            image_format='png')
#
# p = evoked.plot_white(cov, show=False)
# report.add_figs_to_section(p, 'Whitened Evoked after ICA over all events',
#                            'Summary', image_format='png')
#


# raw.save(raw.info['filename'][:-8] + '_ica-raw.fif')
