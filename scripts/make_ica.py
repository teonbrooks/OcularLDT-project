import mne
import os.path as op


path = '/Volumes/teon-backup/Experiments/E-MEG/data/A0148/mne/'
plot_path = '/Applications/packages/E-MEG/output/results/A0148/meg'
raw_file = 'A0148_OLDT_calm_lp40-raw.fif'
raw = mne.io.Raw(op.join(path, raw_file), verbose=False, preload=True)
ica = mne.preprocessing.ICA(.9)
ica.fit(raw)

p = i.plot_components(show=False)
p[0].savefig(op.join(plot_path, 'A0148_OLDT_ica_components.pdf'))
i.plot_overlay(raw, [0])
p.savefig(op.join(plot_path, 'A0148_OLDT_ica_overlay.pdf'))
ica.exclude = [0]
ica.save(raw.info['filename'][:-7] + 'ica.fif')

r = ica.apply(raw)
r.save(raw.info['filename'][:-8] + 'ica-raw.fif')
