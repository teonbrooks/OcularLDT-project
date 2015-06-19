import mne
import os.path as op
from mne.report import Report
import config


layout = mne.channels.read_layout('KIT-AD.lout')
img = config.img
drive = config.drive
exp = 'OLDT'

# define filenames
path = op.join(drive, '%s', 'mne')
ep_fname = op.join(path, '%s_%s_xca_calm_filt-epo.fif')
cov_fname = op.join(path, op.basename(ep_fname)[:10] + '-cov.fif')
proj_fname = op.join(path, '%s_OLDT-proj.fif')
report_fname = op.join(config.results_dir, '%s', '%s_%s_filt_cov-report.html')


for subject in config.subjects:
    print subject

    if not op.exists(cov_fname):
        r = Report()
        epochs = mne.read_epochs(ep_fname % (subject, subject, exp))
        epochs.info['bads'] = config.bads[subject]
        epochs.pick_types(meg=True, exclude='bads')

        # temporary hack
        epochs._raw_times = epochs.times
        epochs._offset = None
        epochs.detrend = None
        epochs.decim = None

        # back to coding
        proj = mne.read_proj(proj_fname % (subject, subject))
        proj = [proj[0]]
        epochs.add_proj(proj)
        epochs.apply_proj()

        # plot evoked
        evoked = epochs.average()
        p = evoked.plot(titles={'mag': 'Evoked Response'}, show=False)
        r.add_figs_to_section(p, 'Evoked Response to Prime Word',
                              'Evoked', image_format=img)

        # plot covariance and whitened evoked
        epochs.crop(-.2, -.1, copy=False)
        cov = mne.compute_covariance(epochs, method='auto', verbose=False)
        p = cov.plot(epochs.info, show_svd=0, show=False)[0]
        comments = ('The covariance matrix is computed on the -200:-100 ms '
                    'baseline. -100:0 ms is confounded with the eye-mvt.')
        r.add_figs_to_section(p, 'Covariance Matrix', 'Covariance',
                              image_format=img, comments=comments)
        p = evoked.plot_white(cov, show=False)
        r.add_figs_to_section(p, 'Whitened Evoked to Prime Word', 'Covariance',
                              image_format=img)

        r.save(report_fname % (subject, subject, exp), overwrite=True,
               open_browser=False)

        # save covariance
        mne.write_cov(cov_fname % (subject, subject, exp), cov)
